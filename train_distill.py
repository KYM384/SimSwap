#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: train.py
# Created Date: Monday December 27th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 22nd April 2022 10:49:26 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.backends import cudnn

from data.data_loader_Swapping import GetLoader
from models.mobile import Mobile_fsModel
from util import util
from util.plot import plot_batch
from train import str2bool, TrainOptions


if __name__ == '__main__':

    opt = TrainOptions()
    opt.initialize()
    opt.parser.add_argument('--teacher_path', type=str, default='path_to_teacher_model')
    opt.initialized = True
    opt = opt.parse()

    iter_path   = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    
    log_path = os.path.join(opt.checkpoints_dir, opt.name, 'summary')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    print("GPU used : ", str(opt.gpu_ids))

    
    cudnn.benchmark = True

    wandb.init(project="DriftGAN", name="SimSwap distill", config=opt)

    

    model = Mobile_fsModel()

    model.initialize(opt)

    #####################################################

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)

    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

    loss_avg        = 0
    refresh_count   = 0
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)

    train_loader    = GetLoader(opt.dataset,opt.batchSize,8,1234)

    randindex = [i for i in range(opt.batchSize)]
    random.shuffle(randindex)

    if not opt.continue_train:
        start   = 0
    else:
        start   = int(opt.which_epoch)
    total_step  = opt.total_step
    import datetime
    print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    from util.logo_class import logo_class
    logo_class.print_start_training()
    model.netD.feature_network.requires_grad_(False)

    # Training Cycle
    for step in range(start, total_step):
        with torch.autocast("cuda", torch.bfloat16):
            model.netG.train()
            for interval in range(2):
                random.shuffle(randindex)
                src_image1, src_image2  = train_loader.next()
                
                if step%2 == 0:
                    img_id = src_image2
                else:
                    img_id = src_image2[randindex]

                img_id_112      = F.interpolate(img_id,size=(112,112), mode='bicubic')
                latent_id       = model.netArc(img_id_112)
                latent_id       = F.normalize(latent_id, p=2, dim=1)
                if interval:
                    with torch.no_grad():
                        img_teacher = model.netG_teacher(src_image1, latent_id)
                    
                    img_fake        = model.netG(src_image1, latent_id)
                    gen_logits,_    = model.netD(img_fake.detach(), None)
                    loss_Dgen       = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                    real_logits,_   = model.netD(img_teacher,None)
                    loss_Dreal      = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                    loss_D          = loss_Dgen + loss_Dreal
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()
                else:
                    with torch.no_grad():
                        img_teacher = model.netG_teacher(src_image1, latent_id)
                    
                    # model.netD.requires_grad_(True)
                    img_fake        = model.netG(src_image1, latent_id)
                    # G loss
                    gen_logits,feat = model.netD(img_fake, None)
                    
                    loss_Gmain      = (-gen_logits).mean()
                    img_fake_down   = F.interpolate(img_fake, size=(112,112), mode='bicubic')
                    latent_fake     = model.netArc(img_fake_down)
                    latent_fake     = F.normalize(latent_fake, p=2, dim=1)
                    loss_G_ID       = (1 - model.cosin_metric(latent_fake, latent_id)).mean()
                    real_feat       = model.netD.get_feature(img_teacher)
                    feat_match_loss = model.criterionFeat(feat["3"],real_feat["3"]) 
                    loss_G          = loss_Gmain + loss_G_ID * opt.lambda_id + feat_match_loss * opt.lambda_feat
                    

                    if step%2 == 0:
                        #G_Rec
                        loss_G_Rec  = model.criterionRec(img_fake, img_teacher) * opt.lambda_rec
                        loss_G      += loss_G_Rec

                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()
                    

        ############## Display results and errors ##########
        ### print out errors
        # Print out log info
        if (step + 1) % opt.log_frep == 0:
            # errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            errors = {
                "G_Loss":loss_Gmain.item(),
                "G_ID":loss_G_ID.item(),
                "G_Rec":loss_G_Rec.item(),
                "G_feat_match":feat_match_loss.item(),
                "D_fake":loss_Dgen.item(),
                "D_real":loss_Dreal.item(),
                "D_loss":loss_D.item()
            }
            wandb.log(errors, step=step+1)

            message = '( step: %d, ) ' % (step)
            for k, v in errors.items():
                message += '%s: %.3f ' % (k, v)

            print(message)
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)

        ### display output images
        if (step + 1) % opt.sample_freq == 0 or step == 0:
            model.netG.eval()
            with torch.no_grad():
                imgs        = list()
                zero_img    = (torch.zeros_like(src_image1[0,...]))
                imgs.append(zero_img.cpu().numpy())
                save_img    = ((src_image1.cpu())* imagenet_std + imagenet_mean).numpy()
                for r in range(opt.batchSize):
                    imgs.append(save_img[r,...])
                arcface_112     = F.interpolate(src_image2,size=(112,112), mode='bicubic')
                id_vector_src1  = model.netArc(arcface_112)
                id_vector_src1  = F.normalize(id_vector_src1, p=2, dim=1)

                for i in range(opt.batchSize):
                    
                    imgs.append(save_img[i,...])
                    image_infer = src_image1[i, ...].repeat(opt.batchSize, 1, 1, 1)
                    img_fake    = model.netG(image_infer, id_vector_src1).cpu()
                    
                    img_fake    = img_fake * imagenet_std
                    img_fake    = img_fake + imagenet_mean
                    img_fake    = img_fake.numpy()
                    for j in range(opt.batchSize):
                        imgs.append(img_fake[j,...])
                print("Save test data")
                imgs = np.stack(imgs, axis = 0).transpose(0,2,3,1)
                plot_batch(imgs, os.path.join(sample_path, 'step_'+str(step+1)+'.jpg'))

                wandb.log({"samples":[wandb.Image(os.path.join(sample_path, 'step_'+str(step+1)+'.jpg'))]}, step=step+1)

        ### save latest model
        if (step+1) % opt.model_freq==0 or step == 0:
            print('saving the latest model (steps %d)' % (step+1))
            model.save(step+1)            
            np.savetxt(iter_path, (step+1, total_step), delimiter=',', fmt='%d')

    wandb.finish()