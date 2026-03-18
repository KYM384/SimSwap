"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fs_networks_fix import InstanceNorm, ApplyStyle, Generator_Adain_Upsample
from pg_modules.projected_discriminator import ProjectedDiscriminator
from .projected_model import fsModel


class MobileConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super(MobileConv2d, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        )

class MobileResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(MobileResnetBlock_Adain, self).__init__()
        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [MobileConv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [MobileConv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)


    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out



class MobileGenerator_Adain_Upsample(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, n_blocks=6, deep=False,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MobileGenerator_Adain_Upsample, self).__init__()

        activation = nn.ReLU(True)
        
        self.deep = deep
        
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), MobileConv2d(input_nc, 64, kernel_size=7, padding=0),
                                         norm_layer(64), activation)
        ### downsample
        self.down1 = nn.Sequential(MobileConv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   norm_layer(128), activation)
        self.down2 = nn.Sequential(MobileConv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   norm_layer(256), activation)
        self.down3 = nn.Sequential(MobileConv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   norm_layer(512), activation)
                                   
        if self.deep:
            self.down4 = nn.Sequential(MobileConv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                       norm_layer(512), activation)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                MobileResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        if self.deep:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                MobileConv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512), activation
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            MobileConv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            MobileConv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            MobileConv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation
        )
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), MobileConv2d(64, output_nc, kernel_size=7, padding=0))

    def forward(self, input, dlatents):
        x = input  # 3*224*224

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)
        bot = []
        bot.append(x)
        features = []
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)
            bot.append(x)

        if self.deep:
            x = self.up4(x)
            features.append(x)
        x = self.up3(x)
        features.append(x)
        x = self.up2(x)
        features.append(x)
        x = self.up1(x)
        features.append(x)
        x = self.last_layer(x)

        return x


class Mobile_fsModel(fsModel):
    def name(self):
        return 'Mobile_fsModel'

    def initialize(self, opt):
        assert opt.isTrain
        assert not opt.continue_train

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        device = torch.device("cuda:0")

        # Generator network
        self.netG = MobileGenerator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=6, deep=False)
        self.netG.to(device)

        self.netG_teacher = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
        self.netG_teacher.to(device).eval()
        self.netG_teacher.load_state_dict(torch.load(opt.teacher_path, map_location=device, weights_only=False))

        # Id network
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"), weights_only=False)
        self.netArc = netArc_checkpoint
        self.netArc = self.netArc.cuda()
        self.netArc.eval()
        self.netArc.requires_grad_(False)

        self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
        # self.netD.feature_network.requires_grad_(False)
        self.netD.cuda()


        # define loss functions
        self.criterionFeat  = nn.L1Loss()
        self.criterionRec   = nn.L1Loss()

        # optimizer G
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)

        # optimizer D
        params = list(self.netD.parameters())
        self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)

        torch.cuda.empty_cache()
