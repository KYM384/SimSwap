import numpy as np
import torch
import torchvision
from imutils import face_utils
from ultralytics import YOLO
from tqdm import tqdm
import argparse
import dlib
import copy
import timm
import cv2
import os

from models.mobile import MobileGenerator_Adain_Upsample


class MobileNetV3MultiTask(torch.nn.Module):
    def __init__(self, model_name: str, num_age_classes: int, num_gender_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,   # pooled feature を返す
            global_pool="avg",
        )
        feat_dim = 1024 #self.backbone.head_hidden_size

        self.dropout = torch.nn.Dropout(p=0.2)
        self.age_head = torch.nn.Linear(feat_dim, num_age_classes)
        self.gender_head = torch.nn.Linear(feat_dim, num_gender_classes)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        age_logits = self.age_head(feat)
        gender_logits = self.gender_head(feat)
        return age_logits, gender_logits


class Face:
    def __init__(self, keypoint: list[tuple[int, int]]):
        self.keypoint = keypoint

        e0, e1, n, m = keypoint
        x_ = e1 - e0
        y_ = 0.5 * (e0 + e1) - m
        c = 0.5 * (e0 + e1) - 0.1 * y_
        cx, cy = int(c[0]), int(c[1])

        theta = np.arctan2(x_[1], x_[0])

        s = max(4.0 * np.linalg.norm(x_), 3.6 * np.linalg.norm(y_))
        s = int(s)

        # bbox: (x, y, w, h)
        self.bbox = (cx-s//2, cy-s//2, s, s)
        self.theta = theta

    def get_center(self):
        return self.bbox[0] + self.bbox[2] // 2, self.bbox[1] + self.bbox[3] // 2

    def get_size(self):
        return self.bbox[2]

    def set_attributes(self, age: int, gender: str):
        self.age = age
        self.gender = gender

    def update(self, keypoint: list[tuple[int, int]]):
        self.__init__(keypoint)

    def calc_iou(self, other) -> float:
        x1 = max(self.bbox[0], other.bbox[0])
        y1 = max(self.bbox[1], other.bbox[1])
        x2 = min(self.bbox[0] + self.bbox[2], other.bbox[0] + other.bbox[2])
        y2 = min(self.bbox[1] + self.bbox[3], other.bbox[1] + other.bbox[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        union_area = self.bbox[2] * self.bbox[3] + other.bbox[2] * other.bbox[3] - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area


class FaceSet:
    latent_ids = np.load("sample_faces/latent_ids.npz")

    def __init__(self):
        self.faces = []
        self.nonused_counter = []

    def append(self, face: Face):
        self.faces.append(face)
        self.nonused_counter.append(0)

    def set_attributes(self, i: int, age: int, gender: str):
        self.faces[i].set_attributes(age, gender)
        self.faces[i].latent_id = self.latent_ids[f"{age[0]}_{gender[0]}_jp"]

    def __len__(self) -> int:
        # s = sum(c == 0 for c in self.nonused_counter)
        # return s
        return len(self.faces)

    def __getitem__(self, idx: int) -> Face:
        return self.faces[idx]

    def __iter__(self):
        # s = sum(c == 0 for c in self.nonused_counter)
        # return iter(self.faces[:s])
        return iter(self.faces)

    def update(self, other, reset_nonused_threshold: int):
        matched_self_indices = []

        for i, other_face in enumerate(other):
            max_iou = 0
            max_j = -1
            for j, self_face in enumerate(self.faces):
                iou = other_face.calc_iou(self_face)
                if iou > max_iou:
                    max_iou = iou
                    max_j = j

            if max_iou > 0.3:
                self.faces[max_j].update(other_face.keypoint)
                self.nonused_counter[max_j] = 0
                matched_self_indices.append(max_j)
            else:
                self.append(other_face)
                matched_self_indices.append(len(self.faces)-1)

        for j in range(len(self.faces)):
            if j not in matched_self_indices:
                self.nonused_counter[j] += 1

        argsort = np.argsort(self.nonused_counter)[::-1]
        self.faces = [self.faces[j] for j in argsort]
        self.nonused_counter = [self.nonused_counter[j] for j in argsort]

        self.faces = [face for j, face in enumerate(self.faces) if self.nonused_counter[j] < reset_nonused_threshold]
        self.nonused_counter = [count for count in self.nonused_counter if count < reset_nonused_threshold]


class FaceCropper:
    def __init__(self, yolo_path: str, dlib_path: str):
        self.size = 256
        self.crop_size = 224
        self.detector = YOLO(yolo_path)
        self.predictor = dlib.shape_predictor(dlib_path)

    def detect_keypoints(self, image: np.ndarray) -> FaceSet:
        height, width = image.shape[:2]
    
        results = self.detector.predict(image, verbose=False, conf=0.8)
        pts = results[0].boxes.data.to("cpu").detach().numpy()
        if len(pts) == 0:
            return FaceSet()

        x0, y0, x1, y1 = pts[:,0], pts[:,1], pts[:,2], pts[:,3]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        s = 0.25 * (x1 - x0 + y1 - y0)
        x0, y0, x1, y1 = cx - s, cy - s, cx + s, cy + s

        faces_list = FaceSet()

        for i in range(len(pts)):
            rect = dlib.rectangle(int(x0[i]), int(y0[i]), int(x1[i]), int(y1[i]))
            shape = self.predictor(image, rect)
            face = face_utils.shape_to_np(shape)

            left_eye = face[36:42].mean(0)
            right_eye = face[42:48].mean(0)
            nose = face[27:36].mean(0)
            mouth = face[48:68].mean(0)

            faces_list.append(Face(keypoint=[left_eye, right_eye, nose, mouth]))

        return faces_list

    def crop_and_resize(self, image: np.ndarray, face: Face) -> np.ndarray:
        cx, cy = face.get_center()
        theta = face.theta
        s = face.get_size()

        M = cv2.getRotationMatrix2D((cx, cy), np.degrees(theta), self.size / s * 1.14)
        M[0, 2] += self.crop_size // 2 - cx
        M[1, 2] += self.crop_size // 2 - cy

        cropped = cv2.warpAffine(image, M, (self.crop_size, self.crop_size), flags=cv2.INTER_LANCZOS4)
        return cropped

    def invert_image(self, image: np.ndarray, cropped: np.ndarray, face: Face) -> np.ndarray:
        cx, cy = face.get_center()
        theta = face.theta
        s = face.get_size()

        M = cv2.getRotationMatrix2D((cx, cy), np.degrees(theta), self.size / s * 1.14)
        M[0, 2] += self.crop_size // 2 - cx
        M[1, 2] += self.crop_size // 2 - cy

        M_inv = cv2.invertAffineTransform(M)
        inverted = cv2.warpAffine(cropped, M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LANCZOS4)

        mask = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)
        mask[8:-8, 8:-8] = 255
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask = cv2.warpAffine(mask, M_inv, (image.shape[1], image.shape[0]))
        mask = mask.astype(np.float32)[:,:,None] / 255.0

        result = image.astype(np.float32) * (1 - mask) + inverted.astype(np.float32) * mask
        result = result.astype(np.uint8)
        return result


class FaceSwapper:
    def __init__(self, model_path: str, classifier_checkpoint: str):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.generator = MobileGenerator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=6, deep=False)
        self.generator.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=False))
        self.generator.to(self.device).eval()

        self.classifier = MobileNetV3MultiTask(model_name="mobilenetv3_small_100", num_age_classes=10, num_gender_classes=2)
        self.classifier.to(self.device).eval()
        self.classifier.load_state_dict(torch.load(classifier_checkpoint, weights_only=False)["model_state_dict"])

        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)

    def np2tensor(self, imgs: np.ndarray) -> torch.Tensor:
        if not isinstance(imgs, list):
            imgs = [imgs]

        imgs = [
            torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
            for img in imgs
        ]
        imgs = torch.cat(imgs, dim=0)
        return (imgs - self.mean) / self.std

    def tensor2np(self, imgs: torch.Tensor) -> np.ndarray:
        imgs = imgs * self.std + self.mean
        imgs = imgs.permute(0, 2, 3, 1).detach().numpy()
        imgs = np.clip(imgs, 0, 1)
        return [(img * 255).astype(np.uint8) for img in imgs]

    @torch.no_grad()
    @torch.autocast("cuda", torch.bfloat16)
    def classify(self, img: np.ndarray) -> list[tuple[int, str]]:
        img_tensor = self.np2tensor(img).to(self.device)
        ages, genders = self.classifier(img_tensor)
        ages = torch.softmax(ages, dim=1)
        genders = torch.softmax(genders, dim=1)
        attributes = []
        for i in range(len(img_tensor)):
            age = ages[i].argmax().item() * 10
            age_logit = ages[i].max().item()
            gender = "F" if genders[i].argmax().item() == 0 else "M"
            gender_logit = genders[i].max().item()
            attributes.append(([age, age_logit], [gender, gender_logit]))
        return attributes

    @torch.no_grad()
    @torch.autocast("cuda", torch.bfloat16)
    def swap(self, img_att: np.ndarray, latent_ids: np.ndarray) -> np.ndarray:
        img_att = self.np2tensor(img_att).to(self.device)
        latent_ids = torch.cat([torch.from_numpy(latent_id) for latent_id in latent_ids]).to(self.device)

        output = self.generator(img_att, latent_ids)
        return self.tensor2np(output.to("cpu"))


def swap_image(target_path: str, output_path: str, face_cropper: FaceCropper, face_swapper: FaceSwapper):
    target_img = cv2.imread(target_path)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    faces = face_cropper.detect_keypoints(target_img)
    if len(faces) == 0:
        print(f"No face detected in {target_path}")
        return

    result = target_img.copy()
    for j, face in enumerate(faces):
        cropped_target = face_cropper.crop_and_resize(target_img, face)

        attributes = face_swapper.classify(cropped_target)
        faces.set_attributes(j, *attributes[0])
        latent_id = face.latent_id

        swapped = face_swapper.swap(cropped_target, latent_id)[0]
        result = face_cropper.invert_image(result, swapped, face)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result)


def swap_video(target_path: str, output_path: str, face_cropper: FaceCropper, face_swapper: FaceSwapper, verbose: bool = False):
    target_video = cv2.VideoCapture(target_path)
    num_frames = int(target_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = target_video.get(cv2.CAP_PROP_FPS)
    width = int(target_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(target_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if verbose:
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height*2))
    else:
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    faces = FaceSet()
    for i in tqdm(range(num_frames)):
        ret, frame = target_video.read()
        if not ret:
            break

        target_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_faces = face_cropper.detect_keypoints(target_img)
        faces.update(current_faces, reset_nonused_threshold=fps*1.0)

        result = target_img.copy()
        if len(faces) > 0:
            cropped_targets = []
            latent_ids = []
            for j, face in enumerate(faces):
                cropped_face = face_cropper.crop_and_resize(target_img, face)

                if not hasattr(face, "age"):
                    attributes = face_swapper.classify(cropped_face)
                    assert len(attributes) == 1
                    faces.set_attributes(j, *attributes[0])

                cropped_targets.append(cropped_face)
                latent_ids.append(face.latent_id)

            swapped = face_swapper.swap(cropped_targets, latent_ids)

            for k, face in enumerate(faces):
                result = face_cropper.invert_image(result, swapped[k], face)

                if verbose:
                    result = cv2.rectangle(result, (face.bbox[0], face.bbox[1]), (face.bbox[0]+face.bbox[2], face.bbox[1]+face.bbox[3]), (0,255,0), 2)
                    result = cv2.putText(result, f"{face.age[0]} {int(face.age[1]*100)}%", (face.bbox[0], face.bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    result = cv2.putText(result, f"{face.gender[0]} {int(face.gender[1]*100)}%", (face.bbox[0], face.bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if verbose:
            result = np.vstack([target_img, result])

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        writer.write(result)

    target_video.release()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face swapping using SimSwap")
    parser.add_argument("--target", type=str, required=True, help="Path to the target image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the output image")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--gen_checkpoint", type=str, required=True, help="Path to the generator model")
    parser.add_argument("--classifier_checkpoint", type=str, default="/content/best_mobilenetv3_multitask.pth", help="Path to the MobileNetV3MultiTask checkpoint")
    parser.add_argument("--yolo_path", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--dlib_path", type=str, required=True, help="Path to the dlib model")
    args = parser.parse_args()

    face_cropper = FaceCropper(args.yolo_path, args.dlib_path)
    face_swapper = FaceSwapper(args.gen_checkpoint, args.classifier_checkpoint)

    if args.target.endswith((".mp4", ".avi", ".mov")):
        swap_video(args.target, args.output, face_cropper, face_swapper, args.verbose)
    else:
        swap_image(args.target, args.output, face_cropper, face_swapper)
