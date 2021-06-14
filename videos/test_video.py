# -*- coding: utf-8 -*-
"""
视频/摄像头测试：人脸关键点检测、矫正、变漫画、贴回原图、保存为视频
"""
import os
from functools import partial

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .face_align.align_utils import detect_face, align_face


class VideoTester:
    """ GAN测试 """

    IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    def __init__(self, args, generator):
        self.args = args
        self.generator = generator
        self.generator.to(args.device)
        self.generator.eval()
        self.preprocess = transforms.Compose([
            partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB),  # 将cv读取的图像转为RGB
            Image.fromarray,
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            partial(torch.unsqueeze, dim=0),
        ])
        width = 1280
        height = 720 * 2
        self.frame_size = (width, height)
        self.text = partial(cv2.putText, org=(width-100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.2, color=(255, 255, 255), thickness=2)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = 30

    def generate(self, img):
        """ 脸部(0, 255)BGR原图生成(0, 255)BGR动漫图 """
        img = self.preprocess(img)
        img = img.to(self.args.device)
        with torch.no_grad():
            img, _, _, _ = self.generator(img)
        img = (img.cpu()[0].numpy().transpose(1, 2, 0) + 1) * 255 / 2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def image2image(self, image):
        """ 整张图像到图像的转换 """
        # 脸部检测及矫正
        facepoints = detect_face(image, detect_scale=0.25, flag_384=self.args.img_size>256)
        if facepoints is None:
            return image
        align_result = align_face(image, facepoints, flag_384=self.args.img_size>256)
        if align_result is None:
            return image
        img_align, face2mean_matrix, mean2face_matrix, _ = align_result
        # 生成
        img_gen = self.generate(img_align)
        # 变换为原来位置
        img_gen = np.clip(img_gen, 0, 255).astype(np.uint8)
        output = np.ones([self.args.img_size, self.args.img_size, 4], dtype=np.uint8) * 255
        output[:, :, :3] = img_gen
        output = cv2.warpAffine(output, mean2face_matrix,
                                (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha_img_gen = (output[:, :, 3:4] / 255.0)
        image_compose = image.astype(np.float32)
        image_compose[:, :, 0:3] = (image_compose[:, :, 0:3] * (1.0 - alpha_img_gen) + output[:, :, 0:3])
        image_compose = np.clip(image_compose, 0, 255).astype(np.uint8)
        return image_compose

    def record(self, cap: cv2.VideoCapture, writer: cv2.VideoWriter,
               frame_num: int):
        """ 从cap视频源读取图像，经过图像转换后，写入到writer中 """
        ret, frame = cap.read()
        if frame is None:
            return -1
        frame = frame[:, ::-1, :]  # 左右flip
        org_frame = frame.copy()
        new_frame = self.image2image(frame)
        new_frame = np.concatenate([org_frame, new_frame], axis=0)
        self.text(new_frame, str(frame_num))
        writer.write(new_frame)
        frame_num += 1
        return frame_num

    def video(self, path):
        """ 视频测试 """
        video_name = os.path.basename(self.args.generator_model)
        video_name = os.path.splitext(os.path.basename(path))[0] + '_' + video_name
        new_record = cv2.VideoWriter(f'{os.path.splitext(video_name)[0]}_cartoon.avi',
                                     self.fourcc, self.fps, self.frame_size)
        cap = cv2.VideoCapture(path)  # 打开视频
        frame_num = 0
        while cap.isOpened():
            frame_num = self.record(cap, new_record, frame_num)
            if frame_num < 0:
                break
        cap.release(), new_record.release()

    def camera(self):
        """ 摄像头测试 """
        video_name = os.path.basename(self.args.generator_model)
        cap = cv2.VideoCapture(0)  # 打开摄像头
        ret = True
        while ret:
            ret, frame = cap.read()
            frame = self.image2image(frame)
            cv2.imshow(video_name, frame)
            cv2.waitKey(1)
            if 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def image_dir(self, img_dir):
        """ 图像文件夹测试 """
        img_paths = [f for f in os.listdir(img_dir) if os.path.splitext(f)[-1].lower() in self.IMAGE_EXT]
        save_dir = os.path.join(img_dir, '..', 'gan_result')
        os.makedirs(save_dir, exist_ok=False)
        for img_path in img_paths:
            image = cv2.imread(os.path.join(img_dir, img_path))
            image_gan = self.image2image(image)
            cv2.imwrite(os.path.join(save_dir, img_path), image_gan)
