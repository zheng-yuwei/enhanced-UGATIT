# -*- encoding: utf-8 -*-
import os
import functools

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from faceseg.networks_faceseg import BiSeNet


class FaceSegmentation:
    part_map = {0: 'background', 1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 6: 'eye_g', 7: 'l_ear',
                8: 'r_ear', 9: 'ear_r', 10: 'nose', 11: 'mouth', 12: 'u_lip', 13: 'l_lip', 14: 'neck', 15: 'neck_l',
                16: 'cloth', 17: 'hair', 18: 'hat'}
    # 不同部分的颜色
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0], [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    def __init__(self, device):
        self.device = device
        self.n_classes = len(FaceSegmentation.part_map.keys())
        # 膨胀、腐蚀、闭运算、开运算
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 模型加载
        net = BiSeNet(n_classes=self.n_classes).to(self.device)
        net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '79999_iter.pth'),
                                       map_location=torch.device('cpu')))
        net.eval()
        self.net = net
        # 预处理
        self.preprocessor = transforms.Compose([
            lambda x: x * 0.5 + 0.5,
            functools.partial(F.interpolate, size=(512, 512), mode='bilinear', align_corners=True),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def face_segmentation(self, input_x):
        """ 人脸分割
        :param input_x: NCHW 标准化的tensor, (image / 255 - 0.5) * 2
        :return mask N1HW tensor，每一个像素点位置取值 0~18 int数，表示属于哪一类
        """
        with torch.no_grad():
            img = self.preprocessor(input_x)
            img = img.to(self.device)
            out = self.net(img)[0]
            out = F.interpolate(out, input_x.shape[2:], mode='bilinear', align_corners=True)
            mask = out.detach().argmax(1, keepdims=True)
        return mask

    def gen_mask(self, mask_tensor, dilate_parts=(4, 5, 6, 11), erode_parts=(0, 14, 15, 16, 17, 18,)):
        """ 根据指定的parts生成mask numpy数组 """
        dilate_mask = np.zeros(mask_tensor.shape, dtype=np.float32)
        erode_mask = np.zeros(mask_tensor.shape, dtype=np.float32)
        mask = torch.ones_like(mask_tensor, dtype=torch.float32)
        for i in dilate_parts:
            dilate_mask[mask_tensor == i] = 1
        for i in erode_parts:
            erode_mask[mask_tensor == i] = 1
        # 闭运算 + 膨胀
        for i in range(len(dilate_mask)):
            dilate_mask[i, 0] = cv2.morphologyEx(dilate_mask[i, 0], cv2.MORPH_CLOSE, self.kernel)
            dilate_mask[i, 0] = cv2.dilate(dilate_mask[i, 0], self.kernel, iterations=1)
        # 闭运算 + 腐蚀
        for i in range(len(erode_mask)):
            erode_mask[i, 0] = cv2.morphologyEx(erode_mask[i, 0], cv2.MORPH_CLOSE, self.kernel)
            erode_mask[i, 0] = cv2.erode(erode_mask[i, 0], self.kernel, iterations=1)
        mask = mask * (1 - (1 - dilate_mask) * (1 - erode_mask))
        return mask

    def vis(self, image, mask, is_show=False):
        """ 可视化人脸分割结果 """
        vis_im = (image.numpy().transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)
        vis_mask = mask.numpy().astype(np.uint8)
        vis_mask_color = np.zeros((vis_mask.shape[0], vis_mask.shape[1], 3)) + 255

        if mask.dtype == torch.int64:
            for pi in range(1, self.n_classes):
                index = np.where(vis_mask == pi)
                vis_mask_color[index[0], index[1], :] = self.part_colors[pi]
        else:
            index = np.where(vis_mask > 0.8)
            vis_mask_color[index[0], index[1], :] = self.part_colors[1]

        vis_mask_color = vis_mask_color.astype(np.uint8)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_mask_color, 0.6, 0)
        if is_show:
            cv2.imshow('seg', vis_im)
            cv2.waitKey(0)
        return vis_im


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    img_dir = '../dataset/cartoon/testA'
    face_seg = FaceSegmentation('cpu')
    train_transform = transforms.Compose([
        cv2.imread,
        functools.partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        functools.partial(torch.unsqueeze, dim=0)
    ])
    for f_name in sorted(os.listdir(img_dir)):
        test_img = train_transform(os.path.join(img_dir, f_name))
        test_mask = face_seg.face_segmentation(test_img)
        test_mask = face_seg.gen_mask(test_mask)  # 注释这句可以看到所有类别
        face_seg.vis(test_img[0], test_mask[0, 0], is_show=True)
