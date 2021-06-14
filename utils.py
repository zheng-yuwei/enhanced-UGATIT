# -*- coding: utf-8 -*-
import os
import random

import cv2
import torch
import numpy as np
from scipy import misc
from tqdm import tqdm


def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img


def preprocessing(x):
    x = x / 127.5 - 1  # -1 ~ 1
    return x


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images + 1.) / 2


def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h * j:h * (j + 1), w * i:w * (i + 1), :] = image

    return img


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def cam(x, size=256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0


def attention_mask(x, size=256):
    attention_img = cv2.resize(np.uint8(255 * x), (size, size))
    attention_img = cv2.applyColorMap(attention_img, cv2.COLORMAP_JET)
    return attention_img / 255.0


def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std


def denorm(x):
    return x * 0.5 + 0.5


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


"""
评估量：记录，打印
"""


class AverageMeter:
    """ 计算并存储 评估量的均值和当前值 """

    def __init__(self, name, fmt=':f'):
        self.name = name  # 评估量名称
        self.fmt = fmt  # 评估量打印格式
        self.val = 0  # 评估量当前值
        self.avg = 0  # 评估量均值
        self.sum = 0  # 历史评估量的和
        self.count = 0  # 历史评估量的数量

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f'{{name}} {{val{self.fmt}}} ({{avg{self.fmt}}})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """ 评估量的进度条打印 """

    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = f'{{:{str(num_digits)}d}}'
        return f'[{fmt}/{fmt.format(num_batches)}]'


"""
制作模糊图像
"""


def generate_blur_images(root, save):
    """ 根据清晰图像制作模糊的图像
    :param root: 清晰图像所在的根目录
    :param save: 模糊图像存放的根目录
    """
    print(f'generating blur images ...')
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    for f in tqdm(file_list):
        try:
            rgb_img = cv2.imread(os.path.join(root, f))
            gray_img = cv2.imread(os.path.join(root, f), 0)
            pad_img = np.pad(rgb_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
            edges = cv2.Canny(gray_img, 100, 200)
            dilation = cv2.dilate(edges, kernel)

            gauss_img = np.copy(rgb_img)
            idx = np.where(dilation != 0)
            for i in range(np.sum(dilation != 0)):
                gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                    np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0],
                                gauss))
                gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                    np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1],
                                gauss))
                gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                    np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2],
                                gauss))

            cv2.imwrite(os.path.join(save, f), gauss_img)
        except Exception as e:
            print(f'{f} failed!\n{e}')

    print(f'finish: blur images over! ')
