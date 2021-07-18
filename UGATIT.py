# -*- coding: utf-8 -*-
import os
import time
import copy
import itertools
from glob import glob

import cv2
import PIL
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from faceseg.FaceSegmentation import FaceSegmentation
from networks import ResnetGenerator, Discriminator, RhoClipper, LIN, AdaLIN
from utils import (AverageMeter, ProgressMeter, generate_blur_images,
                   RGB2BGR, tensor2numpy, attention_mask, cam, denorm)
from dataset import MatchHistogramsDataset, DatasetFolder, get_loader
from metrics import FIDScore


class UGATIT(object):
    def __init__(self, args):
        self.args = args

        if self.args.light > 0:
            self.model_name = 'UGATIT_light' + str(self.args.light)
        else:
            self.model_name = 'UGATIT'

        print(f'\n##### Information #####\n'
              f'# light : {self.args.light}\n'
              f'# dataset : {self.args.dataset}\n'
              f'# batch_size : {self.args.batch_size}\n'
              f'# num_workers : {self.args.num_workers}\n'
              f'# aug_prob : {self.args.aug_prob}\n'
              f'# iteration : {self.args.iteration}\n\n'
              f'##### Generator #####\n'
              f'# residual blocks : {self.args.n_res}\n'
              f'# img_size : {self.args.img_size}\n'
              f'# aug_prob : {self.args.aug_prob}\n'
              f'# match_histograms : {self.args.match_histograms}\n'
              f'# match_mode : {self.args.match_mode}\n'
              f'# match_prob : {self.args.match_prob}\n'
              f'# match_ratio : {self.args.match_ratio}\n'
              f'# use se or not : {self.args.use_se}\n'
              f'# use blur or not : {self.args.has_blur}\n'
              f'# use deconv or not : {self.args.use_deconv}\n'
              f'# use attention gan : {self.args.attention_gan}\n'
              f'# use attention input : {self.args.attention_input}\n\n'
              f'##### Discriminator #####\n'
              f'# discriminator layer : {self.args.n_dis}\n'
              f'# use sn : {self.args.sn}\n\n'
              f'##### Weight #####\n'
              f'# adv_weight : {self.args.adv_weight}\n'
              f'# forward_adv_weight : {self.args.forward_adv_weight}\n'
              f'# backward_adv_weight : {self.args.backward_adv_weight}\n'
              f'# cycle_weight : {self.args.cycle_weight}\n'
              f'# identity_weight : {self.args.identity_weight}\n'
              f'# cam_weight : {self.args.cam_weight}\n'
              f'# seg_weight : {self.args.seg_weight}\n'
              f'# seg_rand_mask : {self.args.seg_rand_mask}\n'
              f'# resume : {self.args.resume}\n\n'
              )
        self.genA2B, self.genB2A = None, None
        self.genA2B_ema, self.genB2A_ema = None, None
        self.disGA, self.disGB, self.disLA, self.disLB = None, None, None, None
        self.FaceSeg, self.fix_maskA, self.fix_maskB = None, None, None
        self.trainA_data_root = os.path.join('dataset', self.args.dataset, 'trainA')
        self.trainB_data_root = os.path.join('dataset', self.args.dataset, 'trainB')
        self.testA_data_root = os.path.join('dataset', self.args.dataset, 'testA')
        self.testB_data_root = os.path.join('dataset', self.args.dataset, 'testB')
        self.blurA_data_root = os.path.join('dataset', self.args.dataset, 'blurA')
        self.blurB_data_root = os.path.join('dataset', self.args.dataset, 'blurB')
        self.train_transform, self.test_transform = None, None
        self.trainAB_loader, self.blurAB_loader = None, None
        self.trainAB_iter, self.blurAB_iter = None, None
        self.testA_loader, self.testB_loader = None, None
        self.testA_iter, self.testB_iter = None, None
        self.L1_loss, self.MSE_loss, self.BCE_loss = None, None, None
        self.G_optim, self.D_optim = None, None
        self.Rho_LIN_clipper, self.Rho_AdaLIN_clipper = None, None
        self.G_adv_loss, self.G_cyc_loss, self.G_idt_loss, self.G_cam_loss = None, None, None, None
        self.Generator_loss, self.G_seg_loss = None, None
        self.discriminator_loss = None
        self.fid_score, self.mean_std_A, self.mean_std_B = None, None, None
        self.fid_loaderA, self.fid_loaderB = None, None

    ##################################################################################
    # Model
    ##################################################################################

    def build_data_loader(self):
        """ 构造data loader """
        self.train_transform = transforms.Compose([
            PIL.Image.fromarray,
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.RandomResizedCrop(size=self.args.img_size, scale=(0.748, 1.0), ratio=(1.0, 1.0),
                                              interpolation=transforms.InterpolationMode.BICUBIC)],
                p=self.args.aug_prob),
            transforms.Resize(size=self.args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.test_transform = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((self.args.img_size, self.args.img_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        trainAB = MatchHistogramsDataset((self.trainA_data_root, self.trainB_data_root),
                                         self.train_transform, is_match_histograms=self.args.match_histograms,
                                         match_mode=self.args.match_mode, b2a_prob=self.args.match_prob,
                                         match_ratio=self.args.match_ratio)
        self.trainAB_loader = get_loader(trainAB, self.args.device, batch_size=self.args.batch_size,
                                         shuffle=True, num_workers=self.args.num_workers)
        testA = DatasetFolder(self.testA_data_root, self.test_transform)
        testB = DatasetFolder(self.testB_data_root, self.test_transform)
        self.testA_loader = get_loader(testA, self.args.device, batch_size=1, shuffle=False,
                                       num_workers=self.args.num_workers)
        self.testB_loader = get_loader(testB, self.args.device, batch_size=1, shuffle=False,
                                       num_workers=self.args.num_workers)

        # 使用模糊图像增强判别器D对模糊的判别，从而增强生成器G生成清晰图像
        if self.args.has_blur:
            if not os.path.exists(self.blurA_data_root):
                generate_blur_images(self.trainA_data_root, self.blurA_data_root)
            if not os.path.exists(self.blurB_data_root):
                generate_blur_images(self.trainB_data_root, self.blurB_data_root)

            blurAB = MatchHistogramsDataset((self.blurA_data_root, self.blurB_data_root), self.train_transform,
                                            is_match_histograms=self.args.match_histograms,
                                            match_mode=self.args.match_mode, b2a_prob=self.args.match_prob,
                                            match_ratio=self.args.match_ratio)
            self.blurAB_loader = get_loader(blurAB, self.args.device, batch_size=self.args.batch_size,
                                            shuffle=True, num_workers=self.args.num_workers)

    def build_model(self):
        """ 构造data loader，Generator，Discriminator 模型，损失，优化器 """
        self.build_data_loader()
        # Define Generator, Discriminator
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.args.ch, n_blocks=self.args.n_res,
                                      img_size=self.args.img_size, args=self.args).to(self.args.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.args.ch, n_blocks=self.args.n_res,
                                      img_size=self.args.img_size, args=self.args).to(self.args.device)
        self.genA2B_ema = copy.deepcopy(self.genA2B).eval().requires_grad_(False)
        self.genB2A_ema = copy.deepcopy(self.genB2A).eval().requires_grad_(False)
        self.disGA = Discriminator(input_nc=3, ndf=self.args.ch, n_layers=7, with_sn=self.args.sn).to(self.args.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.args.ch, n_layers=7, with_sn=self.args.sn).to(self.args.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.args.ch, n_layers=5, with_sn=self.args.sn).to(self.args.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.args.ch, n_layers=5, with_sn=self.args.sn).to(self.args.device)

        # 使用分割区域做L2监督损失，或，分割出来的区域随机填充颜色的填充概率值
        if self.args.seg_weight > 0 or self.args.seg_rand_mask > 0:
            self.FaceSeg = FaceSegmentation(self.args.device)

        # Define Loss
        self.L1_loss = nn.L1Loss().to(self.args.device)
        self.MSE_loss = nn.MSELoss().to(self.args.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.args.device)

        # 优化器
        gen_params = itertools.chain(self.genA2B.parameters(), self.genB2A.parameters())
        self.G_optim = torch.optim.Adam(gen_params, lr=self.args.lr, betas=(0.5, 0.999),
                                        weight_decay=self.args.weight_decay)
        disc_params = itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
                                      self.disLA.parameters(), self.disLB.parameters())
        self.D_optim = torch.optim.Adam(disc_params, lr=self.args.lr, betas=(0.5, 0.999),
                                        weight_decay=self.args.weight_decay)

        # Define Rho clipper to constraint the value of rho in AdaLIN and LIN
        # self.Rho_clipper = RhoClipper(0, 1)
        self.Rho_LIN_clipper = RhoClipper(0, 1, LIN)
        self.Rho_AdaLIN_clipper = RhoClipper(0.0, 0.9, AdaLIN)

    ##################################################################################
    # 工具函数
    ##################################################################################

    def gen_train(self, on=True):
        """ 开启生成网络训练模式 """
        if on:
            self.genA2B.train(), self.genB2A.train()
        else:
            self.genA2B.eval(), self.genB2A.eval()

    def dis_train(self, on=True):
        """ 开启判别网络训练模式 """
        if on:
            self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
        else:
            self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()

    def get_batch(self, mode='train'):
        """ 获取训练数据 """
        if mode == 'train':
            try:
                real_A, real_B = self.trainAB_iter.next()
            except:
                self.trainAB_iter = iter(self.trainAB_loader)
                real_A, real_B = self.trainAB_iter.next()
        else:
            try:
                real_A = self.testA_iter.next()
            except:
                self.testA_iter = iter(self.testA_loader)
                real_A = self.testA_iter.next()

            try:
                real_B = self.testB_iter.next()
            except:
                self.testB_iter = iter(self.testB_loader)
                real_B = self.testB_iter.next()

        real_A, real_B = real_A.to(self.args.device, non_blocking=True), real_B.to(self.args.device, non_blocking=True)

        blur = None
        if self.args.has_blur and mode == 'train':
            try:
                blur_A, blur_B = self.blurAB_iter.next()
            except:
                self.blurAB_iter = iter(self.blurAB_loader)
                blur_A, blur_B = self.blurAB_iter.next()

            blur = (blur_A.to(self.args.device, non_blocking=True), blur_B.to(self.args.device, non_blocking=True))

        return real_A, real_B, blur

    ##################################################################################
    # 训练
    ##################################################################################

    def forward(self, real_A, real_B):
        """ 前向推理：A->B->A, B->A->B, A->A, B->B """
        # cycle
        fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap, fake_A2B_attention = self.genA2B(real_A)
        fake_A2B2A, _, fake_A2B2A_heatmap, fake_A2B2A_attention = self.genB2A(fake_A2B)
        fake_B2A, fake_B2A_cam_logit, fake_B2A_heatmap, fake_B2A_attention = self.genB2A(real_B)
        fake_B2A2B, _, fake_B2A2B_heatmap, fake_B2A2B_attention = self.genA2B(fake_B2A)
        # 单位映射
        fake_A2A, fake_A2A_cam_logit, fake_A2A_heatmap, fake_A2A_attention = self.genB2A(real_A)
        fake_B2B, fake_B2B_cam_logit, fake_B2B_heatmap, fake_B2B_attention = self.genA2B(real_B)

        # 根据人脸分割，获取分割区域 self.fix_maskA (==1)
        if self.args.seg_weight > 0 or self.args.seg_rand_mask > 0:
            tensorA = self.FaceSeg.face_segmentation(real_A)
            self.fix_maskA = self.FaceSeg.gen_mask(tensorA)  # 背景、眼睛、眼镜、嘴巴等的mask
            tensorB = self.FaceSeg.face_segmentation(real_B)
            self.fix_maskB = self.FaceSeg.gen_mask(tensorB)

        return (fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap, fake_A2B_attention,
                fake_A2B2A, fake_A2B2A_heatmap, fake_A2B2A_attention,
                fake_B2A, fake_B2A_cam_logit, fake_B2A_heatmap, fake_B2A_attention,
                fake_B2A2B, fake_B2A2B_heatmap, fake_B2A2B_attention,
                fake_A2A, fake_A2A_cam_logit, fake_A2A_heatmap, fake_A2A_attention,
                fake_B2B, fake_B2B_cam_logit, fake_B2B_heatmap, fake_B2B_attention)

    def backward_D(self, real_A, real_B, fake_A2B, fake_B2A, blur=None):
        """ D网络前向+反向计算 """
        if self.args.seg_rand_mask > 0 and self.args.seg_rand_mask > np.random.rand():
            rand_color = np.expand_dims(np.expand_dims((np.random.rand(3) * 2 - 1), -1), -1).astype(np.float32)
            background_color = torch.ones_like(real_A) * torch.from_numpy(rand_color).to(self.args.device)
            real_A = real_A * (1 - self.fix_maskA) + background_color * self.fix_maskA
            fake_B2A = fake_B2A * (1 - self.fix_maskA) + background_color * self.fix_maskA
            real_B = real_B * (1 - self.fix_maskB) + background_color * self.fix_maskB
            fake_A2B = fake_A2B * (1 - self.fix_maskB) + background_color * self.fix_maskB
            if blur is not None:
                blur_A, blur_B = blur
                blur_A = blur_A * (1 - self.fix_maskA) + background_color * self.fix_maskA
                blur_B = blur_B * (1 - self.fix_maskB) + background_color * self.fix_maskB
                blur = blur_A, blur_B

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A.detach())
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A.detach())

        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B.detach())
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B.detach())

        # 设置目标常量，GA和LA的D网络输出shape不一致，但cam的分类输出是一致的
        flag_GA_1 = torch.ones_like(real_GA_logit, requires_grad=False).to(self.args.device)
        flag_GA_0 = torch.zeros_like(fake_GA_logit, requires_grad=False).to(self.args.device)
        flag_LA_1 = torch.ones_like(real_LA_logit, requires_grad=False).to(self.args.device)
        flag_LA_0 = torch.zeros_like(fake_LA_logit, requires_grad=False).to(self.args.device)
        flag_cam_1 = torch.ones_like(real_GA_cam_logit, requires_grad=False).to(self.args.device)
        flag_cam_0 = torch.zeros_like(fake_GA_cam_logit, requires_grad=False).to(self.args.device)

        # 背景区域不需要判别损失，置为目标值，相当于把背景区域的判别损失置0
        if self.args.seg_weight > 0:
            fix_maskA_G = F.interpolate(self.fix_maskA, fake_GA_logit.shape[2:], mode='area')
            fix_maskA_L = F.interpolate(self.fix_maskA, fake_LA_logit.shape[2:], mode='area')
            real_GA_logit = real_GA_logit * (1 - fix_maskA_G) + flag_GA_1 * fix_maskA_G
            real_LA_logit = real_LA_logit * (1 - fix_maskA_L) + flag_LA_1 * fix_maskA_L
            fake_GA_logit = fake_GA_logit * (1 - fix_maskA_G) + flag_GA_0 * fix_maskA_G
            fake_LA_logit = fake_LA_logit * (1 - fix_maskA_L) + flag_LA_0 * fix_maskA_L
            fix_maskB_G = F.interpolate(self.fix_maskB, fake_GB_logit.shape[2:], mode='area')
            fix_maskB_L = F.interpolate(self.fix_maskB, fake_LB_logit.shape[2:], mode='area')
            real_GB_logit = real_GB_logit * (1 - fix_maskB_G) + flag_GA_1 * fix_maskB_G
            real_LB_logit = real_LB_logit * (1 - fix_maskB_L) + flag_LA_1 * fix_maskB_L
            fake_GB_logit = fake_GB_logit * (1 - fix_maskB_G) + flag_GA_0 * fix_maskB_G
            fake_LB_logit = fake_LB_logit * (1 - fix_maskB_L) + flag_LA_0 * fix_maskB_L

        # D网络损失函数：cam和D网络损失
        D_loss_GA, D_cam_loss_GA = 0, 0
        D_loss_LA, D_cam_loss_LA = 0, 0
        D_loss_GB, D_cam_loss_GB = 0, 0
        D_loss_LB, D_cam_loss_LB = 0, 0
        if blur is not None:
            blur_A, blur_B = blur
            blur_GA_logit, blur_GA_cam_logit, _ = self.disGA(blur_A)
            blur_LA_logit, blur_LA_cam_logit, _ = self.disLA(blur_A)
            blur_GB_logit, blur_GB_cam_logit, _ = self.disGB(blur_B)
            blur_LB_logit, blur_LB_cam_logit, _ = self.disLB(blur_B)
            # 背景区域不需要判别损失，置为目标值，相当于把背景区域的判别损失置0
            if self.args.seg_weight > 0:
                blur_GA_logit = blur_GA_logit * (1 - fix_maskA_G) + flag_GA_0 * fix_maskA_G
                blur_LA_logit = blur_LA_logit * (1 - fix_maskA_L) + flag_LA_0 * fix_maskA_L
                blur_GB_logit = blur_GB_logit * (1 - fix_maskB_G) + flag_GA_0 * fix_maskB_G
                blur_LB_logit = blur_LB_logit * (1 - fix_maskB_L) + flag_LA_0 * fix_maskB_L

            D_loss_GA = self.MSE_loss(blur_GA_logit, flag_GA_0)
            D_cam_loss_GA = self.MSE_loss(blur_GA_cam_logit, flag_cam_0)
            D_loss_LA = self.MSE_loss(blur_LA_logit, flag_LA_0)
            D_cam_loss_LA = self.MSE_loss(blur_LA_cam_logit, flag_cam_0)
            D_loss_GB = self.MSE_loss(blur_GB_logit, flag_GA_0)
            D_cam_loss_GB = self.MSE_loss(blur_GB_cam_logit, flag_cam_0)
            D_loss_LB = self.MSE_loss(blur_LB_logit, flag_LA_0)
            D_cam_loss_LB = self.MSE_loss(blur_LB_cam_logit, flag_cam_0)
        D_loss_GA = D_loss_GA + self.MSE_loss(real_GA_logit, flag_GA_1) + self.MSE_loss(fake_GA_logit, flag_GA_0)
        D_loss_GA *= self.args.forward_adv_weight
        D_cam_loss_GA = D_cam_loss_GA + self.MSE_loss(real_GA_cam_logit, flag_cam_1) + \
                        self.MSE_loss(fake_GA_cam_logit, flag_cam_0)
        D_loss_LA = D_loss_LA + self.MSE_loss(real_LA_logit, flag_LA_1) + self.MSE_loss(fake_LA_logit, flag_LA_0)
        D_loss_LA = D_loss_LA * self.args.forward_adv_weight
        D_cam_loss_LA = D_cam_loss_LA + self.MSE_loss(real_LA_cam_logit, flag_cam_1) + \
                        self.MSE_loss(fake_LA_cam_logit, flag_cam_0)

        D_loss_GB = D_loss_GB + self.MSE_loss(real_GB_logit, flag_GA_1) + self.MSE_loss(fake_GB_logit, flag_GA_0)
        D_loss_GB *= self.args.backward_adv_weight
        D_cam_loss_GB = D_cam_loss_GB + self.MSE_loss(real_GB_cam_logit, flag_cam_1) + \
                        self.MSE_loss(fake_GB_cam_logit, flag_cam_0)
        D_loss_LB = D_loss_LB + self.MSE_loss(real_LB_logit, flag_LA_1) + self.MSE_loss(fake_LB_logit, flag_LA_0)
        D_loss_LB = D_loss_LB * self.args.backward_adv_weight
        D_cam_loss_LB = D_cam_loss_LB + self.MSE_loss(real_LB_cam_logit, flag_cam_1) + \
                        self.MSE_loss(fake_LB_cam_logit, flag_cam_0)

        D_loss_A = self.args.adv_weight * (D_loss_GA + D_cam_loss_GA + D_loss_LA + D_cam_loss_LA)
        D_loss_B = self.args.adv_weight * (D_loss_GB + D_cam_loss_GB + D_loss_LB + D_cam_loss_LB)

        self.discriminator_loss = D_loss_A + D_loss_B
        self.discriminator_loss.backward()
        return D_loss_A, D_loss_B

    def backward_G(self, real_A, real_B, fake_A2B, fake_B2A, fake_A2B2A, fake_B2A2B, fake_A2A, fake_B2B,
                   fake_A2B_cam_logit, fake_B2A_cam_logit, fake_A2A_cam_logit, fake_B2B_cam_logit):
        # 根据人脸分割，获取背景不变性损失项
        if self.args.seg_weight > 0:
            G_seg_loss_B = self.L1_loss(fake_A2B * self.fix_maskA, real_A * self.fix_maskA)
            G_seg_loss_A = self.L1_loss(fake_B2A * self.fix_maskB, real_B * self.fix_maskB)
            self.G_seg_loss = self.args.seg_weight * (G_seg_loss_A + G_seg_loss_B)
            # 将生成图像的背景detach掉，使背景上的对抗损失梯度不影响G
            fake_A2B = fake_A2B * (1.0 - self.fix_maskA) + fake_A2B.detach() * self.fix_maskA
            fake_B2A = fake_B2A * (1.0 - self.fix_maskB) + fake_B2A.detach() * self.fix_maskB
        # 判别器输出
        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
        # 设置目标常量，GA和LA的D网络输出shape不一致，但cam的分类输出是一致的
        flag_GA_1 = torch.ones_like(fake_GA_logit, requires_grad=False).to(self.args.device)
        flag_LA_1 = torch.ones_like(fake_LA_logit, requires_grad=False).to(self.args.device)
        flag_GA_cam_1 = torch.ones_like(fake_GA_cam_logit, requires_grad=False).to(self.args.device)
        # 对抗损失
        if self.args.seg_weight > 0:
            # 背景区域不需要对抗损失，置为目标值，等价于把背景区域的对抗损失置0
            fix_maskA_G = F.interpolate(self.fix_maskA, fake_GA_logit.shape[2:], mode='area')
            fix_maskA_L = F.interpolate(self.fix_maskA, fake_LA_logit.shape[2:], mode='area')
            fake_GA_logit = fake_GA_logit * (1 - fix_maskA_G) + flag_GA_1 * fix_maskA_G
            fake_LA_logit = fake_LA_logit * (1 - fix_maskA_L) + flag_LA_1 * fix_maskA_L
            fix_maskB_G = F.interpolate(self.fix_maskB, fake_GB_logit.shape[2:], mode='area')
            fix_maskB_L = F.interpolate(self.fix_maskB, fake_LB_logit.shape[2:], mode='area')
            fake_GB_logit = fake_GB_logit * (1 - fix_maskB_G) + flag_GA_1 * fix_maskB_G
            fake_LB_logit = fake_LB_logit * (1 - fix_maskB_L) + flag_LA_1 * fix_maskB_L

        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, flag_GA_1) * self.args.forward_adv_weight
        G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, flag_GA_cam_1)
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, flag_LA_1) * self.args.forward_adv_weight
        G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, flag_GA_cam_1)
        G_ad_loss_A = self.args.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA)

        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, flag_GA_1) * self.args.backward_adv_weight
        G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, flag_GA_cam_1)
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, flag_LA_1) * self.args.backward_adv_weight
        G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, flag_GA_cam_1)
        G_ad_loss_B = self.args.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB)
        # 循环一致性损失
        G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A) * self.args.cycle_weight
        G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B) * self.args.cycle_weight
        # 单位映射损失
        G_identity_loss_A = self.L1_loss(fake_A2A, real_A) * self.args.identity_weight
        G_identity_loss_B = self.L1_loss(fake_B2B, real_B) * self.args.identity_weight
        # cam损失
        flag_cam_1 = torch.ones_like(fake_B2A_cam_logit, requires_grad=False).to(self.args.device)
        flag_cam_0 = torch.zeros_like(fake_A2A_cam_logit, requires_grad=False).to(self.args.device)
        G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, flag_cam_1) + self.BCE_loss(fake_A2A_cam_logit, flag_cam_0)
        G_cam_loss_A *= self.args.cam_weight
        flag_cam_1 = torch.ones_like(fake_A2B_cam_logit, requires_grad=False).to(self.args.device)
        flag_cam_0 = torch.zeros_like(fake_B2B_cam_logit, requires_grad=False).to(self.args.device)
        G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, flag_cam_1) + self.BCE_loss(fake_B2B_cam_logit, flag_cam_0)
        G_cam_loss_B *= self.args.cam_weight

        G_loss_A = G_ad_loss_A + G_recon_loss_A + G_identity_loss_A + G_cam_loss_A
        G_loss_B = G_ad_loss_B + G_recon_loss_B + G_identity_loss_B + G_cam_loss_B

        self.G_adv_loss = G_ad_loss_A + G_ad_loss_B
        self.G_cyc_loss = G_recon_loss_A + G_recon_loss_B
        self.G_idt_loss = G_identity_loss_A + G_identity_loss_B
        self.G_cam_loss = G_cam_loss_A + G_cam_loss_B

        self.Generator_loss = G_loss_A + G_loss_B

        # Face Segmentaion
        if self.args.seg_weight > 0:
            self.Generator_loss += self.G_seg_loss

        self.Generator_loss.backward()
        return G_ad_loss_A, G_recon_loss_A, G_identity_loss_A, G_cam_loss_A, \
               G_ad_loss_B, G_recon_loss_B, G_identity_loss_B, G_cam_loss_B

    def train(self):
        train_writer = SummaryWriter(os.path.join(self.args.result_dir, 'logs'))
        D_losses_A = AverageMeter('D_losses_A', ':.4e')
        D_losses_B = AverageMeter('D_losses_B', ':.4e')
        Discriminator_losses = AverageMeter('Discriminator_losses', ':.4e')
        G_ad_losses_A = AverageMeter('G_ad_losses_A', ':.4e')
        G_recon_losses_A = AverageMeter('G_recon_losses_A', ':.4e')
        G_identity_losses_A = AverageMeter('G_identity_losses_A', ':.4e')
        G_cam_losses_A = AverageMeter('G_cam_losses_A', ':.4e')
        G_ad_losses_B = AverageMeter('G_ad_losses_B', ':.4e')
        G_recon_losses_B = AverageMeter('G_recon_losses_B', ':.4e')
        G_identity_losses_B = AverageMeter('G_identity_losses_B', ':.4e')
        G_cam_losses_B = AverageMeter('G_cam_losses_B', ':.4e')
        Generator_losses = AverageMeter('Generator_losses', ':.4e')
        train_progress = ProgressMeter(self.args.iteration, D_losses_A, D_losses_B, Discriminator_losses,
                                       G_ad_losses_A, G_recon_losses_A, G_identity_losses_A, G_cam_losses_A,
                                       G_ad_losses_B, G_recon_losses_B, G_identity_losses_B, G_cam_losses_B,
                                       Generator_losses, prefix=f"Iteration: ")

        # 用于学习率 decay策略
        start_iter = 1
        mid_iter = self.args.iteration // 2
        lr_rate = self.args.lr / mid_iter
        if self.args.resume:
            model_list = glob(os.path.join(self.args.result_dir, self.args.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.args.result_dir, self.args.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if not self.args.no_decay_flag and start_iter > mid_iter:
                    self.G_optim.param_groups[0]['lr'] -= lr_rate * (start_iter - mid_iter)
                    self.D_optim.param_groups[0]['lr'] -= self.G_optim.param_groups[0]['lr']

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.args.iteration + 1):
            if not self.args.no_decay_flag and step > mid_iter:
                self.G_optim.param_groups[0]['lr'] -= lr_rate
                self.D_optim.param_groups[0]['lr'] -= lr_rate

            real_A, real_B, blur = self.get_batch(mode='train')

            self.gen_train(True)
            (fake_A2B, fake_A2B_cam_logit, _, _,
             fake_A2B2A, _, _,
             fake_B2A, fake_B2A_cam_logit, _, _,
             fake_B2A2B, _, _,
             fake_A2A, fake_A2A_cam_logit, _, _,
             fake_B2B, fake_B2B_cam_logit, _, _) = self.forward(real_A, real_B)

            # Update D
            self.dis_train(True)
            self.D_optim.zero_grad()
            D_loss_A, D_loss_B = self.backward_D(real_A, real_B, fake_A2B, fake_B2A, blur)
            self.D_optim.step()
            # 更新统计量
            D_losses_A.update(D_loss_A.detach().cpu().item())
            D_losses_B.update(D_loss_B.detach().cpu().item())
            Discriminator_losses.update(self.discriminator_loss.detach().cpu().item())

            # Update G
            self.dis_train(False)
            self.G_optim.zero_grad()
            G_ad_loss_A, G_recon_loss_A, G_identity_loss_A, G_cam_loss_A, \
            G_ad_loss_B, G_recon_loss_B, G_identity_loss_B, G_cam_loss_B = \
                self.backward_G(real_A, real_B, fake_A2B, fake_B2A, fake_A2B2A, fake_B2A2B, fake_A2A, fake_B2B,
                                fake_A2B_cam_logit, fake_B2A_cam_logit, fake_A2A_cam_logit, fake_B2B_cam_logit)
            self.G_optim.step()
            # 更新统计量
            G_ad_losses_A.update(G_ad_loss_A.detach().cpu().item(), real_A.size(0))
            G_recon_losses_A.update(G_recon_loss_A.detach().cpu().item(), real_A.size(0))
            G_identity_losses_A.update(G_identity_loss_A.detach().cpu().item(), real_A.size(0))
            G_cam_losses_A.update(G_cam_loss_A.detach().cpu().item(), real_A.size(0))
            G_ad_losses_B.update(G_ad_loss_B.detach().cpu().item(), real_B.size(0))
            G_recon_losses_B.update(G_recon_loss_B.detach().cpu().item(), real_B.size(0))
            G_identity_losses_B.update(G_identity_loss_B.detach().cpu().item(), real_B.size(0))
            G_cam_losses_B.update(G_cam_loss_B.detach().cpu().item(), real_B.size(0))
            Generator_losses.update(self.Generator_loss.detach().cpu().item(), real_A.size(0))

            # 可视化中间结果，计算fid，tensorboard统计
            if step % self.args.print_freq == 0:
                # 可视化中间结果
                self.vis_inference_result(step, train_sample_num=5, test_sample_num=5)
                if step > self.args.ema_start * self.args.iteration:
                    temp = self.genA2B, self.genB2A
                    self.genA2B, self.genB2A = self.genA2B_ema, self.genB2A_ema
                    self.vis_inference_result(step, train_sample_num=5, test_sample_num=5, name='_ema')
                    self.genA2B, self.genB2A = temp
                # 计算fid
                if step % self.args.calc_fid_freq == 0:
                    temp_prefix = train_progress.prefix
                    fid_score_A2B, fid_score_B2A = self.calc_fid_score()
                    train_writer.add_scalar('13_fid_score_A2B', fid_score_A2B, step)
                    train_writer.add_scalar('13_fid_score_B2A', fid_score_B2A, step)
                    train_progress.prefix = f"Iteration: fid: A2B {fid_score_A2B:.4e}, B2A {fid_score_B2A:.4e}"
                    if step > self.args.ema_start * self.args.iteration:
                        temp = self.genA2B, self.genB2A
                        self.genA2B, self.genB2A = self.genA2B_ema, self.genB2A_ema
                        fid_score_A2B, fid_score_B2A = self.calc_fid_score()
                        self.genA2B, self.genB2A = temp
                        train_writer.add_scalar('14_fid_score_A2B_ema', fid_score_A2B, step)
                        train_writer.add_scalar('14_fid_score_B2A_ema', fid_score_B2A, step)
                        train_progress.prefix += f" A2B_ema {fid_score_A2B:.4e}, B2A_ema {fid_score_B2A:.4e}"
                    train_progress.print(step)
                    train_progress.prefix = temp_prefix
                else:
                    train_progress.print(step)

                # 打印统计量
                train_writer.add_scalar('01_D_losses_A', D_losses_A.avg, step)
                train_writer.add_scalar('02_D_losses_B', D_losses_B.avg, step)
                train_writer.add_scalar('03_Discriminator_losses', Discriminator_losses.avg, step)
                train_writer.add_scalar('04_G_ad_losses_A', G_ad_losses_A.avg, step)
                train_writer.add_scalar('05_G_recon_losses_A', G_recon_losses_A.avg, step)
                train_writer.add_scalar('06_G_identity_losses_A', G_identity_losses_A.avg, step)
                train_writer.add_scalar('07_G_cam_losses_A', G_cam_losses_A.avg, step)
                train_writer.add_scalar('08_G_ad_losses_B', G_ad_losses_B.avg, step)
                train_writer.add_scalar('09_G_recon_losses_B', G_recon_losses_B.avg, step)
                train_writer.add_scalar('10_G_identity_losses_B', G_identity_losses_B.avg, step)
                train_writer.add_scalar('11_G_cam_losses_B', G_cam_losses_B.avg, step)
                train_writer.add_scalar('12_Generator_losses', Generator_losses.avg, step)
                train_writer.add_scalar('Learning rate', self.G_optim.param_groups[0]['lr'], step)
                train_writer.flush()
                D_losses_A.reset(), D_losses_B.reset(), Discriminator_losses.reset()
                G_ad_losses_A.reset(), G_recon_losses_A.reset(), G_identity_losses_A.reset()
                G_cam_losses_A.reset(), G_ad_losses_B.reset(), G_recon_losses_B.reset()
                G_identity_losses_B.reset(), G_cam_losses_B.reset(), Generator_losses.reset()

            if step % self.args.save_freq == 0:
                self.save(os.path.join(self.args.result_dir, self.args.dataset, 'model'), step)

            if step % 1000 == 0:
                self.save(self.args.result_dir, step=None, name='_params_latest.pt')
        train_writer.close()

    def calc_fid_score(self):
        self.gen_train(False)
        if self.fid_score is None:
            self.fid_score = FIDScore(self.args.device, batch_size=self.args.fid_batch, num_workers=1)
            self.mean_std_A = self.fid_score.calc_mean_std(self.trainA_data_root)
            self.mean_std_B = self.fid_score.calc_mean_std(self.trainB_data_root)
            self.fid_loaderA = get_loader(DatasetFolder(self.trainA_data_root, self.test_transform), self.args.device,
                                          batch_size=self.args.fid_batch, shuffle=False,
                                          num_workers=self.args.num_workers)
            self.fid_loaderB = get_loader(DatasetFolder(self.trainB_data_root, self.test_transform), self.args.device,
                                          batch_size=self.args.fid_batch, shuffle=False,
                                          num_workers=self.args.num_workers)
            self.fid_score.inception_model.normalize_input = False
        mean_std_A2B = self.fid_score.calc_mean_std_with_gen(lambda batch: self.genA2B(batch)[0].float(),
                                                             self.fid_loaderA)
        fid_score_A2B = self.fid_score.calc_fid(self.mean_std_B, mean_std_A2B)
        mean_std_B2A = self.fid_score.calc_mean_std_with_gen(lambda batch: self.genB2A(batch)[0].float(),
                                                             self.fid_loaderB)
        fid_score_B2A = self.fid_score.calc_fid(self.mean_std_A, mean_std_B2A)
        return fid_score_A2B, fid_score_B2A

    def vis_inference_result(self, step, train_sample_num=5, test_sample_num=5, name=''):
        A2B = np.zeros((self.args.img_size * (7 + self.args.attention_gan), 0, 3))
        B2A = np.zeros((self.args.img_size * (7 + self.args.attention_gan), 0, 3))
        self.gen_train(False), self.dis_train(False)
        for _ in range(train_sample_num):
            real_A, real_B, _ = self.get_batch(mode='train')
            with torch.no_grad(), autocast(enabled=False):
                (fake_A2B, _, fake_A2B_heatmap, fake_A2B_attention,
                 fake_A2B2A, fake_A2B2A_heatmap, fake_A2B2A_attention,
                 fake_B2A, _, fake_B2A_heatmap, fake_B2A_attention,
                 fake_B2A2B, fake_B2A2B_heatmap, fake_B2A2B_attention,
                 fake_A2A, _, fake_A2A_heatmap, fake_A2A_attention,
                 fake_B2B, _, fake_B2B_heatmap, fake_B2B_attention) = \
                    self.forward(real_A, real_B)

            A2B_list = [RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                        cam(tensor2numpy(fake_A2A_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                        cam(tensor2numpy(fake_A2B_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                        cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))
                        ]
            if self.args.attention_gan > 0:
                for i in range(self.args.attention_gan):
                    A2B_list.append(attention_mask(tensor2numpy(fake_A2B_attention[0][i:(i + 1)]),
                                                   self.args.img_size))
            A2B = np.concatenate((A2B, np.concatenate(A2B_list, 0)), 1)

            B2A_list = [RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                        cam(tensor2numpy(fake_B2B_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                        cam(tensor2numpy(fake_B2A_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                        cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))]
            if self.args.attention_gan > 0:
                for i in range(self.args.attention_gan):
                    B2A_list.append(attention_mask(tensor2numpy(fake_B2A_attention[0][i:(i + 1)]),
                                                   self.args.img_size))
            B2A = np.concatenate((B2A, np.concatenate(B2A_list, 0)), 1)

        for _ in range(test_sample_num):
            real_A, real_B, _ = self.get_batch(mode='test')
            with torch.no_grad(), autocast(enabled=False):
                (fake_A2B, _, fake_A2B_heatmap, fake_A2B_attention,
                 fake_A2B2A, fake_A2B2A_heatmap, fake_A2B2A_attention,
                 fake_B2A, _, fake_B2A_heatmap, fake_B2A_attention,
                 fake_B2A2B, fake_B2A2B_heatmap, fake_B2A2B_attention,
                 fake_A2A, _, fake_A2A_heatmap, fake_A2A_attention,
                 fake_B2B, _, fake_B2B_heatmap, fake_B2B_attention) = \
                    self.forward(real_A, real_B)

            A2B_list = [RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                        cam(tensor2numpy(fake_A2A_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                        cam(tensor2numpy(fake_A2B_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                        cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))
                        ]
            if self.args.attention_gan > 0:
                for i in range(self.args.attention_gan):
                    A2B_list.append(attention_mask(tensor2numpy(fake_A2B_attention[0][i:(i + 1)]),
                                                   self.args.img_size))
            A2B = np.concatenate((A2B, np.concatenate(A2B_list, 0)), 1)

            B2A_list = [RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                        cam(tensor2numpy(fake_B2B_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                        cam(tensor2numpy(fake_B2A_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                        cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))]
            if self.args.attention_gan > 0:
                for i in range(self.args.attention_gan):
                    B2A_list.append(attention_mask(tensor2numpy(fake_B2A_attention[0][i:(i + 1)]),
                                                   self.args.img_size))
            B2A = np.concatenate((B2A, np.concatenate(B2A_list, 0)), 1)

        cv2.imwrite(os.path.join(self.args.result_dir, self.args.dataset, 'img', f'A2B{name}_{step:07d}.png'),
                    A2B * 255.0)
        cv2.imwrite(os.path.join(self.args.result_dir, self.args.dataset, 'img', f'B2A{name}_{step:07d}.png'),
                    B2A * 255.0)
        return

    def model_ema(self, step, G_ema, G):
        if step > self.args.ema_start * self.args.iteration:
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, self.args.ema_beta))
        else:
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p_ema)
        for b_ema, b in zip(G_ema.buffers(), G.buffers()):
            b_ema.copy_(b)
        return

    def save(self, root, step, name=None):
        if name is None:
            name = '_params_%07d.pt' % step
        params = {'genA2B': self.genA2B.state_dict(), 'genB2A': self.genB2A.state_dict(),
                  'genA2B_ema': self.genA2B_ema.state_dict(), 'genB2A_ema': self.genB2A_ema.state_dict(),
                  'disGA': self.disGA.state_dict(), 'disGB': self.disGB.state_dict(), 'disLA': self.disLA.state_dict(),
                  'disLB': self.disLB.state_dict()}
        torch.save(params, os.path.join(root, self.args.dataset + name))
        g_params = {'genA2B': self.genA2B.state_dict(), 'genA2B_ema': self.genA2B_ema.state_dict()}
        torch.save(g_params, os.path.join(root, self.args.dataset + f'_g{name}'))

    def load(self, root, step):
        params = torch.load(os.path.join(root, self.args.dataset + '_params_%07d.pt' % step),
                            map_location=torch.device("cpu"))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.genA2B_ema.load_state_dict(params['genA2B_ema'])
        self.genB2A_ema.load_state_dict(params['genB2A_ema'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test(self):
        model_list = glob(os.path.join(self.args.result_dir, '*_g_params_latest.pt'))
        if len(model_list) == 0:
            model_list = glob(os.path.join(self.args.result_dir, self.args.dataset, 'model', '*.pt'))
        if len(model_list) != 0:
            model_list.sort()
            if not (self.args.generator_model and os.path.isfile(self.args.generator_model)):
                self.args.generator_model = model_list[-1]

        if self.args.generator_model and os.path.isfile(self.args.generator_model):
            params = torch.load(self.args.generator_model, map_location=torch.device("cpu"))
            self.genA2B.load_state_dict(params['genA2B'])
            self.genB2A.load_state_dict(params['genB2A'])
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        for n, (real_A, _) in enumerate(self.testA_loader):
            real_A = real_A.to(self.args.device)
            with torch.no_grad():
                fake_A2B, _, fake_A2B_heatmap, fake_A2B_attention = self.genA2B(real_A)
                fake_A2B2A, _, fake_A2B2A_heatmap, fake_A2B2A_attention = self.genB2A(fake_A2B)
                fake_A2A, _, fake_A2A_heatmap, fake_A2A_attention = self.genB2A(real_A)

            A2B_list = [RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                        cam(tensor2numpy(fake_A2A_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                        cam(tensor2numpy(fake_A2B_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                        cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))
                        ]
            if self.args.attention_gan > 0:
                for i in range(self.args.attention_gan):
                    A2B_list.append(attention_mask(tensor2numpy(fake_A2B_attention[0][i:(i + 1)]), self.args.img_size))
            A2B = np.concatenate(A2B_list, 0)
            cv2.imwrite(os.path.join(self.args.result_dir, self.args.dataset, 'test', 'A2B_%d.png' % (n + 1)),
                        A2B * 255.0)

        for n, (real_B, _) in enumerate(self.testB_loader):
            real_B = real_B.to(self.args.device)
            with torch.no_grad():
                fake_B2A, _, fake_B2A_heatmap, fake_B2A_attention = self.genB2A(real_B)
                fake_B2A2B, _, fake_B2A2B_heatmap, fake_B2A2B_attention = self.genA2B(fake_B2A)
                fake_B2B, _, fake_B2B_heatmap, fake_B2B_attention = self.genA2B(real_B)

            B2A_list = [RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                        cam(tensor2numpy(fake_B2B_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                        cam(tensor2numpy(fake_B2A_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                        cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.args.img_size),
                        RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))]
            if self.args.attention_gan > 0:
                for i in range(self.args.attention_gan):
                    B2A_list.append(attention_mask(tensor2numpy(fake_B2A_attention[0][i:(i + 1)]), self.args.img_size))
            B2A = np.concatenate(B2A_list, 0)
            cv2.imwrite(os.path.join(self.args.result_dir, self.args.dataset, 'test', 'B2A_%d.png' % (n + 1)),
                        B2A * 255.0)
