# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional
from torch.nn.parameter import Parameter


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, args=None):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.args = args

        self.light = args.light
        self.attention_gan = args.attention_gan
        self.attention_input = args.attention_input
        self.use_se = args.use_se
        self.use_deconv = args.use_deconv

        # 下采样模块：特征抽取、下采样、Bottleneck(resnet-block)特征编码
        DownBlock = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.InstanceNorm2d(ngf, affine=True),
                     nn.ReLU(True)]
        # 下采样
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2, affine=True),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False, use_se=self.use_se)]

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=True)
        # self.gmp_fc = self.gap_fc  # tf版本
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=True)  # pytorch版本
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # 生成 gamma 和 beta，小模型和大模型
        if self.light > 0:
            FC = [nn.Linear(self.light * self.light * ngf * mult, ngf * mult, bias=True)]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=True)]
        FC += [nn.ReLU(True),
               nn.Linear(ngf * mult, ngf * mult, bias=True),
               nn.ReLU(True)]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=True)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=True)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaLINBlock(ngf * mult, use_bias=True, use_se=self.use_se))
        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if self.use_deconv:
                up_sample = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                                padding=1, output_padding=1, bias=True)]
            else:
                up_sample = [nn.Upsample(scale_factor=2, mode='nearest'),
                             nn.ReflectionPad2d(1),
                             nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=True)]
            UpBlock2 += up_sample
            UpBlock2 += [LIN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        if self.attention_gan > 0:
            UpBlock_attention = []
            mult = 2 ** n_downsampling
            for i in range(n_blocks):
                UpBlock_attention += [ResnetBlock(ngf * mult, use_bias=False, use_se=self.use_se)]
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - i)
                if self.use_deconv:
                    up_sample = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                                    padding=1, output_padding=1, bias=True)]
                else:
                    up_sample = [nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3,
                                           stride=1, padding=0, bias=True)]
                UpBlock_attention += up_sample
                UpBlock_attention += [LIN(int(ngf * mult / 2)),
                                      nn.ReLU(True)]
            UpBlock_attention += [nn.Conv2d(ngf, self.attention_gan, kernel_size=1, stride=1, padding=0, bias=True),
                                  nn.Softmax(dim=1)]
            self.UpBlock_attention = nn.Sequential(*UpBlock_attention)
            if self.attention_input:
                output_nc *= (self.attention_gan - 1)
            else:
                output_nc *= self.attention_gan

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=True),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, input_x):
        attention = None
        x = self.DownBlock(input_x)
        # cam作为attention加权x
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = sum(list(self.gap_fc.parameters()))
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = sum(list(self.gmp_fc.parameters()))
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)
        # 生成上采样模块的adaILN的gamma和beta
        if self.light > 0:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, self.light)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        new_x = x
        for i in range(self.n_blocks):
            new_x = getattr(self, 'UpBlock1_' + str(i + 1))(new_x, gamma, beta)

        out = self.UpBlock2(new_x)

        if self.attention_gan > 0:
            attention = self.UpBlock_attention(x)
            batch_size, attention_ch, height, width = attention.shape
            if self.attention_input:
                out = torch.cat([input_x, out], dim=1)
            out = out.view(batch_size, 3, attention_ch, height, width)
            out = out * attention.view(batch_size, 1, attention_ch, height, width)
            out = out.sum(dim=2)

        return out, cam_logit, heatmap, attention


class ChannelSELayer(nn.Module):
    def __init__(self, in_size, reduction=4, min_hidden_channel=8):
        super(ChannelSELayer, self).__init__()

        hidden_channel = max(in_size // reduction, min_hidden_channel)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, hidden_channel, kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, in_size, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.se(x) * x


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)
        return squeeze_tensor * input_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks,
        MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=4):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        attention = self.cSE(input_tensor) + self.sSE(input_tensor)
        return attention


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias, use_se=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim, affine=True),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim, affine=True)]
        if use_se:
            conv_block += [ChannelSpatialSELayer(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaLINBlock(nn.Module):
    def __init__(self, dim, use_bias, use_se=False):
        super(ResnetAdaLINBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = AdaLIN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = AdaLIN(dim)
        self.use_se = use_se
        if use_se:
            self.se = ChannelSpatialSELayer(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        if self.use_se:
            out = self.se(out)
        return out + x


class AdaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaLIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, in_x, gamma, beta):
        in_mean, in_var = torch.mean(in_x, dim=[2, 3], keepdim=True), torch.var(in_x, dim=[2, 3], keepdim=True)
        out_in = (in_x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(in_x, dim=[1, 2, 3], keepdim=True), torch.var(in_x, dim=[1, 2, 3], keepdim=True)
        out_ln = (in_x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(in_x.shape[0], -1, -1, -1) * out_in + (
                    1 - self.rho.expand(in_x.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class LIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, in_x):
        in_mean, in_var = torch.mean(in_x, dim=[2, 3], keepdim=True), torch.var(in_x, dim=[2, 3], keepdim=True)
        out_in = (in_x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(in_x, dim=[1, 2, 3], keepdim=True), torch.var(in_x, dim=[1, 2, 3], keepdim=True)
        out_ln = (in_x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(in_x.shape[0], -1, -1, -1) * out_in + (
                    1 - self.rho.expand(in_x.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(in_x.shape[0], -1, -1, -1) + self.beta.expand(in_x.shape[0], -1, -1, -1)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5, with_sn=True):
        super(Discriminator, self).__init__()
        conv1 = nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(conv1) if with_sn else conv1,
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            conv = nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(conv) if with_sn else conv,
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        conv2 = nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(conv2) if with_sn else conv2,
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        linear_gmp = nn.Linear(ndf * mult, 1, bias=True)
        linear_gap = nn.Linear(ndf * mult, 1, bias=True)
        self.gmp_fc = nn.utils.spectral_norm(linear_gmp) if with_sn else linear_gmp
        self.gap_fc = nn.utils.spectral_norm(linear_gap) if with_sn else linear_gap
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        conv3 = nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=True)
        self.conv = nn.utils.spectral_norm(conv3) if with_sn else conv3

        self.model = nn.Sequential(*model)

    def forward(self, in_x):
        x = self.model(in_x)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        # gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = sum(list(self.gap_fc.parameters()))
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        # gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = sum(list(self.gmp_fc.parameters()))
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):

    def __init__(self, min_num, max_num, module_type=None):
        self.clip_min = min_num
        self.clip_max = max_num
        self.module_type = module_type
        assert min_num < max_num

    def __call__(self, module):
        if (self.module_type is None and hasattr(module, 'rho')) or (type(module) == self.module_type):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
