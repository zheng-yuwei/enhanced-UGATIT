# -*- coding: utf-8 -*-
import os
import argparse

import torch
import torch.backends.cudnn

import utils
from UGATIT import UGATIT


def parse_args():
    """parsing and configuration"""
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'video', 'camera', 'img_dir'],
                        help='[train | test | video | camera | img_dir ]')
    parser.add_argument('--light', type=int, default=-1,
                        help='[U-GAT-IT full version / U-GAT-IT light version]，求gamma和beta的MLP的输入尺寸')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='每个 DataLoader 的进程数')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--no_decay_flag', action='store_false', help='在中间iteration时，使用学习率下降策略，默认使用')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=float, default=1, help='Weight for GAN，建议值：0.8')
    parser.add_argument('--forward_adv_weight', type=float, default=1, help='前向对抗损失的权重，建议值：2')
    parser.add_argument('--backward_adv_weight', type=float, default=1, help='后向对抗损失的权重，建议值：1')
    parser.add_argument('--cycle_weight', type=float, default=10, help='Weight for Cycle，建议值：3')
    parser.add_argument('--identity_weight', type=float, default=10, help='Weight for Identity，建议值：1.5')
    parser.add_argument('--cam_weight', type=float, default=1000, help='Weight for CAM，建议值：1000')
    parser.add_argument('--seg_weight', type=float, default=-1, help='Weight for Face Segmentation')
    parser.add_argument('--seg_rand_mask', type=float, default=0.8, help='将分割出来的背景随机填充颜色，再训练D')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
    # 增强U-GAT-IT选项
    parser.add_argument('--aug_prob', type=float, default=0.2, help='对数据应用 resize & crop 数据增强的概率，建议值，<=0.2')
    parser.add_argument('--sn', action='store_false', help='默认D网络使用sn，建议使用，tf版本使用了')
    parser.add_argument('--match_histograms', action='store_true', help='默认不使用直方图匹配，建议使用')
    parser.add_argument('--match_mode', type=str, default='hsl', help='默认直方图匹配使用hsl')
    parser.add_argument('--match_prob', type=float, default=0.5, help='从 B->A 进行直方图匹配的概率，否则 A->B 进行直方图匹配')
    parser.add_argument('--match_ratio', type=float, default=1.0, help='直方图匹配的比例')
    parser.add_argument('--has_blur', action='store_true', help='默认不使用模糊数据增强D网络，建议使用')
    parser.add_argument('--use_se', action='store_true', help='resblock是否使用se-block，可以使用')
    parser.add_argument('--attention_gan', type=int, default=0, help='attention_gan，可以尝试')
    parser.add_argument('--attention_input', action='store_true', help='attention_gan时，是否把输入加入做attention，可以尝试')
    parser.add_argument('--use_deconv', action='store_true', help='生成网络的上采样是否使用deconv，不建议使用')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--resume', action='store_true', help='是否继续最后的一次训练')

    # 测试
    parser.add_argument('--generator_model', type=str, default='', help='测试的A2B生成器路径')
    parser.add_argument('--video_path', type=str, default='', help='测试的A2B生成器路径所用的视频路径')
    parser.add_argument('--img_dir', type=str, default='', help='测试的A2B生成器路径所用的图像文件夹路径')

    return check_args(parser.parse_args())


def check_args(args):
    """checking arguments"""
    if args.phase in ('video', 'camera', 'img_dir'):
        return args
    # --result_dir
    utils.check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    utils.check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    utils.check_folder(os.path.join(args.result_dir, args.dataset, 'test'))
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'
    return args


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # 视频/摄像头/图像文件夹 测试
    if args.phase in ('video', 'camera', 'img_dir'):
        from videos import test_video
        from networks import ResnetGenerator
        torch.set_flush_denormal(True)
        # 定义及加载生成模型
        generator = ResnetGenerator(input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res,
                                    img_size=args.img_size, args=args).to(args.device)
        if args.generator_model == '':
            raise ValueError('No define A2B G model path!')
        params = torch.load(args.generator_model, map_location=torch.device("cpu"))
        generator.load_state_dict(params['genA2B'])
        # 模型测试
        tester = test_video.VideoTester(args, generator)
        if args.phase == 'video':
            assert args.video_path and os.path.exists(args.video_path), f'video path ({args.video_path}) error!'
            tester.video(args.video_path)
        elif args.phase == 'camera':
            tester.camera()
        else:
            assert args.img_dir and os.path.exists(args.img_dir), f'image directory ({args.img_dir}) error!'
            tester.image_dir(args.img_dir)
        return

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
    utils.setup_seed(0)

    # open session
    gan = UGATIT(args)

    # build graph
    gan.build_model()

    if args.phase == 'train':
        gan.train()
        print(" [*] Training finished!")
        args.phase = 'test'

    if args.phase == 'test':
        torch.set_flush_denormal(True)
        gan.test()
        print(" [*] Test finished!")


if __name__ == '__main__':
    main()
