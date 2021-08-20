# -*- coding: utf-8 -*-
"""
CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset YOUR_DATA_SET --result_dir results --img_size 384 --device cuda:0 --num_workers 1 \
# G网络计算AdaLIN的大小
--light -1 \
# 指标打印
--print_freq 1000 --calc_fid_freq 10000 --save_freq 100000 \
# 模型ema
--ema_start 0.5 --ema_beta 0.9999 \
# 分割损失
--seg_fix_weight 50 --seg_fix_glass_mouth --seg_D_mask --seg_G_detach --seg_D_cam_fea_mask --seg_D_cam_inp_mask \
--cam_D_weight -1 \
# attention gan
--attention_gan 2 --attention_input \
# 不同损失项权重
--adv_weight 1.0 --forward_adv_weight 1 --cycle_weight 10 --identity_weight 10 \
# 直方图匹配
--match_histograms --match_mode hsl --match_prob 0.5 --match_ratio 1.0 \
# 模糊、se
--has_blur --use_se

e.g.
CUDA_VISIBLE_DEVICES=0 python main.py --dataset transfer/yuwei/styleGAN/dy_cartoon/dy_cartoon.tar \
--light 32 --result_dir transfer/yuwei/styleGAN/oilpaint_result/pai2/dy_only \
--img_size 384 --device cuda:0 --num_workers 4
"""
import os
import time
import datetime
import argparse

import torch
import torch.backends.cudnn

import utils
from UGATIT import UGATIT

VIDEO_EXT = ('.ts', '.mp4')


def parse_args():
    """parsing and configuration"""
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'val', 'test', 'video', 'video_dir', 'camera', 'img_dir', 'generate'],
                        help='训练/验证/测试/视频/视频文件夹/摄像头/图像文件夹/以对齐的人脸图像文件夹 模式')
    parser.add_argument('--light', type=int, default=-1,
                        help='[U-GAT-IT full version / U-GAT-IT light version]，求gamma和beta的MLP的输入尺寸')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')

    parser.add_argument('--ema_start', type=float, default=0.5, help='start ema after ratio of --iteration')
    parser.add_argument('--ema_beta', type=float, default=0.9999, help='ema gamma for genA2B/B2A, 0.9999^10000=0.37')
    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='每个 DataLoader 的进程数')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--calc_fid_freq', type=int, default=10000, help='The number of fid print freq')
    parser.add_argument('--fid_batch', type=int, default=50, help='计算fid score时的batch size')
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

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_global_dis', type=int, default=7, help='The number of global discriminator layer')
    parser.add_argument('--n_local_dis', type=int, default=5, help='The number of local discriminator layer')
    parser.add_argument('--img_size', type=int, default=384, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--result_dir', type=str, default='project/results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda:0'], help='Set gpu mode; [cpu, cuda:0]')  # noqa, E501
    parser.add_argument('--resume', action='store_true', help='是否继续最后的一次训练')

    # 增强U-GAT-IT选项
    parser.add_argument('--aug_prob', type=float, default=0.2, help='对数据应用 resize & crop 数据增强的概率，建议值，<=0.2')
    parser.add_argument('--sn', action='store_false', help='默认D网络使用sn，建议使用，tf版本使用了')
    parser.add_argument('--has_blur', action='store_true', help='默认不使用模糊数据增强D网络，建议使用')
    parser.add_argument('--tv_loss', action='store_true', help='是否对生成图像使用TVLoss，默认不适用')
    parser.add_argument('--tv_weight', type=float, default=1.0, help='Weight for TVLoss，建议值：1.0')
    parser.add_argument('--use_se', action='store_true', help='resblock是否使用se-block，可以使用')
    parser.add_argument('--attention_gan', type=int, default=0, help='attention_gan，可以尝试')
    parser.add_argument('--attention_input', action='store_true', help='attention_gan时，是否把输入加入做attention，可以尝试')
    parser.add_argument('--cam_D_weight', type=float, default=1, help='判别器的CAM分类损失项权重，建议值：1')
    parser.add_argument('--cam_D_attention', action='store_false', help='是否使用判别器的CAM注意力机制，默认使用')
    # 直方图匹配
    parser.add_argument('--match_histograms', action='store_true', help='默认不使用直方图匹配，两个域真实域存在部分分布差异可尝试')
    parser.add_argument('--match_mode', type=str, default='hsl', help='默认直方图匹配使用hsl')
    parser.add_argument('--match_prob', type=float, default=0.5, help='从 B->A 进行直方图匹配的概率，否则 A->B 进行直方图匹配')
    parser.add_argument('--match_ratio', type=float, default=1.0, help='直方图匹配的比例')
    # 固定背景选项
    parser.add_argument('--hard_seg_edge', action='store_true', help='分割边界是否为硬边界，默认为软边界')
    parser.add_argument('--seg_fix_weight', type=float, default=-1, help='对生成图像的分割区域与原图做L1损失项的权重，建议值：50')
    parser.add_argument('--seg_fix_glass_mouth', action='store_true', help='分割是否固定眼镜边框和嘴巴内部(作为背景)，默认不固定')
    parser.add_argument('--seg_D_mask', action='store_true', help='只计算分割mask区域的判别损失，默认都计算')
    parser.add_argument('--seg_G_detach', action='store_true', help='对生成图像的非分割mask区域做detach，默认不detach')
    parser.add_argument('--seg_D_cam_fea_mask', action='store_true', help='将判别器cam的feature map做mask替换，默认不替换')
    parser.add_argument('--seg_D_cam_inp_mask', action='store_true', help='将输入给判别器cam的图像做mask替换，默认不替换')

    # 测试
    parser.add_argument('--generator_model', type=str, default='', help='测试的A2B生成器路径')
    parser.add_argument('--video_path', type=str, default='', help='测试的A2B生成器路径所用的视频路径')
    parser.add_argument('--img_dir', type=str, default='', help='测试的A2B生成器路径所用的图像文件夹路径')

    return check_args(parser.parse_args())


def check_args(args):
    """checking arguments"""
    utils.check_folder(args.result_dir)
    utils.Logger(file_name=os.path.join(args.result_dir, f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"),
                 file_mode='a', should_flush=True)
    if args.cam_D_weight <= 0:
        args.cam_D_attention = False
        print(f'can not use D cam attention while D cam weight = {args.cam_D_weight} <= 0')

    if args.phase in ('video', 'video_dir', 'camera', 'img_dir', 'generate'):
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
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)
    print(args)

    # 视频/摄像头/图像文件夹 测试
    if args.phase in ('video', 'video_dir', 'camera', 'img_dir', 'generate'):
        if args.generator_model == '':
            raise ValueError('No define A2B G model path!')
        from videos import test_video
        from networks import ResnetGenerator
        torch.set_flush_denormal(True)
        # 定义及加载生成模型
        generator = ResnetGenerator(input_nc=3, output_nc=3, ngf=args.ch, n_blocks=args.n_res,
                                    img_size=args.img_size, args=args)
        params = torch.load(args.generator_model, map_location=torch.device("cpu"))
        generator.load_state_dict(params['genA2B_ema'])
        # 模型测试
        tester = test_video.VideoTester(args, generator)
        if args.phase in ('video', 'video_dir'):
            assert args.video_path and os.path.exists(args.video_path), f'video path ({args.video_path}) error!'
            if args.phase == 'video_dir':
                video_paths = [os.path.join(args.video_path, video_name) for video_name in os.listdir(args.video_path)
                               if video_name.endswith(VIDEO_EXT)]
            else:
                video_paths = [args.video_path]
            for video_path in video_paths:
                print(f'generating video: {video_path} ...')
                tester.video(video_path)
        elif args.phase == 'camera':
            tester.camera()
        elif args.phase == 'img_dir':
            assert args.img_dir and os.path.exists(args.img_dir), f'image directory ({args.img_dir}) error!'
            tester.image_dir(args.img_dir)
        elif args.phase == 'generate':
            assert args.img_dir and os.path.exists(args.img_dir), f'image directory ({args.img_dir}) error!'
            tester.generate_images(args.img_dir)
        else:
            raise Exception(f'unknown phase: {args.phase}')
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
    
    return


if __name__ == '__main__':
    main()
