#!/usr/bin/env bash
# 原始训练
CUDA_VISIBLE_DEVICES=0 python main.py --dataset YOUR_DATA_SET --result_dir results --img_size 256

# 直方图匹配
CUDA_VISIBLE_DEVICES=0 python main.py --dataset YOUR_DATA_SET --result_dir results --img_size 256 \
--match_histograms --match_mode hsl --match_prob 0.5 --match_ratio 1.0

# 背景不变
CUDA_VISIBLE_DEVICES=0 python main.py --dataset YOUR_DATA_SET --result_dir results --img_size 256 \
--seg_weight 50 --seg_rand_mask 0.8

# se + blur
CUDA_VISIBLE_DEVICES=0 python main.py --dataset YOUR_DATA_SET --result_dir results --img_size 256 --use_se --has_blur

# attention + 原图
CUDA_VISIBLE_DEVICES=0 python main.py --dataset YOUR_DATA_SET --result_dir results --img_size 256 \
--attention_gan 3 --attention_input

# 损失权重调整
CUDA_VISIBLE_DEVICES=0 python main.py --dataset YOUR_DATA_SET --result_dir results --img_size 256 \
--adv_weight 0.8 --forward_adv_weight 2 --cycle_weight 5 --identity_weight 5

# cpu调试
python main.py --dataset YOUR_DATA_SET --result_dir results --img_size 256 --device cpu --num_workers 0

# 复杂cpu测试：直方图匹配 + 背景不变 + se + blur + attention + 原图
python main.py --dataset YOUR_DATA_SET --result_dir results --img_size 256 --device cpu --num_workers 0 \
--match_histograms --match_mode hsl --match_prob 0.5 --match_ratio 1.0 \
--seg_weight 50 --seg_rand_mask 0.8 --use_se --has_blur --attention_gan 3 --attention_input \
--adv_weight 0.8 --forward_adv_weight 2 --cycle_weight 20 --identity_weight 5
