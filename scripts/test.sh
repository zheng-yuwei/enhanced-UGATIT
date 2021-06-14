#!/usr/bin/env bash
# 模型测试视频 + attention 模型 + scse
python main_light.py --phase video --generator_model checkpoints/big_normal_100w.pt \
--use_se --attention_gan 3 --attention_input --device cpu --img_size 384 --light 32
