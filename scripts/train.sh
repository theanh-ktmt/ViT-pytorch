#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python3 train.py \
    --name hymenoptera_0 \
    --dataset hymenoptera \
    --model_type ViT-B_16 \
    --pretrained_dir checkpoint/ViT-B_16.npz \
    --num_steps 100 --warmup_steps 10 \
    --train_batch_size 1024 --eval_batch_size 1024
