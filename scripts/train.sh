#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python3 train.py \
    --name baseline \
    --dataset hymenoptera \
    --model_type ViT-B_16 \
    --pretrained_dir checkpoint/ViT-B_16.npz \
    --num_steps 100 --eval_every 100 --warmup_steps 10 \
    --train_batch_size 1024 --eval_batch_size 1024 \
    --k-fold 5 --output_dir output/baseline
