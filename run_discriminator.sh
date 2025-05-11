#!/bin/bash

# ----------- Step 0: (可选) 清理缓存，避免显存没释放干净 -----------
echo "Clearing GPU memory..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

nvidia-smi

# ----------- Step 1: 启动 discriminator_train.py 训练 -----------
echo "Starting training..."
CUDA_VISIBLE_DEVICES="1,3,7"

accelerate launch \
  --main_process_port=29919 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  --use_deepspeed \
  discriminator_train.py

# ----------- Step 2: 完成提示 -----------
echo "Training finished!"
