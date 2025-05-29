#!/bin/bash
# -------- Step 0 环境 --------
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 打开 DeepSpeed 保存进度日志
export DEEPSPEED_ZERO3_SAVE_16BIT_MODEL=1
export DEEPSPEED_LOG_LEVEL=info
export DEEPSPEED_LOGGING=stdout   # 可选

# -------- Step 1 启动训练 --------
CUDA_VISIBLE_DEVICES=0,1,4,5,6,7 \
accelerate launch \
  --main_process_port=29919 \
  --num_processes=6 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  --use_deepspeed \
  discriminator_train.py | tee train.log