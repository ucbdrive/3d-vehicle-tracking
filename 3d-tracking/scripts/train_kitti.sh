#!/usr/bin/env bash

# Training | KITTI
set -ex

# Adjust batchsize to fit GPU memory ('-b 4' suits 1 P100)
GPU_ID=${1} # 0,1,2,3,4

ROOT_DIR="$( cd "$(dirname "$0")"/.. ; pwd -P )"

JSON_PATH=${ROOT_DIR}/data/kitti_tracking/training/

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python $ROOT_DIR/mono_3d_estimation.py \
  kitti \
  train \
  --is_tracking \
  --json_path ${JSON_PATH} \
  -j 4 \
  -b 30 \
  --session 630 \
  --n_box_limit 40 \
  --percent 100 \
  --is_normalizing \
  --use_tfboard \
  # If you want to resume from pre-trained model
  # Uncomment the following two lines
  #--resume ./checkpoint/621_kitti_checkpoint_100.pth.tar \
  #--start_epoch 0
