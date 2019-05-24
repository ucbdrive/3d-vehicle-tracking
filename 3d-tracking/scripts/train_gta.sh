#!/usr/bin/env bash

# Training - 3D for tracking | GTA
set -ex

# Adjust batchsize to fit GPU memory ('-b 4' suits 1 P100)
GPU_ID=${1} # 0,1,2,3,4

ROOT_DIR="$( cd "$(dirname "$0")"/.. ; pwd -P )"

JSON_PATH=${ROOT_DIR}/data/gta5_tracking/train/

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python $ROOT_DIR/mono_3d_estimation.py \
  gta \
  train \
  --json_path ${JSON_PATH} \
  -j 4 \
  -b 20 \
  --session 620 \
  --n_box_limit 40 \
  --percent 100 \
  --has_val \
  --use_tfboard

