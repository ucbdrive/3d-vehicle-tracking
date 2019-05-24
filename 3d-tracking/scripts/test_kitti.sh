#!/usr/bin/env bash

# Testing - 3D for tracking | KITTI
set -ex

GPU_ID=${1} # 0

ROOT_DIR="$( cd "$(dirname "$0")"/.. ; pwd -P )"

SESS=623
EP=100
SPLIT=test #train

JSON_PATH=${ROOT_DIR}/data/kitti_tracking/${SPLIT}ing/

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python $ROOT_DIR/mono_3d_estimation.py \
  kitti \
  test \
  --data_split ${SPLIT} \
  --json_path ${JSON_PATH} \
  --is_tracking \
  -j 4 \
  -b 1 \
  --session ${SESS} \
  --epoch ${EP} \
  --n_box_limit 300 \
  --is_normalizing \
  --track_name ${SESS}_${EP}_kitti_${SPLIT}_bdd_roipool_output.pkl \
  --resume ./checkpoint/${SESS}_kitti_checkpoint_${EP}.pth.tar 

