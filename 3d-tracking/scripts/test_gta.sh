#!/usr/bin/env bash

# Testing - 3D for tracking | GTA
set -ex

GPU_ID=${1} # 0

ROOT_DIR="$( cd "$(dirname "$0")"/.. ; pwd -P )"

SESS=616
EP=030
SPLIT=val
# NOTE: This might eat up too much memory
#JSON_PATH=${ROOT_DIR}/data/gta5_tracking/${SPLIT}/
# Can run one sequence at a time instead
JSON_PATH=${ROOT_DIR}/data/gta5_tracking/val/label/rec_10090911_clouds_21h53m_x-968y-1487tox2523y214_bdd.json

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python $ROOT_DIR/mono_3d_estimation.py \
  gta \
  test \
  --data_split ${SPLIT} \
  --json_path ${JSON_PATH} \
  -j 4 \
  -b 1 \
  --session ${SESS} \
  --epoch ${EP} \
  --n_box_limit 300 \
  --track_name ${SESS}_${EP}_gta_${SPLIT}_bdd_roipool_output.pkl \
  --resume ./checkpoint/${SESS}_gta_checkpoint_${EP}.pth.tar 
