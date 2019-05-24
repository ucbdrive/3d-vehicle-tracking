#!/usr/bin/env bash

# Dataset path (Change _ROOT to your own path)
USER=$(whoami)
_DIR="Det"
_ROOT="/data4/${USER}/Workspace"

# Link large space folder to vis to save storage on main disk
mkdir -p ${_ROOT}/${_DIR}/vis/
ln -s ${_ROOT}/${_DIR}/vis/
mkdir -p ${_ROOT}/${_DIR}/models/res101/gta_det/
mkdir -p ${_ROOT}/${_DIR}/models/res101/kitti/
ln -s ${_ROOT}/${_DIR}/models/

# (optional) Copy a detection pretrained model
echo "Please move faster_rcnn_200_14_18895.pth to models/res101/gta_det/"
echo "Please move faster_rcnn_300_100_175.pth to models/res101/kitti/"

