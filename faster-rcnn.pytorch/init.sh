#!/usr/bin/env bash

set -ex

_PWD=$(pwd)
_DATA_GTA=${_PWD}'/../3d-tracking/data/gta5_tracking/'
_DATA_KITTI_T=${_PWD}'/../3d-tracking/data/kitti_tracking/'
_DATA_KITTI_D=${_PWD}'/../3d-tracking/data/kitti_object/'


# make .so files
cd lib/
./make.sh
cd ${_PWD}

# download pretrain model
mkdir -p data/pretrained_model
cd data/pretrained_model
wget https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth
mv resnet101-5d3b4d8f.pth resnet101_caffe.pth
cd ${_PWD}

# Link GTA and KITTI dataset to data
# get gta data
cd data/
ln -s ${_DATA_GTA} gta5_tracking
echo "mv GTA dataset (train, val, test) to data/gta5_tracking"

# get kitti data
cd ${_PWD}/data/
ln -s ${_DATA_KITTI_T} kitti_tracking
echo "mv KITTI tracking dataset (train, test) to data/kitti_tracking"

# If you downloaded detection dataset, uncomment the following two lines
#ln -s ${_DATA_KITTI_D} kitti_object 
#echo "mv KITTI detection dataset (train, test) to data/kitti_object"

# create folder vis/ (detection output) and models (model weights)
cd ${_PWD}
mkdir vis/
mkdir -p models/res101/gta_det/
mkdir models/res101/kitti/

# (optional) Copy a detection pretrained model
echo "Please move faster_rcnn_200_14_18895.pth to models/res101/gta_det/"
echo "Please move faster_rcnn_300_100_175.pth to models/res101/kitti/"
