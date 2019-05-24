#!/usr/bin/env bash

set -ex

# Move to parent folder
ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P )"
cd $ROOT

# Install relative modules
pip install -r $ROOT/requirements.txt --user

# Make data path
mkdir -p 'data/gta5_tracking/val/image'
mkdir 'data/gta5_tracking/val/label'
mkdir -p 'data/gta5_tracking/train/image'
mkdir 'data/gta5_tracking/train/label'
mkdir -p 'data/gta5_tracking/test/image'
mkdir 'data/gta5_tracking/test/label'
mkdir 'data/kitti_object'
mkdir 'data/kitti_tracking'

# Make checkpoint, output path
mkdir 'checkpoint'
mkdir 'output'
mkdir -p 'output/gta5_tracking/val/pred'
mkdir -p 'output/gta5_tracking/test/pred'
mkdir -p 'output/kitti_tracking/testing/pred_02'
mkdir -p 'output/kitti_object/testing/pred_02'
mkdir -p 'output/gta5_tracking/train/pred'
mkdir -p 'output/kitti_tracking/training/pred_02'
mkdir -p 'output/kitti_object/training/pred_02'

# Make .so files
cd $ROOT/lib
bash make.sh
cd $ROOT

