# Documentation

## Files

These files are for our monocular 3D Tracking pipeline:

`requirements.txt` installing list of requirements

`_init_paths.py` import pymot path for tracking evaluation

`mono_3d_estimation.py` train and test on roipool feature based 3d and tracking

`mono_3d_tracking.py` compute correlation by IoU and deep feature, and evaluate tracking result via `pymot/`

`motion_lstm.py` training script for lstm motion model

scripts/

`init.sh` builds up the environment

`train_gta.sh` example training script for gta

`train_kitti.sh` example training script for kitti

`test_gta.sh` example testing script for gta

`test_kitti.sh` example testing script for kitti

model/

`dla.py`, `dla_up.py` defines base model of 3D estimation

`model.py` defines our 3D network architecture

`model_cen.py` defines our 3D network architecture with an extra node predicts projection of 3D center

`motion_model.py` defines our lstm architecture

`tracker_model.py` defines the kalman filter and lstm tracker

`tracker_3d.py`, `tracker_2d.py` for computing correlation of objects between frames

loader/

`dataset.py` data loader for our mono_3d_estimation.py

`dla_dataset.py` data format for dla.py

`gen_dataset.py` generate image based features for KITTI and GTA with detection bbox match to gt, BDD format in json files

`gen_pred.py` prediction placeholder data generation script for BDD format

utils/

`config.py` defines cfg, including KITTI and GTA

`bdd_helper.py` helper for generating BDD format

`network_utils.py` Utility functions for 3D estimation

`tracking_utils.py` Utility functions for tracking, 3D transformation

`plot_utils.py` Utility functions for plot 3D bounding boxes and bird's eye view

`labels.py` defines dataformat for show_labels.py

tools/

`convert_estimation_bdd.py` converts 3D estimation to BDD format

`convert_tracking_bdd.py` converts tracking results to BDD format

`eval_dep_bdd.py` evaluates 3D estimation results using metrics for depth, orientation and center

`eval_mot_bdd.py` evaluates tracking results using pymot/

`plot_tracking.py` visualization of 3D tracklets

`show_labels.py` show ground truth label

`visualize_kitti.py` visualize and convert format to kitti txt files for devkit/

`devkit/` official kitti developement kit to evaluate tracking result

`object-ap-eval/` 3D AP evaluation tool

`pymot/` multiple object tracking evaluation tool


lib/

`make.sh` create execution files



## Dataset

```bash
''' 
Using BDD format with json files
'''

# GTA train
# 447256/457467 frames
gta_data/gta_train_list.json

# For each sequences
gta_data/train/{}_bdd.json

# GTA val with detection available
# 46250/45152 frames
gta_data/gta_val_list.json

# For each sequences
gta_data/val/{}_bdd.json

# GTA test with detection available
# 184459/181034 frames
gta_data/gta_test_list.json

# For each sequences
gta_data/test/{}_bdd.json
```

## Checkpoint

Checkpoint filename is using the following format.
```bash
{SESSION}_{DATASET}_checkpoint_{EPOCH}.pth.tar
```

```bash
# For mono_3d_estimation.py
# GTA
./checkpoint/616_gta_checkpoint_030.pth.tar
# KITTI
./checkpoint/623_kitti_checkpoint_100.pth.tar


# For mono_3d_tracking.py
./checkpoint/803_kitti_300_linear.pth
```

