# Joint Monocular 3D Vehicle Detection and Tracking

![](../imgs/teaser.gif)

We present a novel framework that jointly detects and tracks 3D vehicle bounding boxes.
Our approach leverages 3D pose estimation to learn 2D patch association overtime and uses temporal information from tracking to
obtain stable 3D estimation.
<br/>

**Joint Monocular 3D Vehicle Detection and Tracking**
<br/>
[Hou-Ning Hu](https://eborboihuc.github.io/), 
[Qi-Zhi Cai](https://www.linkedin.com/in/qi-zhi-cai-7130a4155), 
[Dequan Wang](https://dequan.wang/), 
[Ji Lin](http://linji.me/), 
[Min Sun](https://aliensunmin.github.io/), 
[Philipp Krähenbühl](https://www.philkr.net/), 
[Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), 
[Fisher Yu](https://www.yf.io/).
<br/>
In ICCV, 2019.

[Paper](https://arxiv.org/abs/1811.10742)
[Website](https://eborboihuc.github.io/Mono-3DT/)


## Quick start
To get started as quickly as possible, follow the instructions in this section. 
This should allow you train a model from scratch, test our pretrained models, and reproduce our evaluation results.
For more detailed instructions, please refer to [`DOCUMENTATION.md`](DOCUMENTATION.md).

### Data Preparation

For a quick start, we suggest using GTA `val` set as a starting point. You can get all needed data via the following script.

```bash
# We recommand using GTA `val` set (using `mini` flag) to get familiar with the data pipeline first, then using `all` flag to obtain all the data
python loader/download.py mini
```

Or you can also download each files one by one:

- GTA Dataset
Download and extract our GTA dataset and put it under `./3d-tracking/data/gta5_tracking/{split}/{type}`.
Where _split_ = {train, val, test} and _type_ = {image, label}
```bash
python loader/download.py {type} {split}
```

- Detection
Download our detection results and put it under `./3d-tracking/data/gta5_tracking/`.
```bash
python loader/download.py detection
```
Generates BDD format meta data using `python loader/gen_pred.py gta val`

- Pre-train Model
Download our pretrain model and put it under `./3d-tracking/checkpoint/`.
```bash
python loader/download.py checkpoint
```

### Execution

- Installation and dataset preparation are both finished
- Given object proposal bounding boxes and 3D center from faster-rcnn.pytorch directory

For running a whole pipeline (3D estimation and tracking):
```bash
# Step 00 - Data Preprocessing
# Collect features into json files (check variables in the code)
python loader/gen_pred.py gta val 

# And similarly,
python loader/gen_pred.py gta test

# For KITTI, it's the same procedure
python loader/gen_dataset.py kitti train
python loader/gen_pred.py kitti train

# Step 01 - 3D Estimation
# Running single task scripts mentioned below and training by yourself
# or alternatively, using multi-GPUs and multi-processes to run through all 100 sequences
python run_estimation.py gta val --session 616 --epoch 030

# Step 02 - 3D Tracking and Evaluation
# 3D helps tracking part. For tracking evaluation, 
# using multi-GPUs and multi-processes to run through all 100 sequences
python run_tracking.py gta val --session 616 --epoch 030

# Step 03 - 3D AP Evaluation
# Convert tracking output to evaluation format
python tools/convert_estimation_bdd.py gta val --session 616 --epoch 030
python tools/convert_tracking_bdd.py gta val --session 616 --epoch 030

# Evaluation of 3D Estimation
python tools/eval_dep_bdd.py gta val --session 616 --epoch 030

# 3D helps Tracking part
python tools/eval_mot_bdd.py --gt_path output/616_030_gta_val_set --pd_path output/616_030_gta_val_set/kf3doccdeep_age20_aff0.1_hit0_100m_803

# Tracking helps 3D part
cd tools/object-ap-eval/
python test_det_ap.py gta val --session 616 --epoch 030
```

> Note: If facing `ModuleNotFoundError: No module named 'utils'` problem, please add `PYTHONPATH=.` before `python {scripts} {arguments}`.


For running a whole pipeline of 3D estimation (training and testing):
```bash
# In Step 01, run the following scripts instead
# Make sure your args for training, testing

# Training (Optional):
bash scripts/train_gta.sh

# Testing
bash scripts/test_gta.sh
```

## License
This work is licensed under BSD 3-Clause License. See [LICENSE](../LICENSE) for details. 
Third-party datasets and tools are subject to their respective licenses.

If you use our code/models in your research, please cite our paper:
```
@inproceedings{Hu3DT19,
author = {Hu, Hou-Ning and Cai, Qi-Zhi and Wang, Dequan and Lin, Ji and Sun, Min and Krähenbühl, Philipp and Darrell, Trevor and Yu, Fisher},
title = {Joint Monocular 3D Vehicle Detection and Tracking},
journal = {ICCV},
year = {2019}
}
```

## Acknowledgements
We thank [pymot](https://github.com/Videmo/pymot) for their MOT evaluation tool and [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python) for the 3D AP calculation tool.
