# Joint Monocular 3D Vehicle Detection and Tracking

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ucbdrive/3d-vehicle-tracking.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ucbdrive/3d-vehicle-tracking/context:python)

![](imgs/teaser.gif)

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

## Prerequisites

- Linux (tested on Ubuntu 16.04.4 LTS)
- Python 3.6.9
    - `3.6.4` tested
    - `3.6.9` tested
- PyTorch 1.3.1 
    - `1.0.0` (with CUDA 9.0, torchvision 0.2.1)
    - `1.1.0` (with CUDA 9.0, torchvision 0.3.0)
    - `1.3.1` (with CUDA 10.1, torchvision 0.4.2)
- nvcc 10.1
    - `9.0.176`, `10.1` compiling and execution tested
    - `9.2.88` execution only
- gcc 5.4.0
- Pyenv or Anaconda

and Python dependencies list in `3d-tracking/requirements.txt` 

## Quick Start
In this section, you will train a model from scratch, test our pretrained models, and reproduce our evaluation results.
For more detailed instructions, please refer to [`DOCUMENTATION.md`](3d-tracking/DOCUMENTATION.md).

### Installation
- Clone this repo:
```bash
git clone -b pytorch1.0 --single-branch https://github.com/ucbdrive/3d-vehicle-tracking.git
cd 3d-vehicle-tracking/
```

- Install PyTorch 1.0.0+ and torchvision from http://pytorch.org and other dependencies. You can create a virtual environment by the following:
```bash
# Add path to bashrc 
echo -e '\nexport PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

# Install pyenv
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

# Restart a new terminal if "exec $SHELL" doesn't work
exec $SHELL

# Install and activate Python in pyenv
pyenv install 3.6.9
pyenv local 3.6.9
```

- Install requirements, create folders and compile binaries for detection
```bash
cd 3DTracking
bash scripts/init.sh
cd ..

cd faster-rcnn.pytorch
bash init.sh
```

> NOTE: For [faster-rcnn-pytorch](faster-rcnn-pytorch/lib/setup.py) compiling problems 
[[1](https://github.com/jwyang/faster-rcnn.pytorch/issues/410#issuecomment-450709668)], please compile COCO API and replace pycocotools.

> NOTE: For [object-ap-eval](https://github.com/traveller59/kitti-object-eval-python#dependencies) compiling problem. It only supports python 3.6+, need `numpy`, `skimage`, `numba`, `fire`. If you have Anaconda, just install `cudatoolkit` in anaconda. Otherwise, please reference to this [page](https://github.com/numba/numba#custom-python-environments) to set up llvm and cuda for numba.

### Data Preparation

For a quick start, we suggest using GTA `val` set as a starting point. You can get all needed data via the following script.

```bash
# We recommand using GTA `val` set (using `mini` flag) to get familiar with the data pipeline first, then using `all` flag to obtain all the data
python loader/download.py mini
```

More details can be found in [3d-tracking](3d-tracking/README.md).

### Execution

For running a whole pipeline (2D proposals, 3D estimation and tracking):
```bash
# Generate predicted bounding boxes for object proposals
cd faster-rcnn.pytorch/

# Step 00 (Optional) - Training on GTA dataset
./run_train.sh

# Step 01 - Generate bounding boxes
./run_test.sh
```

```bash
# Given object proposal bounding boxes and 3D center from faster-rcnn.pytorch directory
cd 3d-tracking/

# Step 00 - Data Preprocessing
# Collect features into json files (check variables in the code)
python loader/gen_pred.py gta val

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


## Citation
If you find our code/models useful in your research, please cite our paper:
```
@inproceedings{Hu3DT19,
author = {Hu, Hou-Ning and Cai, Qi-Zhi and Wang, Dequan and Lin, Ji and Sun, Min and Krähenbühl, Philipp and Darrell, Trevor and Yu, Fisher},
title = {Joint Monocular 3D Vehicle Detection and Tracking},
journal = {ICCV},
year = {2019}
}
```

## License
This work is licensed under BSD 3-Clause License. See [LICENSE](LICENSE) for details. 
Third-party datasets and tools are subject to their respective licenses.

## Acknowledgements
We thank [faster.rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) for the detection codebase, [pymot](https://github.com/Videmo/pymot) for their MOT evaluation tool and [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python) for the 3D AP calculation tool.
