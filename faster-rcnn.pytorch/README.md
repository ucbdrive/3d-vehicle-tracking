# Joint Monocular 3D Detection and Tracking

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
In this section, you will train a model from scratch, test our pretrained models, and reproduce our evaluation results.

### Execution

For running a whole pipeline (training and testing):
```bash
# Generate predicted bounding boxes for object proposals

# Step 00 (Optional) - Training on GTA dataset
./run_train.sh

# Step 01 - Generate bounding boxes
./run_test.sh

# For 3D AP evaluation, please follow the step list in [3d-tracking](../3d-tracking/object-ap-eval)
# The bbox results of 3D AP are the 2D detection evaluation.
```

## License
Third-party datasets are subject to their respective licenses.

If you use our code/models in your research, please cite our paper:
```
@inproceedings{Hu3DT19,
author = {Hu, Hou-Ning and Cai, Qi-Zhi and Wang, Dequan and Lin, Ji and Sun, Min and Krähenbühl, Philipp and Darrell, Trevor and Yu, Fisher},
title = {Joint Monocular 3D Detection and Tracking},
journal = {ICCV},
year = {2019}
}
```

## Acknowledgements
We thank [faster.rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) for the detection codebase and [kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python) for the 3D AP calculation tool.
