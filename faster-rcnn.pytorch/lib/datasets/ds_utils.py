# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce  # forward compatibility for Python 3
import operator

import numpy as np


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (x2 >= x1).all()
    assert (y2 >= y1).all()
    assert (x2 < width).all()
    assert (y2 < height).all()


def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep


def get_from_dict(data_dict, map_list):
    assert isinstance(map_list, list)
    return reduce(operator.getitem, map_list, data_dict)


def get_label_array(boxes, key_list, empty_shape):
    if len(boxes) == 0:
        return np.empty(empty_shape)
    return np.array([get_from_dict(box, key_list) for box in boxes])


def get_box2d_array(boxes):
    if len(boxes) == 0:
        return np.empty([0, 5])
    return np.array([[box['box2d']['x1'],
                      box['box2d']['y1'],
                      box['box2d']['x2'],
                      box['box2d']['y2'],
                      box['box2d']['confidence']] for box in boxes],
                    dtype=np.float32)

