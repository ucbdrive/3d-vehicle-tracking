import json
import operator
from collections import Iterable
from functools import reduce  # forward compatibility for Python 3

import numpy as np


def load_json(filename):
    return json.load(open(filename, 'r'))


def dump_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)


def read_labels(filename):
    labels = load_json(filename)
    if not isinstance(labels, Iterable):
        labels = [labels]
    return labels


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


def get_cen_array(boxes):
    if len(boxes) == 0:
        return np.empty([0, 2])
    return np.array([[box['box3d']['xc'],
                      box['box3d']['yc']] for box in boxes],
                    dtype=np.float32)


def get_time_of_day(hour):
    # daytime|night|dawn|dusk
    # 24 hours format
    if hour < 6 or hour > 19:
        return 'night'
    elif hour < 7 or hour > 18:
        return 'dawn/dusk'
    else:
        return 'daytime'


def init_frame_format():
    data = dict()
    data['name'] = ''
    data['url'] = ''
    data['videoName'] = ''
    # image resolution, H, W
    data['resolution'] = {'width': 0,
                          'height': 0}
    data['attributes'] = {'weather': 'undefined',
                          'scene': 'undefined',
                          'timeofday': 'undefined'}
    data['intrinsics'] = {'focal': [0, 0],
                          'center': [0, 0],
                          # field of view
                          'fov': 0,
                          # near clip of viewing volume
                          'nearClip': 0}
    data['extrinsics'] = {'location': [0.0, 0.0, 0.0],
                          'rotation': [0.0, 0.0, 0.0]}
    data['timestamp'] = -1
    data['frameIndex'] = -1
    data['labels'] = []
    # just for results, not offcial attribute
    data['prediction'] = []  
    return data


def init_labels_format():
    data = dict()
    # Tracking ID
    data['id'] = -1
    # Class of object. e.g, Car, Van, Pedastrian
    data['category'] = ''
    data['manualShape'] = False
    data['manualAttributes'] = False
    data['attributes'] = {'occluded': False,
                          'truncated': False,
                          # Ignore if it's far away and too small
                          'ignore': False,
                          'trafficLightColor': 'none',
                          'areaType': 'undefined',
                          'laneDirection': 'undefined',
                          'laneStyle': 'undefined',
                          'laneTypes': 'undefined'}
    data['box2d'] = {'x1': 0,
                     'y1': 0,
                     'x2': 0,
                     'y2': 0,
                     # Optional, saved for detection
                     'confidence': 0.0}
    data['box3d'] = {'alpha': 0.0,
                     # The absolute orientation angle of an object.
                     # Not necessary if we have perfect alpha and box2d center.
                     'orientation': 0.0,
                     # The object location in *camera* coordinate
                     'location': [0, 0, 0],
                     'dimension': [0, 0, 0]}
    data['poly2d'] = {'vertices': [],
                      'types': 'L',
                      'closed': True}
    return data
