import json
import os.path as osp
from glob import glob

import numpy as np
from nms_cpu import nms_cpu as nms


'''
load annotation from BDD format json files
'''
def load_annos_bdd(path, folder=True, json_name='*/*_final.json'):
    print("Loading GT file {} ...".format(path))
    if folder:
        jsonlist = sorted(glob(osp.join(path, json_name)))
    else:
        jsonlist = json.load(open(path, 'r'))

    assert len(jsonlist) > 0, "{} has no files".format(path)

    anno = []
    for idx, trackjson in enumerate(jsonlist):
        if folder:
            trackinfo = json.load(open(trackjson, 'r'))
        else:
            trackinfo = trackjson

        annotations = {}
        annotations.update({
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'orientation': []
        })

        for obj in trackinfo['labels']:
            if not obj['attributes']['ignore']:
                annotations['name'].append('Car')
                annotations['truncated'].append(obj['attributes']['truncated'])
                annotations['occluded'].append(obj['attributes']['occluded'])
                annotations['bbox'].append(
                            [obj['box2d']['x1'], obj['box2d']['y1'],
                             obj['box2d']['x2'], obj['box2d']['y2']])
                annotations['alpha'].append(obj['box3d']['alpha'])
                annotations['dimensions'].append(obj['box3d']['dimension'])
                annotations['location'].append(obj['box3d']['location'])
                annotations['orientation'].append(obj['box3d']['orientation'])

        annotations['name'] = np.array(annotations['name'])
        annotations['truncated'] = np.array(annotations['truncated'])
        annotations['occluded'] = np.array(annotations['occluded'])
        annotations['alpha'] = np.array(annotations['alpha']).astype(
            'float')
        annotations['bbox'] = np.array(annotations['bbox']).astype(
            'float').reshape(-1, 4)
        annotations['dimensions'] = np.array(
            annotations['dimensions']).reshape(-1, 3)[:, [2, 0, 1]]
        annotations['location'] = np.array(annotations['location']).reshape(
            -1, 3)
        annotations['orientation'] = np.array(
            annotations['orientation']).reshape(-1)
        anno.append(annotations)

    return anno

def load_preds_bdd(path, use_nms=True, folder=False, json_name='*bdd_3d.json'):
    print("Loading PD file {} ...".format(path))

    # A flag indicate using kitti (bottom center) or gta (3D box center) format
    use_kitti_location = 'kitti' in path

    if folder:
        jsonlists = sorted(glob(osp.join(path, json_name)))
        jsonlist = [itm for ji in jsonlists for itm in json.load(open(ji, 'r'))]
    else:
        jsonlist = json.load(open(path, 'r'))

    assert len(jsonlist) > 0, "{} has no files".format(path)

    anno = []
    for idx, trackinfo in enumerate(jsonlist):
        annotations = {}
        annotations.update({
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'score': [],
            'dimensions': [],
            'location': [],
            'orientation': []
        })

        for obj in trackinfo['prediction']:
            if not obj['attributes']['ignore']:
                annotations['name'].append('Car')
                annotations['truncated'].append(0)
                annotations['occluded'].append(0)
                annotations['bbox'].append(
                            [obj['box2d']['x1'], obj['box2d']['y1'],
                             obj['box2d']['x2'], obj['box2d']['y2']])
                annotations['score'].append(obj['box2d']['confidence'])
                annotations['alpha'].append(obj['box3d']['alpha'])
                annotations['dimensions'].append(obj['box3d']['dimension'])
                annotations['location'].append(obj['box3d']['location'])
                annotations['orientation'].append(obj['box3d']['orientation'])

        name = np.array(annotations['name'])
        truncated = np.array(annotations['truncated'])
        occluded = np.array(annotations['occluded'])
        box = np.array(annotations['bbox']).astype('float').reshape(-1, 4)
        score = np.array(annotations['score']).astype('float').reshape(-1)
        dim = np.array(annotations['dimensions']).reshape(-1, 3)[:, [2, 0, 1]]
        alpha = np.array(annotations['alpha']).astype('float')
        loc = np.array(annotations['location']).reshape(-1, 3)
        if use_kitti_location:
            # Bottom center of a 3D object, instead of 3D box center
            loc[:, 1] += dim[:, 2] / 2
        rot_y = np.array(annotations['orientation']).reshape(-1)

        if use_nms:
            # print("Using NMS to suppress number of bounding box")
            keep = nms(np.hstack([box, score.reshape(-1, 1)]), 0.3)
            name = name[keep]
            truncated = truncated[keep]
            occluded = occluded[keep]
            box = box[keep]
            score = score[keep]
            dim = dim[keep].reshape(-1, 3)
            alpha = alpha[keep]
            loc = loc[keep].reshape(-1, 3)
            rot_y = rot_y[keep].reshape(-1)

        annotations['name'] = name
        annotations['truncated'] = truncated
        annotations['occluded'] = occluded
        annotations['alpha'] = alpha
        annotations['bbox'] = box
        annotations['dimensions'] = dim
        annotations['location'] = loc
        annotations['orientation'] = rot_y
        annotations['score'] = score
        anno.append(annotations)

    return anno

