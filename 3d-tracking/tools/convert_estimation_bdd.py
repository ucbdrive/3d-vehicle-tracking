import os
import sys
import pickle
import argparse
import numpy as np

import utils.bdd_helper as bh
import utils.tracking_utils as tu
from utils.config import cfg


def parse_args():
    parser = argparse.ArgumentParser(
                        description='3D Estimation BDD Format Convertor',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('set', choices=['gta', 'kitti'])
    parser.add_argument('split', choices=['train', 'val', 'test'], 
                        help='Which data split to use in testing')
    parser.add_argument('--session', default='616',
                        help='Name of the session, to separate exp')
    parser.add_argument('--epoch', default='030',
                        help='How many epochs you used to separate exp')
    parser.add_argument('--root', default='output/',
                        help='Where to put the files')
    parser.add_argument('--flag', default='-j 1',
                        help='Flags for running evaluation code')
    parser.add_argument('--lstm_name', default='803',
                        help='lstm ckpt name')
    parser.add_argument('--max_age', help='Maximum lifespan of a track',
                        default=20, type=int)
    parser.add_argument('--min_hits', help='Minimum hits to set a track',
                        default=0, type=int)
    parser.add_argument('--affinity_thres', help='Affinity threshold',
                        default=0.1, type=float)
    parser.add_argument('--max_depth', help='tracking within max_depth meters',
                        default=100, type=int)
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Show command without running')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite the output files')
    parser.add_argument('--load_refine', action='store_true', default=False,
                        help='Load refined files')
    args = parser.parse_args()

    return args


def load_single_frame_result(pkl_path):

    print("Loading 3d estimation results from {}...".format(pkl_path))
    trk_load = pickle.load(open(pkl_path, 'rb'))
    trks = [n for seq in trk_load for n in seq]

    print("Single frame result loaded")

    return trks


def load_label_path(json_path):
    assert os.path.isdir(json_path), "Empty path".format(json_path)

    # Load lists of json files
    folders = [n for n in sorted(os.listdir(json_path)) 
                if os.path.isdir(os.path.join(json_path, n))]

    paths = {n: [] for n in folders}
    for n in folders:
        for fn in sorted(os.listdir(os.path.join(json_path, n))):
            if fn.endswith('.json'):
                paths[n].append(os.path.join(json_path, n, fn))

    assert len(paths), "Not label files found in {}".format(json_path)

    return folders, paths


def convert_app(det_placeholder, det_out):


    if len(det_out['rois_pd']):
        depth_pd = det_out['depth_pd'].copy()
        rot_y = tu.alpha2rot_y(det_out['alpha_pd'],
                           det_out['center_pd'][:, 0] - cfg.GTA.W // 2,
                           cfg.GTA.FOCAL_LENGTH)
        # In camera coordinates
        location = tu.imagetocamera(det_out['center_pd'],
                            depth_pd,
                            det_out['cam_calib'])

    for idx, prediction in enumerate(det_placeholder['prediction']):

        if len(det_out['rois_pd']):
            pred_box2d = bh.get_box2d_array([prediction])
            pred_cen = bh.get_cen_array([prediction])
            roi_match = np.sum(det_out['rois_pd'] == pred_box2d, axis=1) == 5
            cen_match = np.sum(det_out['center_pd'] == pred_cen, axis=1) == 2

            match = np.where(roi_match * cen_match)[0]

            if len(match) == 0:
                continue

            box = det_out['rois_pd'][match].reshape(5)
            cen = det_out['center_pd'][match].reshape(2)
            alpha = det_out['alpha_pd'][match].reshape(1)
            dim = det_out['dim_pd'][match].reshape(3)
            orient = rot_y[match].reshape(1)
            loc = location[match].reshape(3)

            prediction['box2d']['x1'] = int(box[0])
            prediction['box2d']['y1'] = int(box[1])
            prediction['box2d']['x2'] = int(box[2])
            prediction['box2d']['y2'] = int(box[3])
            prediction['box2d']['confidence'] = float(box[4])

            # 3D box projected center
            prediction['box3d']['xc'] = int(cen[0])
            prediction['box3d']['yc'] = int(cen[1])
            prediction['box3d']['alpha'] = float(alpha)
            prediction['box3d']['dimension'] = dim.tolist()
            prediction['box3d']['location'] = loc.tolist()
            prediction['box3d']['orientation'] = orient.tolist()

    return det_placeholder


if __name__ == '__main__':

    print(' '.join(sys.argv))
    args = parse_args()

    if args.set == 'gta':
        # GTA training / testing set with tracking / detection
        DATASET = cfg.GTA.TRACKING
    else:
        # KITTI tracking dataset with tracking / detection
        DATASET = cfg.KITTI.TRACKING
        assert args.split in ['train', 'test'], "KITTI has no validation set"

    DATASET.PRED_PATH = DATASET.PRED_PATH.replace('train', args.split)

    # Load data
    print("Load label file from path: {}".format(DATASET.PRED_PATH))
    folders, data_path = load_label_path(DATASET.PRED_PATH)

    for seq_idx, folder in enumerate(folders):

        pkl_path = '{ROOT}{SESS}_{EP}_{DT}_{PH}_set/{SESS}_{EP}_{' \
                   'SQ}_bdd_roipool_output.pkl'.format(
            **{'ROOT': args.root, 'SESS': args.session, 'EP': args.epoch, 
                'DT': args.set, 'PH': args.split, 'SQ': folder})

        save_path = '{ROOT}{SESS}_{EP}_{DT}_{PH}_set/{SQ}_bdd_3d.json'.format(
            **{'ROOT': args.root, 'SESS': args.session, 'EP': args.epoch, 
                'DT': args.set, 'PH': args.split, 'SQ': folder})

        det_pred = load_single_frame_result(pkl_path)

        hypos = []
        for fr_idx, frame_path in enumerate(data_path[folder]):

            det_placeholder = bh.load_json(frame_path)

            hypo = convert_app(det_placeholder, det_pred[fr_idx])
            hypos.append(hypo)

        print("Saving updated tracking results with {} frames at {}...".format(
            len(hypos), save_path))
        bh.dump_json(save_path, hypos)

