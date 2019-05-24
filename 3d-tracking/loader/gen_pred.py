import argparse
import numpy as np
import os
import pickle
import re

import utils.bdd_helper as bh
from utils.config import cfg


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
                        description='3DT BDD format dataset generation',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('set', choices=['gta', 'kitti'],
                        help='Generate GTA or KITTI dataset')
    parser.add_argument('split', choices=['train', 'val', 'test'], 
                        help='Generate GTA train, validation or test dataset')
    parser.add_argument('--mode', choices=['train', 'test'], default='test',
                        help='Test mode dont filter anything, train mode will')
    parser.add_argument('--kitti_task', default='track',
                        choices=['detect', 'track'],
                        help='KITTI task [detect, track]')
    parser.add_argument('--max_depth', dest='max_depth',
                        help='filter if depth exceed this threshold',
                        default=150, type=int)
    parser.add_argument('--min_pixel', dest='min_pixel',
                        help='filter if #pixels lower than this threshold',
                        default=256, type=int)
    parser.add_argument('--verbose_interval', dest='verbose_interval',
                        help='show info every N frames',
                        default=10, type=int)
    args = parser.parse_args()
    return args


# Global var
args = parse_args()

if args.set == 'gta':
    # GTA training / testing set with tracking / detection
    DATASET = cfg.GTA.TRACKING

    det_name = os.path.join(DATASET.PATH,
                            'gta_{}_detections.pkl'.format(args.split))
    save_name = os.path.join(DATASET.PATH,
                            'gta_{}_list.json'.format(args.split))
    DATASET.LABEL_PATH = DATASET.LABEL_PATH.replace('train', args.split)
    DATASET.PRED_PATH = DATASET.PRED_PATH.replace('train', args.split)
    DATASET.IM_PATH = DATASET.IM_PATH.replace('train', args.split)
else:
    # KITTI tracking / object dataset
    if args.kitti_task == 'detect':
        DATASET = cfg.KITTI.OBJECT
        det_name = os.path.join(DATASET.PATH,
                            'kitti_{}_obj_detections.pkl'.format(
                                args.split))
        save_name = os.path.join(DATASET.PATH,
                            'kitti_{}_obj_list.json'.format(args.split))
    elif args.kitti_task == 'track':
        DATASET = cfg.KITTI.TRACKING
        det_name = os.path.join(DATASET.PATH,
                            'kitti_{}_trk_detections.pkl'.format(
                                args.split))
        save_name = os.path.join(DATASET.PATH,
                             'kitti_{}_trk_list.json'.format(args.split))

    DATASET.LABEL_PATH = DATASET.LABEL_PATH.replace('train', args.split)
    DATASET.PRED_PATH = DATASET.PRED_PATH.replace('train', args.split)
    DATASET.IM_PATH = DATASET.IM_PATH.replace('train', args.split)
    assert args.split in ['train', 'test'], "KITTI has no validation set"


def load_label_path(json_path, pattern='final.json'):
    assert os.path.isdir(json_path), "Empty path".format(json_path)

    # Load lists of json files
    folders = [os.path.join(json_path, n) for n in
                    sorted(os.listdir(json_path)) if
                    os.path.isdir(os.path.join(json_path, n))]
    paths = [os.path.join(n, fn) 
                for n in folders 
                    for fn in sorted(os.listdir(n))
                        if fn.endswith(pattern)]

    assert len(paths), "Not label files found in {}".format(json_path)

    return paths


class Dataset():

    def __init__(self):

        self.set = args.set
        self.kitti_task = args.kitti_task

        # Load data
        print("Load label file from path: {}".format(DATASET.LABEL_PATH))
        pattern = 'final.json' if args.set == 'gta' else '.json'
        self.data_path = load_label_path(DATASET.LABEL_PATH, pattern)

        # Load detection
        if os.path.isfile(det_name):
            with open(det_name, 'rb') as f:
                det_result = pickle.load(f)
            self.det_result = np.array(det_result[1])

        # Assertion
        assert len(self.data_path) == self.det_result.shape[0], \
            'ERROR: length of GT frames {} not equals detected ones {}'.format(
                len(self.data_path), self.det_result.shape[0])

        print("Overall Data length {}"
              " Valid data length {}".format(len(self.data_path),
                                                self.det_result.shape[0]))

    def build_pred(self):
        seq_data = []
        if self.data_path == []:
            print("Empty data")
            return seq_data

        for idx, file_path in enumerate(self.data_path):

            frame_data = bh.load_json(file_path)

            if idx % args.verbose_interval == 0 and idx != 0:
                print('{} images.'.format(idx))

            match_box = self.det_result[idx]

            prediction_data = []
            for i, box in enumerate(match_box):
                prediction = bh.init_labels_format()
                prediction['box2d']['x1'] = int(box[0])
                prediction['box2d']['y1'] = int(box[1])
                prediction['box2d']['x2'] = int(box[2])
                prediction['box2d']['y2'] = int(box[3])
                prediction['box2d']['confidence'] = float(box[6])
                # 3D box projected center
                prediction['box3d']['xc'] = int(box[4])
                prediction['box3d']['yc'] = int(box[5])
                del prediction['poly2d']
                del prediction['attributes']['trafficLightColor']
                del prediction['attributes']['areaType']
                del prediction['attributes']['laneDirection']
                del prediction['attributes']['laneStyle']
                del prediction['attributes']['laneTypes']
                prediction_data.append(prediction)

            frame_data['prediction'] = prediction_data

            print("Frame {}, GT: {} Boxes, PD: {} Boxes".format(
                idx,
                len(frame_data['labels']),
                len(frame_data['prediction'])))
            
            del frame_data['labels']

            filename = os.path.join(
                            DATASET.PRED_PATH, 
                            frame_data['videoName'],
                            os.path.basename(file_path))
            if not os.path.exists(os.path.dirname(filename)):
                os.mkdir(os.path.dirname(filename))

            bh.dump_json(filename, frame_data)
            seq_data.append(filename)

        print("Saving {} frames...".format(idx))

        return seq_data


    def det_valid_idx(self, det_box):
        is_valid = []
        for j, det in enumerate(det_box):
            if (det[2] > det[0] + 2 and det[3] > det[1] + 2) \
                    or (self.mode == 'test' and self.kitti_task == 'track'):
                is_valid.append(j)
        return np.array(is_valid)


def main():
    ds = Dataset()
    ds.build_pred()


if __name__ == '__main__':
    main()
