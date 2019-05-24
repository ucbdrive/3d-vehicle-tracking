
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

import utils.network_utils as nu
import utils.tracking_utils as tu
import utils.bdd_helper as bh 
from utils.config import cfg

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
                        description='3D Estimation BDD Format Evaluation',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('set', choices=['gta', 'kitti'],
                        help='Use GTA or KITTI dataset')
    parser.add_argument('split', choices=['train', 'val', 'test'], 
                        help='Train, validation or test dataset')
    parser.add_argument('--session', default='616',
                        help='Name of the session, to separate exp')
    parser.add_argument('--epoch', default='030',
                        help='How many epochs you used to separate exp')
    parser.add_argument('--min_depth', dest='min_depth', default=0.0,
                        type=float)
    parser.add_argument('--max_depth', dest='max_depth', default=150.0,
                        type=float)
    parser.add_argument('-j', dest='n_jobs', help='How many jobs in parallel',
                        default=10, type=int)

    parser.add_argument('--verbose', help='Show more information',
                        default=False, action='store_true')

    args = parser.parse_args()
    return args


def load_label_path(json_path, pattern='bdd.json'):
    assert os.path.isdir(json_path), "Empty path {}".format(json_path)

    # Load lists of json files
    paths = []
    for fn in sorted(os.listdir(json_path)):
        if fn.endswith(pattern):
            paths.append(os.path.join(json_path, fn))

    assert len(paths), "Not label files found in {}".format(json_path)

    return paths


class ThreeDimEvaluation:

    def __init__(self, gt_path, pd_path, max_depth=100, min_depth=0, verbose=False):

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.verbose = verbose

        self.seq_gt_path_list = load_label_path(gt_path, pattern='bdd.json')
        self.seq_pd_path_list = load_label_path(pd_path, pattern='bdd_3d.json')

        self.name_line = "{:>10}, {:>10}, {:>10}, {:>10}, " \
                        "{:>10}, {:>10}, {:>10}, " \
                        "{:>10}, {:>10}, {:>10}".format(
                        'abs_rel', 'sq_rel', 'rms', 'log_rms', 
                        'a1', 'a2', 'a3', 'AOS', 'DIM', 'CEN')

    def eval_app(self):
        """
        Evaluation of 3d estimation results
        """

        result = {i:[] for i in range(len(self.seq_gt_path_list))}

        print('=> Begin evaluation...')
        for i_s, (seq_gt_path, seq_pd_path) in enumerate(tqdm(
                zip(self.seq_gt_path_list, self.seq_pd_path_list), 
                disable=not self.verbose)):
            result[i_s] = self.eval_parallel(seq_gt_path, seq_pd_path)

            print("Validation Result of Sequence: {}".format(self.seq_pd_path_list[i_s]))
            if len(result[i_s]['dm']) == 0:
                print("Empty result")
                continue

            depth_metrics = np.mean(result[i_s]['dm'], axis=0)
            data_line = "{:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, " \
                        "{:10.3f}, {:10.3f}, {:10.3f}, " \
                        "{:10.3f}".format(
                depth_metrics[0].mean(), depth_metrics[1].mean(), 
                depth_metrics[2].mean(), depth_metrics[3].mean(), 
                depth_metrics[5].mean(), depth_metrics[6].mean(), 
                depth_metrics[7].mean(), 
                result[i_s]['aos'].avg, 
                result[i_s]['dim'].avg, 
                result[i_s]['cen'].avg)
            print(self.name_line)
            print(data_line)


    def eval_parallel(self, seq_gt_path, seq_pd_path):
        aos_meter = tu.AverageMeter()
        dim_meter = tu.AverageMeter()
        cen_meter = tu.AverageMeter()
        dm = []

        seq_gt = [json.load(open(_l, 'r')) for _l in json.load(open(seq_gt_path, 'r'))]
        seq_pd = json.load(open(seq_pd_path, 'r')) 

        for i, (frame_gt, frame_pd) in enumerate(zip(seq_gt, seq_pd)):
            labels = frame_gt['labels']
            cam_calib = np.array(frame_gt['intrinsics']['cali'])
            predictions = frame_pd['prediction']
            if len(predictions) == 0 or len(labels) == 0:
                continue

            box_gt = bh.get_box2d_array(labels)
            box_pd = bh.get_box2d_array(predictions)

            # Dim: H, W, L
            dim_pd = bh.get_label_array(predictions, 
                        ['box3d', 'dimension'], (0, 3)).astype(float)
            dim_gt = bh.get_label_array(labels, 
                        ['box3d', 'dimension'], (0, 3)).astype(float)
            # Alpha: -pi ~ pi
            alpha_pd = bh.get_label_array(predictions, 
                        ['box3d', 'alpha'], (0)).astype(float)
            alpha_gt = bh.get_label_array(labels, 
                        ['box3d', 'alpha'], (0)).astype(float)
            # Location in cam coord: x-right, y-down, z-front
            loc_pd = bh.get_label_array(predictions, 
                        ['box3d', 'location'], (0, 3)).astype(float)
            loc_gt = bh.get_label_array(labels, 
                        ['box3d', 'location'], (0, 3)).astype(float)
            # Depth
            depth_pd = np.maximum(0, loc_pd[:, 2])
            depth_gt = np.maximum(0, loc_gt[:, 2])
            center_pd = bh.get_cen_array(predictions)
            center_gt = tu.cameratoimage(loc_gt, cam_calib)

            if len(box_gt) > 0:
                iou, idx, valid = tu.get_iou(box_gt, box_pd[:, :4], 0.8)
            else:
                valid = np.array([False])

            if valid.any():
                # TODO: unmatched prediction and ground truth
                box_pd_v = box_pd[idx]
                alpha_pd_v = alpha_pd[idx]
                dim_pd_v = dim_pd[idx]
                depth_pd_v = depth_pd[idx]
                center_pd_v = center_pd[idx]

                aos_meter.update(np.mean(nu.compute_os(alpha_gt, alpha_pd_v)),
                                 alpha_gt.shape[0])
                dim_meter.update(np.mean(nu.compute_dim(dim_gt, dim_pd_v)),
                                 dim_gt.shape[0])
                w = (box_pd_v[:, 2:3] - box_pd_v[:, 0:1] + 1)
                h = (box_pd_v[:, 3:4] - box_pd_v[:, 1:2] + 1)
                cen_meter.update(
                    np.mean(nu.compute_cen(center_gt, center_pd_v, w, h)),
                    center_gt.shape[0])

                # Avoid zero in calculating a1, a2, a3
                mask = np.logical_and(depth_gt > self.min_depth,
                                      depth_gt < self.max_depth)
                mask = np.logical_and(mask, depth_pd_v > 0)
                if mask.any():
                    dm.append(
                        nu.compute_depth_errors(depth_gt[mask], depth_pd_v[mask]))
                else:
                    print("Not a valid depth range in GT")

        result = {'aos': aos_meter, 'dim': dim_meter, 'cen': cen_meter, 'dm': dm}
        return result


def main():
    if not os.path.exists('output'):
        os.makedirs('output')

    args = parse_args()
    if args.set == 'gta':
        gt_path = cfg.GTA.TRACKING.LABEL_PATH.replace('train', args.split)
    else:
        gt_path = cfg.KITTI.TRACKING.LABEL_PATH.replace('train', args.split)


    pd_path = 'output/{}_{}_{}_{}_set/'.format(args.session,
                                            args.epoch,
                                            args.set,
                                            args.split)

    te = ThreeDimEvaluation(gt_path, 
                            pd_path, 
                            args.max_depth, 
                            args.min_depth,
                            args.verbose)
    te.eval_app()


if __name__ == '__main__':
    main()
