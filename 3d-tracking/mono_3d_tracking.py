import _init_paths

import argparse
import json
import os
import pickle
import time

from joblib import Parallel, delayed
from tqdm import tqdm

import utils.tracking_utils as tu
from model.tracker_2d import Tracker2D
from model.tracker_3d import Tracker3D
from tools.eval_mot_bdd import TrackEvaluation

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Monocular 3D Tracking',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', help='Which dataset for tracking',
                        choices=['gta', 'kitti'], type=str)
    parser.add_argument('-j', dest='n_jobs', help='How many jobs in parallel'
                        ' (will brake the tracking ID) if more than 1',
                        default=10, type=int)
    parser.add_argument('--path', help='Path of input info for tracking',
                        default='./output/623_100_kitti_train_set/*bdd_roipool_output.pkl', type=str)
    parser.add_argument('--out_path', help='Path of tracking sequence output',
                        default='./623_100_kitti_train_set/', type=str)
    parser.add_argument('--debug_log_file', help='Path of debug log')
    parser.add_argument('--gpu', help='Using which GPU(s)',
                        default=None)
    parser.add_argument('--verbose', help='Show more information',
                        default=False, action='store_true')
    parser.add_argument('--visualize', help='Show current prediction',
                        default=False, action='store_true')

    parser.add_argument('--max_age', help='Maximum lifespan of a track',
                        default=10, type=int)
    parser.add_argument('--min_hits', help='Minimum hits to set a track',
                        default=3, type=int)
    parser.add_argument('--affinity_thres', help='Affinity threshold',
                        default=0.3, type=float)
    parser.add_argument('--skip', help='Skip frame by n',
                        default=1, type=int)
    parser.add_argument('--min_seq_len',
                        help='skip a sequence if less than n frames',
                        default=10, type=int)
    parser.add_argument('--max_depth', help='tracking within max_depth meters',
                        default=150, type=int)

    parser.add_argument('--occ', dest='use_occ',
                        help='use occlusion and depth ordering to help '
                             'tracking',
                        default=False, action='store_true')
    parser.add_argument('--deep', dest='deep_sort',
                        help='feature similarity to associate',
                        default=False, action='store_true')

    method = parser.add_mutually_exclusive_group(required=False)
    method.add_argument('--kf2d', dest='kf2d',
                        help='2D Kalman filter to smooth',
                        default=False, action='store_true')
    method.add_argument('--kf3d', dest='kf3d',
                        help='3D Kalman filter to smooth',
                        default=False, action='store_true')
    method.add_argument('--lstm', dest='lstm3d',
                        help='Estimate motion using LSTM to help 3D prediction',
                        default=False, action='store_true')
    method.add_argument('--lstmkf', dest='lstmkf3d',
                        help='Estimate motion in LSTM Kalman Filter to help '
                             '3D prediction',
                        default=False, action='store_true')

    args = parser.parse_args()
    args.device = 'cpu' if args.gpu is None else 'cuda'
    return args


class Mono3DTracker:

    def __init__(self, args):
        self.seq_hypo_list = []
        self.seq_gt_list = []
        self.args = args
        if args.device == 'cuda':
            assert args.gpu != '', 'No gpu specific'
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        if os.path.isdir(args.path):
            print('Load lists of pkl files')
            input_names = sorted(
                [n for n in os.listdir(args.path)
                 if n.endswith('bdd_roipool_output.pkl')])
            self.label_paths = [os.path.join(args.path, n) for n in input_names]
        elif args.path.endswith('bdd_roipool_output.pkl'):
            print('Load single pkl file')
            self.label_paths = [args.path]
        elif args.path.endswith('gta_roipool_output.pkl'):
            print('Load bundled pkl file')
            self.label_paths = args.path
        else:
            self.label_paths = []

    def run_app(self):
        """
        Entry function of calling parallel tracker on sequences
        """
        self.seq_gt_name = os.path.join(os.path.dirname(self.args.path),
                                         'gt.json')
        self.seq_pd_name = self.args.out_path + '_pd.json'

        if isinstance(self.label_paths, str):
            label_paths = pickle.load(open(self.label_paths, 'rb'))
        else:
            label_paths = self.label_paths

        n_seq = len(label_paths)
        print('* Number of sequence: {}'.format(n_seq))
        assert n_seq > 0, "Number of sequence is 0!"

        print('=> Building gt & hypo...')
        result = Parallel(n_jobs=self.args.n_jobs)(
            delayed(self.run_parallel)(seq_path, i_s)
            for i_s, seq_path in enumerate(tqdm(
                label_paths,
                disable=not self.args.verbose))
        )

        self.seq_gt_list = [n[0] for n in result]
        self.seq_hypo_list = [n[1] for n in result]

        if not os.path.isfile(self.seq_gt_name):
            with open(self.seq_gt_name, 'w') as f:
                print("Writing to {}".format(self.seq_gt_name))
                json.dump(self.seq_gt_list, f)
        with open(self.seq_pd_name, 'w') as f:
            print("Writing to {}".format(self.seq_pd_name))
            json.dump(self.seq_hypo_list, f)

    def run_parallel(self, seq_path, i_s):
        """
        Major function inside parallel calling run_seq with tracker and seq
        """

        if isinstance(seq_path, str):
            with open(seq_path, 'rb') as f:
                seqs = pickle.load(f)
            seq = seqs[0]

            if self.args.verbose: print(
                "Seq {} has {} frames".format(seq_path, len(seq)))
        else:
            # Bundled file
            seq = seq_path
            if self.args.verbose: print(
                "Seq {} has {} frames".format(i_s, len(seq)))

        # NOTE: Not in our case but will exclude small sequences from computing
        if len(seq) < self.args.min_seq_len:
            print("Warning: Skip sequence due to short length {}".format(
                len(seq)))
            seq_gt = {'frames': None, 'class': 'video', 'filename': 'null'}
            seq_hypo = {'frames': None, 'class': 'video', 'filename': 'null'}
            return seq_gt, seq_hypo

        # create instance of the SORT tracker
        if self.args.kf3d \
                or self.args.lstm3d \
                or self.args.lstmkf3d:
            mot_tracker = Tracker3D(
                dataset=self.args.dataset,
                max_depth=self.args.max_depth,
                max_age=self.args.max_age,
                min_hits=self.args.min_hits,
                affinity_threshold=self.args.affinity_thres,
                deep_sort=self.args.deep_sort,
                use_occ=self.args.use_occ,
                kf3d=self.args.kf3d,
                lstm3d=self.args.lstm3d,
                lstmkf3d=self.args.lstmkf3d,
                device=self.args.device,
                verbose=self.args.verbose,
                visualize=self.args.visualize)
        else:
            mot_tracker = Tracker2D(
                dataset=self.args.dataset,
                max_depth=self.args.max_depth,
                max_age=self.args.max_age,
                min_hits=self.args.min_hits,
                affinity_threshold=self.args.affinity_thres,
                kf2d=self.args.kf2d,
                deep_sort=self.args.deep_sort,
                verbose=self.args.verbose,
                visualize=self.args.visualize)

        if self.args.verbose: print("Processing seq{}".format(i_s))
        frames_anno, frames_hypo = self.run_seq(mot_tracker, seq)

        seq_gt = {'frames': frames_anno, 'class': 'video', 'filename': seq_path}
        seq_hypo = {'frames': frames_hypo, 'class': 'video', 'filename': seq_path}

        return seq_gt, seq_hypo

    def run_seq(self, mot_tracker, seq):
        """
        Core function of tracking along a sequence using mot_tracker
        """
        frame = 0
        batch_time = tu.AverageMeter()
        frames_hypo = []
        frames_anno = []
        for i_f, data in tqdm(enumerate(seq), disable=not self.args.verbose):
            if not i_f % self.args.skip == 0:
                continue
            frame += 1  # detection and frame numbers begin at 1

            end = time.time()
            trackers = mot_tracker.update(data)
            batch_time.update(time.time() - end)


            # save gt frame annotations
            gt_anno = mot_tracker.frame_annotation
            frame_gt = {'timestamp': i_f,
                        'num': i_f,
                        'im_path': data['im_path'],
                        'class': 'frame',
                        'annotations': gt_anno}
            frames_anno.append(frame_gt)

            # save detect results
            frame_hypo = {'timestamp': i_f,
                          'num': i_f,
                          'im_path': data['im_path'],
                          'class': 'frame',
                          'hypotheses': trackers}
            frames_hypo.append(frame_hypo)
            
        if self.args.verbose:
            print(
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(
            batch_time=batch_time))

        return frames_anno, frames_hypo


def main():
    if not os.path.exists('output'):
        os.makedirs('output')

    args = parse_args()
    assert not args.use_occ or (args.lstmkf3d or args.lstm3d or args.kf3d), \
        "Occlusion only be used in 3D method"

    tracker = Mono3DTracker(args)
    tracker.run_app()

    te = TrackEvaluation(
            tracker.seq_gt_list, 
            tracker.seq_hypo_list,
            args.out_path, 
            _debug = args.debug_log_file, 
            verbose = args.verbose)
    te.eval_app()

if __name__ == '__main__':
    main()
