import _init_paths

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from pymot import MOTEvaluation


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
                        description='3D Tracking BDD Format Evaluation',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt_path', help='Path of input info for tracking',
                        default='output/616_030_gta_val_set', type=str)
    parser.add_argument('--pd_path', help='Path of tracking sequence output',
                        default='output/616_030_gta_val_set/' \
    'lstmoccdeep_age20_aff0.1_hit0_100m_803', type=str)
    parser.add_argument('--debug_log_file', help='Path of debug log')

    parser.add_argument('--verbose', help='Show more information',
                        default=False, action='store_true')

    args = parser.parse_args()
    return args


def load_label_path(json_path):
    assert os.path.isdir(json_path), "Empty path {}".format(json_path)

    # Load lists of json files
    paths = []
    for fn in sorted(os.listdir(json_path)):
        if fn.endswith('bdd_3d.json'):
            paths.append(os.path.join(json_path, fn))

    assert len(paths), "Not label files found in {}".format(json_path)

    raw_list = [json.load(open(_l, 'r')) for _l in paths]

    seqs = []
    for seq_idx, seq in enumerate(raw_list):
        frms = []
        for fr_idx, frm in enumerate(seq):
            trks = []
            for it_idx, itm in enumerate(frm['prediction']):
                if itm['id'] == -1: continue
                trks.append({'height': itm['box2d']['y2'] - itm['box2d']['y1'],
                            'width': itm['box2d']['x2'] - itm['box2d']['x1'],
                            'id': itm['id'],
                            'x': itm['box3d']['xc'],
                            'y': itm['box3d']['yc']})

            frms.append({'timestamp': fr_idx,
             'im_path': frm['name'],
             'class': 'frame',
             'hypotheses': trks})
        seqs.append(frms)


    seq_list = [{'frames': frs, 
                'class': 'video', 
                'filename': _l} for frs, _l in zip(seqs, paths) ]

    return seq_list


class TrackEvaluation:

    def __init__(self, gt_list, pd_list, pd_path, _debug=None, verbose=False):
        self.pd_path = pd_path
        self.debug_log_file = _debug
        self.verbose = verbose

        self.seq_gt_list = gt_list
        self.seq_pd_list = pd_list

    def eval_app(self):
        """
        Evaluation of tracking results using PYMOT
        """
        abs_stats = None
        relative_stats = None
        total_n_gt = 0
        result = {i:[] for i in range(len(self.seq_gt_list))}
        seq_mota = []
        seq_motp = []
        print('=> Begin evaluation...')

        for i_s, (seq_gt, seq_pd) in enumerate(tqdm(
                zip(self.seq_gt_list, self.seq_pd_list), disable=not self.verbose)):
            result[i_s] = self.eval_parallel(seq_gt, seq_pd)

        for res_key in result:
            this_abs_stats, this_relative_stats = result[res_key]
            if this_abs_stats == [] or this_relative_stats == []:
                print("Empty results")
                continue

            n_gt = this_abs_stats['ground truths']

            seq_mota.append(this_relative_stats['MOTA'])
            seq_motp.append(this_relative_stats['MOTP'])

            if abs_stats is None:
                abs_stats = this_abs_stats
            else:
                abs_stats = {k: abs_stats[k] + this_abs_stats[k] for k in
                             this_abs_stats.keys()}

            if relative_stats is None:
                relative_stats = {k: v * n_gt for k, v in
                                  this_relative_stats.items()}
            else:
                relative_stats = {
                    k: relative_stats[k] + this_relative_stats[k] * n_gt for k
                in
                    this_relative_stats.keys()}
            total_n_gt += n_gt

        if relative_stats is None or abs_stats is None:
            return

        for k, v in relative_stats.items():
            relative_stats[k] = v * 1. / total_n_gt

        # Results
        print("Results")
        if self.verbose:
            for k, v in relative_stats.items():
                relative_stats[k] = np.round(v, 4)
            print(relative_stats)
            print(abs_stats)

            # For check
            print('n_seq: {}'.format(len(seq_mota)))
            print(seq_mota)
            print(np.argsort(seq_mota))
            print(seq_motp)
            print(np.argsort(seq_motp))


        print("MOTA, MOTP, TP, TR, MM, NM, RM, FP, FN - {}".format(
            self.pd_path))
        print(
            "{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, "
            "{:.3f}".format(
                relative_stats["MOTA"] * 100,
                relative_stats['MOTP'] * 100,
                relative_stats['track precision'] * 100,
                relative_stats['track recall'] * 100,
                relative_stats['mismatch rate'] * 100,
                relative_stats['non-recoverable mismatch rate'] * 100,
                relative_stats['recoverable mismatch rate'] * 100,
                relative_stats['false positive rate'] * 100,
                relative_stats['miss rate'] * 100)
        )


    def eval_parallel(self, seq_gt, seq_pd):
        if seq_gt['frames'] is None or seq_pd['frames'] is None or \
            np.all([item['annotations'] == [] for item in seq_gt['frames']]) or \
            np.all([item['hypotheses'] == [] for item in seq_pd['frames']]):
            return [], []

        evaluator = MOTEvaluation(seq_gt, seq_pd, 0.5)
        evaluator.evaluate()

        abs_stats = evaluator.getAbsoluteStatistics()
        relative_stats = evaluator.getRelativeStatistics()

        if self.debug_log_file:
            with open(self.debug_log_file, 'w') as fp:
                json.dump(evaluator.getVisualDebug(), fp, indent=4)


        return abs_stats, relative_stats


def main():
    if not os.path.exists('output'):
        os.makedirs('output')

    args = parse_args()
    seq_gt_name = os.path.join(args.gt_path, 'gt.json')
    seq_pd_name = os.path.join(args.pd_path, 'data')

    seq_gt_list = json.load(open(seq_gt_name, 'r'))
    seq_pd_list = load_label_path(seq_pd_name)

    te = TrackEvaluation(
            seq_gt_list, 
            seq_pd_list,
            args.pd_path, 
            _debug = args.debug_log_file, 
            verbose = args.verbose)
    te.eval_app()


if __name__ == '__main__':
    main()
