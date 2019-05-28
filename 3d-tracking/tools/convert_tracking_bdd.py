import os
import sys
import argparse
import numpy as np

import utils.bdd_helper as bh
import utils.tracking_utils as tu


def parse_args():
    parser = argparse.ArgumentParser(
                        description='3D Tracking BDD Format Convertor',
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


def dump_output_json(filename, frame_data):
    if not os.path.exists(os.path.dirname(filename)):
        print("Create {} ...".format(os.path.dirname(filename)))
        os.makedirs(os.path.dirname(filename))

    bh.dump_json(filename, frame_data)


def load_label_path(json_path):
    assert os.path.isdir(json_path), "Empty path".format(json_path)

    # Load lists of json files
    paths = []
    for fn in sorted(os.listdir(os.path.join(json_path))):
        if fn.endswith('bdd_3d.json'):
            paths.append(os.path.join(json_path, fn))

    assert len(paths), "Not label files found in {}".format(json_path)

    return paths


def convert_app(det_seq, trk_seq, save_path):
    for fr_idx, (det_frm, trk_frm) in enumerate(zip(det_seq, trk_seq)):
        cam_calib = np.array(det_frm['intrinsics']['cali'])
        for det_idx, det_out in enumerate(det_frm['prediction']):
            depth_pd = det_out['box3d']['location'][2]

            for td in trk_frm:
                match = np.where(np.sum(bh.get_box2d_array([det_out]) == 
                            np.array(td['det_box']), axis=1) == 5)[0]
                if len(match):
                    #print(match, depth_pd, td['depth'], det_out['box3d']['xc'], td['x'], det_out['box3d']['yc'], td['y'])
                    depth_pd = td['depth']
                    det_out['id'] = td['id']
                    det_out['box3d']['xc'] = td['x']
                    det_out['box3d']['yc'] = td['y']

            # In camera coordinates
            location = tu.imagetocamera(
                            np.array([[det_out['box3d']['xc'], det_out['box3d']['yc']]]),
                            np.array([depth_pd]),
                            cam_calib)

            det_out['box3d']['location'] = location.tolist()

    print("Saving updated tracking results with {} frames at {}...".format(
        len(det_seq), save_path))
    dump_output_json(save_path, det_seq)


if __name__ == '__main__':

    print(' '.join(sys.argv))
    args = parse_args()

    METHODS = ['none', 'kf2d', 'kf2ddeep', 
                 'kf3d', 'kf3ddeep', 'kf3docc',
                 'kf3doccdeep', 'lstmoccdeep', 
                 'lstmdeep', 'lstm', 'lstmocc']
    setting_str = '_age{AGE}_aff{AFF}_hit{HIT}_{DEP}m_{CKPT}'.format(
        **{'AGE': args.max_age, 
            'AFF': args.affinity_thres, 
            'HIT': args.min_hits, 
            'DEP': args.max_depth, 
            'CKPT': args.lstm_name})

    # Load data
    json_path = '{ROOT}{SESS}_{EP}_{DT}_{PH}_set/'.format(
            **{'ROOT': args.root, 'SESS': args.session, 'EP': args.epoch, 
                'DT': args.set, 'PH': args.split})
    data_path = load_label_path(json_path)

    for method in METHODS:
        # Update input json path and save path
        method = method + setting_str
        trk_path = '{ROOT}{SESS}_{EP}_{DT}_{PH}_set/{MT}_pd.json'.format(
            **{'ROOT': args.root, 'SESS': args.session, 'EP': args.epoch, 
                'DT': args.set, 'PH': args.split, 'MT': method})
        save_path = '{ROOT}{SESS}_{EP}_{DT}_{PH}_set/{MT}/data/'.format(
            **{'ROOT': args.root, 'SESS': args.session, 'EP': args.epoch, 
                'DT': args.set, 'PH': args.split, 'MT': method})

        # Load tracked results
        trk_result = bh.load_json(trk_path)
        print(len(trk_result), len(data_path))

        for seq_idx, dpath in enumerate(data_path):

            trk_seq = [n['hypotheses'] for n in trk_result[seq_idx]['frames']]
            det_seq = bh.load_json(dpath)
            convert_app(det_seq, trk_seq, os.path.join(save_path, os.path.basename(dpath)))

