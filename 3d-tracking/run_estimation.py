import os
import sys
import argparse
import pickle
import subprocess
from time import sleep

'''
Multiple GPUs and processes script for monocular 3D Tracking
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Monocular 3D Estimation',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('set', choices=['gta', 'kitti'])
    parser.add_argument('split', choices=['train', 'val', 'test'], 
                        help='Which data split to use in testing')
    parser.add_argument('--session', default='616',
                        help='Name of the session, to separate exp')
    parser.add_argument('--epoch', default='030',
                        help='How many epochs you used to separate exp')
    parser.add_argument('--flag', default='-j 4 -b 1 --n_box_limit 300',
                        help='Flags for running evaluation code')
    parser.add_argument('--gpu', type=str, default='0,1,2,3,4', 
                        help='Which GPU to use in testing.')
    parser.add_argument('--n_tasks', type=int, default=1,
                        help='number of tasks running per GPU. n=1 is enough.')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Show command without running')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite the output files')
    parser.add_argument('--not_gen_output', action='store_true', default=False,
                        help='Run 3D estimation and store tracking info')
    parser.add_argument('--not_merge_result', action='store_true', default=False,
                        help='Merge 3D result for tracking')
    args = parser.parse_args()
    args.gen_output = not args.not_gen_output
    args.merge_result = not args.not_merge_result

    return args


print(' '.join(sys.argv))
args = parse_args()
GPUS = args.gpu.split(',')

# Metadata
if args.set == 'gta':
    JSON_ROOT = './data/gta5_tracking/{}/label/'.format(args.split)
    CMD = 'python mono_3d_estimation.py gta test \
--data_split {SPLIT} \
--resume ./checkpoint/{CKPT} \
--json_path {JSON} \
--track_name {TRK} \
--session {SESS} {FLAG} --start_epoch {EPOCH}'

else:
    JSON_ROOT = './data/kitti_tracking/{}ing/label_02/'.format(args.split)
    CMD = 'python mono_3d_estimation.py kitti test \
--data_split {SPLIT} \
--resume ./checkpoint/{CKPT} \
--json_path {JSON} \
--track_name {TRK} \
--is_tracking \
--is_normalizing \
--session {SESS} {FLAG} --start_epoch {EPOCH}'


SAVE_PATH = 'output/{SESS}_{EP}_{SET}_{SPLIT}_set/'.format(
    **{'SESS': args.session, 'EP': args.epoch, 'SET': args.set, 'SPLIT': args.split})
if not os.path.isdir(SAVE_PATH):
    print("Making {}...".format(SAVE_PATH))
    os.mkdir(SAVE_PATH)

CKPT = '{}_{}_checkpoint_{}.pth.tar'.format(args.session, args.set, args.epoch)
SAVE_NAME = '{PATH}{SESS}_{EP}_{SET}_roipool_output.pkl'.format(
    **{'PATH': SAVE_PATH, 'SESS': args.session, 'EP': args.epoch, 'SET': args.set})
JSON_PATHS = sorted(
    [n for n in os.listdir(JSON_ROOT) if n.endswith('bdd.json')])

# Script
def gen_3d_output():
    m = len(GPUS) * args.n_tasks
    ps = []
    for i in range(len(JSON_PATHS) // m + 1):
        for JSON, GPU in zip(JSON_PATHS[m * i:m * i + m], GPUS * args.n_tasks):
            TRK = '{}{}_{}_{}_roipool_output.pkl'.format(
                SAVE_PATH.replace('output/', ''), args.session, args.epoch,
                JSON.replace('.json', ''))
            cmd = CMD.format(
                **{'CKPT': CKPT, 'JSON': os.path.join(JSON_ROOT, JSON),
                   'TRK': TRK, 'SESS': args.session, 'EPOCH': args.epoch, 
                   'FLAG': args.flag, 'SPLIT': args.split})
            print(i, GPU, cmd)
            if not args.dry_run:
                if not args.overwrite and os.path.isfile(os.path.join('output', TRK)):
                    print("SKIP running. Generated file {} Found".format(TRK))
                    continue
                subprocess_env = os.environ.copy()
                subprocess_env['CUDA_VISIBLE_DEVICES'] = GPU
                p = subprocess.Popen(cmd, shell=True, env=subprocess_env)
                ps.append(p)
                sleep(1)
        if not args.dry_run:
            for p in ps:
                p.wait()


def merge_3d_results():
    all_pkl = []
    for JSON in JSON_PATHS:
        TRK = '{}{}_{}_{}_roipool_output.pkl'.format(SAVE_PATH, args.session, args.epoch,
                                                     JSON.replace('.json', ''))
        print("Reading {}...".format(TRK))
        if not args.dry_run: all_pkl += pickle.load(open(TRK, 'rb'))

    if not args.dry_run and len(all_pkl) > 0:
        print("Save to {}".format(SAVE_NAME))
        with open(SAVE_NAME, 'wb') as f:
            pickle.dump(all_pkl, f)


if __name__ == '__main__':
    if args.gen_output:
        gen_3d_output()
    if args.merge_result:
        merge_3d_results()

