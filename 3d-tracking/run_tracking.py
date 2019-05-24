import os
import sys
import argparse
import subprocess
from time import sleep

'''
Multiple GPUs and processes script for monocular 3D Tracking
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Monocular 3D Tracking',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('set', choices=['gta', 'kitti'])
    parser.add_argument('split', choices=['train', 'val', 'test'], 
                        help='Which data split to use in testing')
    parser.add_argument('--session', default='616',
                        help='Name of the session, to separate exp')
    parser.add_argument('--epoch', default='030',
                        help='How many epochs you used to separate exp')
    parser.add_argument('--flag', default='-j 1',
                        help='Flags for running evaluation code')
    parser.add_argument('--lstm_name', default='803',
                        help='lstm ckpt name')
    parser.add_argument('--gpu', type=str, default='0,1,2', 
                        help='Which GPU to use in testing. Default is 0,1,2.')
    parser.add_argument('--n_tasks', type=int, default=1,
                        help='number of tasks running per GPU. n=1 is enough.')
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
    parser.add_argument('--not_gen_output', action='store_true', default=False,
                        help='Run 3D estimation and store tracking info')
    parser.add_argument('--not_merge_result', action='store_true', default=False,
                        help='Merge 3D result for tracking')
    args = parser.parse_args()

    return args


print(' '.join(sys.argv))
args = parse_args()

# Task Arguments
GPUS = ['--gpu {}'.format(gpu) for gpu in args.gpu.split(',')]
#'''
_METHOD_NAMES = ['none', 'kf2d', 'kf2ddeep', 
                 'kf3d', 'kf3ddeep', 'kf3docc',
                 'kf3doccdeep', 'lstmoccdeep', 
                 'lstmdeep', 'lstm', 'lstmocc']
'''
_METHOD_NAMES = ['kf3d', 'kf3ddeep', 'kf3doccdeep',
                 'lstmoccdeep', 'lstmdeep', 'lstm']
#'''

ID = '{}_{}'.format(args.session, args.epoch)
FLAG = '--max_age {AGE} --affinity_thres {AFF} --min_hits {HIT} --max_depth {DEP} {FLAG}'.format(
    **{'AGE': args.max_age, 'AFF': args.affinity_thres, 'HIT': args.min_hits, 'DEP': args.max_depth, 'FLAG': args.flag})
setting_str = 'age{AGE}_aff{AFF}_hit{HIT}_{DEP}m_{CKPT}'.format(
    **{'AGE': args.max_age, 'AFF': args.affinity_thres, 'HIT': args.min_hits, 'DEP': args.max_depth, 'CKPT': args.lstm_name})

# Metadata
_PATH = 'output/{ID}_{SET}_{PH}_set/'.format(**{'ID': ID, 'SET': args.set, 'PH': args.split})
_OUT = '{}{}_{}'.format(_PATH, '{}', setting_str)

_METHODS = {'none': '',
            'kf2d': '--kf2d',
            'kf2ddeep': '--kf2d --deep',
            'kf3d': '--kf3d',
            'kf3ddeep': '--kf3d --deep',
            'kf3docc': '--kf3d --occ',
            'kf3doccdeep': '--kf3d --occ --deep',
            'lstmoccdeep': '--lstm --occ --deep',
            'lstmdeep': '--lstm --deep',
            'lstmocc': '--lstm --occ',
            'lstm': '--lstm',
            }

CMD = "python mono_3d_tracking.py "\
"{SET} "\
"--path {PATH} "\
"--out_path {OUT} "\
"{METHOD} "\
"{FLAG} "\
"{GPU} "

# Script
def run_3d_tracking():
    m = len(GPUS) * args.n_tasks
    ps = []
    for i in range(len(_METHOD_NAMES) // m + 1):
        for method, GPU in zip(_METHOD_NAMES[m * i:m * i + m], GPUS * args.n_tasks):
            cmd = CMD.format(**{'SET': args.set,
                                'PATH': _PATH,
                                'OUT': _OUT.format(method),
                                'METHOD': _METHODS[method],
                                'FLAG': FLAG,
                                'GPU': GPU})
            print(i, GPU, cmd)
            if not args.dry_run:
                if not args.overwrite and os.path.isfile('{}_pd.json'.format(_OUT.format(method))):
                    print("SKIP running. Generated file {} Found".format(_OUT.format(method)))
                    continue
                p = subprocess.Popen(cmd, shell=True)
                ps.append(p)
                sleep(1)
        if not args.dry_run:
            for p in ps:
                p.wait()

if __name__ == '__main__':
    run_3d_tracking()
