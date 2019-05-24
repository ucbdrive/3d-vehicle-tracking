import os
import argparse
import subprocess
from time import sleep


'''
Dataset downloading script for monocular 3D Tracking
'''

link_root = 'http://dl.yf.io/bdd-data/3d-vehicle-tracking/'
path_root = './data/zip/'

_items = {
'note': ['README.md'],
'checkpoint': ['3d_tracking_checkpoint.zip', 'faster_rcnn_checkpoint.zip'],
'label': 
    {'train': ['label/gta_3d_tracking_train_label_{:04d}.zip'.format(i) for i in range(1, 101)],
    'test': ['label/gta_3d_tracking_test_label_{:04d}.zip'.format(i) for i in range(1, 41)],
    'val': ['label/gta_3d_tracking_val_label_{:04d}.zip'.format(i) for i in range(1, 11)]},
'image': 
    {'train': ['image/gta_3d_tracking_train_image_{:04d}.zip'.format(i) for i in range(1, 101)],
    'test': ['image/gta_3d_tracking_test_image_{:04d}.zip'.format(i) for i in range(1, 41)],
    'val': ['image/gta_3d_tracking_val_image_{:04d}.zip'.format(i) for i in range(1, 11)]},
'detection': ['kitti_detections_RRC.zip', 'gta_3d_tracking_detections.zip']
}


def parse_args():
    parser = argparse.ArgumentParser(description='Monocular 3D Tracking',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('task', choices=['all', 'mini', 'image', 'label', 
                        'checkpoint', 'detection'], 
                        help='Which data split to use in testing')
    parser.add_argument('split', choices=['train', 'val', 'test'], 
                        default='val', nargs='?', 
                        help='Which data split to use in testing')
    parser.add_argument('--n_tasks', type=int, default=5,
                        help='number of tasks running.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite the output files')
    args = parser.parse_args()

    return args


def gen_task(args):
    files = get_link('note', None)
    if args.task == 'all':
        for task in _items:
            for split in ['train', 'val', 'test']:
                files += get_link(task, split)
    elif args.task == 'mini':
        for task in _items:
            files += get_link(task, 'val')
    else:
        files += get_link(args.task, args.split)
    downloading(files, args.n_tasks, overwrite=args.overwrite)


def get_link(task, split):
    if task in ['image', 'label']:
        return [link_root + it for it in _items[task][split]]
    else:
        return [link_root + it for it in _items[task]]


def downloading(files, n_tasks, overwrite=False):
    ps = []
    for i in range(len(files) // n_tasks + 1):
        for link in files[n_tasks*i:n_tasks*(i+1)]:
            if not overwrite and os.path.isfile(os.path.basename(link)):
                print("SKIP running. Downloaded file {} Found".format(link))
                continue
            p = subprocess.Popen('wget -N {0} -P {1}'.format(link, path_root), shell=True)
            ps.append(p)
            sleep(1)

        for p in ps:
            p.wait()

if __name__ == '__main__':
    args = parse_args()
    gen_task(args)

