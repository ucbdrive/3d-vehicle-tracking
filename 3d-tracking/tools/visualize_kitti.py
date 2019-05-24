import os
import re
import sys
import argparse
import json
import numpy as np
from glob import glob
import cv2

from utils.plot_utils import RandomColor


def parse_args():
    parser = argparse.ArgumentParser(
                    description='Monocular 3D Tracking Visualizer',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('set', choices=['gta', 'kitti'])
    parser.add_argument('split', choices=['train', 'val', 'test'], 
                        help='Which data split to use in testing')
    parser.add_argument('--session', default='623',
                        help='Name of the session, to separate exp')
    parser.add_argument('--epoch', default='100',
                        help='How many epochs you used to separate exp')
    parser.add_argument('--flag', default='kf3doccdeep_age15_aff0.1_hit0_80m_pd',
                        help='Flags for running evaluation code')
    parser.add_argument('--save_vid', action='store_true', default=False,
                        help='Flags for saving video')
    parser.add_argument('--save_txt', action='store_true', default=False,
                        help='Flags for saving txt')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Show command without running')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite the output files')
    args = parser.parse_args()

    return args


print(' '.join(sys.argv))
args = parse_args()

if args.set == 'kitti':
    IMAGE_PATH = 'data/kitti_tracking/{SPLIT}ing/image_02/{SEQ}/*.png'.format(**{'SPLIT': args.split, 'SEQ': '{:04d}'})
    re_pattern = re.compile('[0-9]{4}')
else:
    IMAGE_PATH = 'data/gta5_tracking/{SPLIT}/image/{SEQ}/*.jpg'.format(**{'SPLIT': args.split, 'SEQ': '{}'})
    re_pattern = re.compile('rec_(.{8})_(.+)_(.+)h(.+)m_(.+[0-9])')

SAVE_PATH = 'output/{SESS}_{EP}_{SET}_{SPLIT}_set/'.format(
    **{'SESS': args.session, 'EP': args.epoch, 'SET': args.set, 'SPLIT': args.split})
out_name = '{SESS}_{EP}_{SET}_{SETTING}'.format(
    **{'SESS': args.session, 'EP': args.epoch, 'SET': args.set, 'SETTING': args.flag})


FONT = cv2.FONT_HERSHEY_SIMPLEX
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15

np.random.seed(777)
rm_color = RandomColor(30)
tid2color = {}

def mkdir(path):
    if not os.path.isdir(path):
        print("Making directory {}".format(path))
        os.makedirs(path) # Use with care

def gen_result(out_path, out_name, save_vid=False, save_txt=True, 
                                    dry_run=False, overwrite=False):
    print("Reading meta data...")
    info = json.load(open('{}{}.json'.format(out_path, out_name), 'r'))

    if not dry_run: mkdir('{}{}/data/'.format(out_path, out_name))

    for seqid in range(len(info)):
        file_seq = re_pattern.search(info[seqid]['filename']).group(0)
        print('Reading {} from {}{}...'.format(file_seq, out_path, out_name))
        if dry_run:
            continue

        seqout = []
        vid_name = '{}{}/data/{}.mp4'.format(out_path, out_name, file_seq)
        txt_name = '{}{}/data/{}.txt'.format(out_path, out_name, file_seq)

        if not overwrite:
            if not os.path.isfile(txt_name) and save_txt:
                pass
            elif not os.path.isfile(vid_name) and save_vid:
                pass
            else:
                print("SKIP running. Generated file {} Found".format(txt_name))
                continue

        if save_vid:
            images = sorted(glob(IMAGE_PATH.format(file_seq)))
            img = cv2.imread(images[0])
            vidsize = (img.shape[1], img.shape[0]) # height, width
            out = cv2.VideoWriter(vid_name, FOURCC, fps, vidsize)

        demoinfo = info[seqid]['frames']
        for idx, frame in enumerate(demoinfo):
            if save_vid: 
                img = cv2.imread(images[idx])
                img = cv2.putText(img, str(idx), (20, 30),
                          cv2.FONT_HERSHEY_COMPLEX, 1,
                          (180, 180, 180), 2)
            for trk in frame['hypotheses']:
                x1, y1, x2, y2, conf = trk['det_box']
                xc, yc = trk['xc'], trk['yc']

                if save_vid:
                    if trk['id'] not in tid2color:
                        tid2color[trk['id']] = rm_color.get_random_color(scale=255)
                    img = cv2.rectangle(img, (int(xc-1), int(yc-1)), (int(xc+1), int(yc+1)),
                                        tid2color[trk['id']], 2)
                    img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                        tid2color[trk['id']], 4)
                    img = cv2.putText(img, str(int(trk['id'])), (int(x1), int(y1)),
                                      cv2.FONT_HERSHEY_COMPLEX, 1,
                                      tid2color[trk['id']], 2)
                    img = cv2.putText(img, str(int(trk['depth'])), (int(x2)-14, int(y2)),
                                      cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                      tid2color[trk['id']], 2)
                if save_txt:
                    '''
                    submit_txt = ' '.join([
                                        str(idx), 
                                        str(int(trk['id'])), 
                                        'Car', 
                                        '-1 -1',
                                        trk['alpha'],
                                        str(x1), str(y1), str(x2), str(y2), 
                                        trk['dim'],
                                        trk['loc'],
                                        trk['rot'], 
                                        str(conf)])
                    '''
                    submit_txt = ' '.join([
                                        str(idx), 
                                        str(int(trk['id'])), 
                                        'Car', 
                                        '-1 -1 -10',
                                        str(x1), str(y1), str(x2), str(y2), 
                                        '-1 -1 -1',
                                        '-1000 -1000 -1000 -10', 
                                        str(conf)])
                    #'''
                    submit_txt += '\n'
                    seqout.append(submit_txt)
            if save_vid: out.write(img)

        if save_txt:
            print("{} saved.".format(txt_name))
            with open(txt_name, 'w') as f:
                f.writelines(seqout)

        if save_vid:
            print("{} saved.".format(vid_name))
            out.release()

if __name__ == '__main__':

    # Not using out_name, too slow
    output_list = [os.path.splitext(item)[0] for item in os.listdir(SAVE_PATH) if item.endswith('_pd.json')]
    my_list = ['none', 'kf2ddeep', 'kf3doccdeep', 'lstmdeep', 'lstmoccdeep']

    for dir_name in output_list:
        print(dir_name)
        save_vid = args.save_vid
        if save_vid:
            is_in = False
            for ml in my_list:
                is_in = is_in or (ml in dir_name)
            save_vid = is_in

        gen_result(SAVE_PATH, 
                    dir_name,
                    save_vid=save_vid, 
                    save_txt=args.save_txt,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite
                    )
