import argparse
import numpy as np
import os
import pickle
import re

import utils.bdd_helper as bh
import utils.tracking_utils as tu
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
    parser.add_argument('-m', '--metadata', dest='metadata',
                        help='gta metadata name',
                        default='track_parse.json', type=str)
    parser.add_argument('-c', '--load_cache', dest='load_cache',
                        help='load cache files',
                        default=False, action='store_true')
    parser.add_argument('--kitti_task', default='track',
                        choices=['detect', 'track'],
                        help='KITTI task [detect, track]')
    parser.add_argument('--is_pred', dest='is_pred',
                        help='generate detection on data set',
                        default=False, action='store_true')
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
    focal = cfg.GTA.FOCAL_LENGTH  # 935.3074360871937
    im_size_h = cfg.GTA.H  # 1080
    im_size_w = cfg.GTA.W  # 1920
    DATASET = cfg.GTA.TRACKING

    DATASET.PKL_FILE = os.path.join(DATASET.PATH,
                            'gta_{}_labels.pkl'.format(args.split))
    det_name = os.path.join(DATASET.PATH,
                            'gta_{}_detections.pkl'.format(args.split))
    save_name = os.path.join(DATASET.PATH,
                            'gta_{}_list.json'.format(args.split))
    DATASET.LABEL_PATH = DATASET.LABEL_PATH.replace('train', args.split)
    DATASET.PRED_PATH = DATASET.PRED_PATH.replace('train', args.split)
    DATASET.IM_PATH = DATASET.IM_PATH.replace('train', args.split)
else:
    # KITTI tracking / object dataset
    focal = cfg.KITTI.FOCAL_LENGTH  # 721.5377
    im_size_h = cfg.KITTI.H  # 384
    im_size_w = cfg.KITTI.W  # 1248
    if args.kitti_task == 'detect':
        DATASET = cfg.KITTI.OBJECT
        DATASET.PKL_FILE = os.path.join(DATASET.PATH,
                            'kitti_{}_obj_labels.pkl'.format(
                                            args.split))
        det_name = os.path.join(DATASET.PATH,
                            'kitti_{}_obj_detections.pkl'.format(
                                args.split))
        save_name = os.path.join(DATASET.PATH,
                            'kitti_{}_obj_list.json'.format(args.split))
    elif args.kitti_task == 'track':
        DATASET = cfg.KITTI.TRACKING
        DATASET.PKL_FILE = os.path.join(DATASET.PATH,
                            'kitti_{}_trk_labels.pkl'.format(
                                args.split))
        det_name = os.path.join(DATASET.PATH,
                            'kitti_{}_trk_detections.pkl'.format(
                                args.split))
        save_name = os.path.join(DATASET.PATH,
                             'kitti_{}_trk_list.json'.format(args.split))

    DATASET.LABEL_PATH = DATASET.LABEL_PATH.replace('train', args.split)
    DATASET.PRED_PATH = DATASET.PRED_PATH.replace('train', args.split)
    DATASET.IM_PATH = DATASET.IM_PATH.replace('train', args.split)
    DATASET.PKL_FILE = DATASET.PKL_FILE.replace('train', args.split)
    DATASET.CALI_PATH = DATASET.CALI_PATH.replace('train', args.split)
    DATASET.OXT_PATH = DATASET.OXT_PATH.replace('train', args.split)
    assert args.split in ['train', 'test'], "KITTI has no validation set"


def load_label_path(json_path):
    assert os.path.isdir(json_path), "Empty path".format(json_path)

    # Load lists of json files
    json_name = args.metadata
    folder_names = [os.path.join(json_path, n) for n in
                    sorted(os.listdir(json_path))]
    label_paths = [os.path.join(n, json_name) for n in folder_names if
                   os.path.isfile(os.path.join(n, json_name))]
    assert len(label_paths), "Not label files found in {}".format(json_path)

    return label_paths


def load_label_pickle(DATASET):
    with open(DATASET.PKL_FILE, 'rb') as f:
        return pickle.load(f)


class Dataset():
    def __init__(self):

        self.split = args.split
        self.mode = args.mode
        self.kitti_task = args.kitti_task

        # load data
        if args.load_cache:
            print("Load pickle file: {}".format(DATASET.PKL_FILE))
            self.data_label = load_label_pickle(DATASET)
        else:
            print("Load label file from path: {}".format(DATASET.LABEL_PATH))
            if args.set == 'gta':
                self.data_label = self.gta_label()
            elif args.kitti_task == 'detect':
                self.data_label = self.kitti_detection_label()
            else:
                self.data_label = self.kitti_tracking_label()

        if args.is_pred:
            with open(det_name, 'rb') as f:
                det_result = pickle.load(f)
            det_result = np.array(det_result[1])
            assert sum([len(n) for n in self.data_label]) == det_result.shape[
                0], \
                'ERROR: length of GT frames {} not equals detected ones {' \
                '}'.format(
                    sum([len(n) for n in self.data_label]), det_result.shape[0])
        else:
            det_result = None

        self.data_idx = []
        self.box_idx = []
        self.det_result = []
        data_len = [len(n) for n in self.data_label]

        st = 0
        for d_idx, data_label in enumerate(self.data_label):
            det_ = det_result[st:st + data_len[d_idx]] if args.is_pred else None
            data_idx, box_idx, det_res = self.clean_idx(d_idx, list(data_label),
                                                        det_)
            st += data_len[d_idx]
            self.data_idx.append(data_idx)
            self.box_idx.append(box_idx)
            self.det_result.append(det_res)
        self.data_len = [len(n) for n in self.data_idx]
        print(
            "Overall Data length {} Valid data length {}".format(sum(data_len),
                                                                 sum(
                                                                     self.data_len)))
        print("Data preparation done!!")

    def gta_label(self):
        # get information at boxes level. Collect dict. per box, not image.
        file_list = load_label_path(os.path.join(DATASET.LABEL_PATH))
        print("{} with {} sequences".format(args.split, len(file_list)))
        data_label_list = []
        re_pattern = re.compile('rec_(.{8})_(.+)_(.+)h(.+)m_(.+)')
        # rec_10090911_clouds_21h53m_x-968y-1487tox2523y214
        # ('10090911', 'clouds', '21', '53', 'x-968y-1487tox2523y214')
        for fidx, filename in enumerate(file_list):
            print(fidx, filename)
            dataset = bh.load_json(filename)
            n_fr = len(dataset)
            data_label = {}
            for fr_idx, frame in enumerate(dataset):
                img_name = os.path.join(
                                        frame['dset_name'],
                                        str(frame['timestamp']) + '_final.jpg'
                                        )

                num_boxes = len(frame['object'])
                obj = frame['object']
                data_label[img_name] = {}
                data_label[img_name]['vid_name'] = os.path.join(
                                                    frame['dset_name'])

                img_log = re_pattern.match(frame['dset_name'])
                img_weather = img_log.group(2)
                img_hour = img_log.group(3)
                
                data_label[img_name]['weather'] = img_weather
                data_label[img_name]['timeofday'] = bh.get_time_of_day(int(img_hour))
                data_label[img_name]['timestamp'] = frame['timestamp']
                data_label[img_name]['fov'] = frame['camera']['fov']
                data_label[img_name]['nearClip'] = frame['camera']['nearClip']
                data_label[img_name]['pose'] = {
                                    'rotation': [np.pi * angle / 180.0
                                         for angle in frame['pose']['rotation']],
                                    'position': [p_t - p_0 for (p_t, p_0) in \
                                            zip(frame['pose']['position'],
                                                dataset[0]['pose']['position'])]}
                data_label[img_name]['img_name'] = img_name
                data_label[img_name]['num_boxes'] = num_boxes
                data_label[img_name]['pixel'] = [obj[i]['n_pixel'] for i in
                                                 range(num_boxes)]
                data_label[img_name]['class'] = [
                    DATASET.CLASS_PARSER[obj[i]['kitti']['type']] for i in
                    range(num_boxes)]
                data_label[img_name]['ignore'] = [obj[i]['ignore'] for i in
                                                  range(num_boxes)]
                data_label[img_name]['tracking_id'] = [obj[i]['tracking_id'] for
                                                       i in range(num_boxes)]
                data_label[img_name]['truncated'] = [
                    obj[i]['kitti']['truncated'] for i in range(num_boxes)]
                data_label[img_name]['occluded'] = [obj[i]['kitti']['occluded']
                                                    for i in range(num_boxes)]
                data_label[img_name]['boxes'] = [obj[i]['kitti']['bbox'] for i
                                                 in range(num_boxes)]
                data_label[img_name]['alpha'] = [obj[i]['kitti']['alpha'] for i
                                                 in range(num_boxes)]
                data_label[img_name]['dims'] = [obj[i]['kitti']['dimensions']
                                                for i in range(num_boxes)]
                data_label[img_name]['trans'] = [obj[i]['kitti']['location'] for
                                                 i in range(num_boxes)]
                data_label[img_name]['rot_y'] = [obj[i]['kitti']['rotation_y']
                                                 for i in range(num_boxes)]
            data_label_list.append(data_label)
        with open(os.path.join(DATASET.PKL_FILE), 'wb') as f:
            pickle.dump(data_label_list, f, -1)
        return data_label_list

    def kitti_detection_label(self):
        # get information at boxes level. Collect dict. per box, not image.
        file_list = sorted(
            [n for n in os.listdir(DATASET.IM_PATH) if n.endswith('png')])
        data_label = {}
        n_fr = len(file_list)
        for fidx, fname in enumerate(file_list):
            obj_dict = {}
            obj_dict['class'] = []
            obj_dict['truncated'] = []
            obj_dict['occluded'] = []
            obj_dict['boxes'] = []
            obj_dict['alpha'] = []
            obj_dict['dims'] = []
            obj_dict['trans'] = []
            obj_dict['rot_y'] = []
            obj_dict['num_boxes'] = []
            obj_dict['pixel'] = []
            obj_dict['ignore'] = []
            obj_dict['tracking_id'] = []
            obj_dict['vid_name'] = 'object'
            obj_dict['timestamp'] = fidx
            obj_dict['pose'] = {'rotation': [0, 0, 0], 'position': [0, 0, 0]}
            obj_dict['fov'] = 60
            obj_dict['nearClip'] = 0.15
            obj_dict['img_name'] = os.path.join(fname[:-4] + '.png')
            label_file = os.path.join(DATASET.LABEL_PATH,
                                      fname.replace('png', 'txt'))
            if os.path.isfile(label_file) and label_file.endswith('txt'):
                num_lines = sum(1 for line in open(label_file, 'r'))
                with open(label_file, 'r') as f:
                    for line in f:
                        obj_class, truncated, occluded, alpha, bx1, by1, bx2, \
                        by2, dz, dy, dx, tx, ty, tz, rot_y = line.split()
                        obj_dict['class'].append(obj_class)
                        obj_dict['tracking_id'].append(-1)
                        obj_dict['ignore'].append(obj_class == 'DontCare')
                        obj_dict['truncated'].append(float(truncated))
                        obj_dict['occluded'].append(float(occluded))
                        obj_dict['boxes'].append(
                            [int(float(bx1)), int(float(by1)), int(float(bx2)),
                             int(float(by2))])
                        obj_dict['alpha'].append(float(alpha))
                        obj_dict['dims'].append([float(dz), float(dy), float(
                            dx)])  # h(z), w(y), l (x)
                        obj_dict['trans'].append([float(tx), float(ty), float(
                            tz)])  # This is the center, bottom (on the ground)
                        obj_dict['rot_y'].append(float(rot_y))
                        obj_dict['num_boxes'].append(num_lines)
                        obj_dict['pixel'].append(args.min_pixel + 1)
            data_label[obj_dict['img_name']] = obj_dict
        data_label_list = [data_label]
        with open(os.path.join(DATASET.PKL_FILE), 'wb') as f:
            pickle.dump(data_label_list, f, -1)
        return data_label_list

    def kitti_tracking_label(self):
        # get information at boxes level. Collect dict. per box, not image.
        data_label_list = []
        # Tracking train has 21 folders, each is a sequence

        for n in range(len([ip[0] for ip in os.walk(DATASET.IM_PATH)]) - 1):
            fields = tu.read_oxts(DATASET.OXT_PATH, n)
            poses = [tu.KittiPoseParser(fields[i]) for i in range(len(fields))]
            projection = tu.read_calib(n, calib_dir=DATASET.CALI_PATH)
            seq_fname = os.path.join(DATASET.IM_PATH, str(n).zfill(4))
            file_list = sorted(
                [sf for sf in os.listdir(seq_fname) if sf.endswith('.png')])
            data_label = {}
            n_fr = len(file_list)
            for fr, fname in enumerate(file_list):
                # relative position from frame 0
                position = poses[fr].position - poses[0].position
                obj_dict = {}
                obj_dict['class'] = []
                obj_dict['truncated'] = []
                obj_dict['occluded'] = []
                obj_dict['boxes'] = []
                obj_dict['alpha'] = []
                obj_dict['dims'] = []
                obj_dict['trans'] = []
                obj_dict['rot_y'] = []
                obj_dict['num_boxes'] = []
                obj_dict['ignore'] = []
                obj_dict['tracking_id'] = []
                obj_dict['pixel'] = []
                obj_dict['vid_name'] = seq_fname[-4:]
                obj_dict['timestamp'] = fr
                obj_dict['fov'] = 60
                obj_dict['nearClip'] = -1
                # TODO: Add kitti pose
                obj_dict['cali'] = projection.tolist()
                obj_dict['pose'] = {
                    # pitch, -roll, yaw in camera coord
                    # for x-, y-, z- rotation in world coord.
                    'rotation': [poses[fr].roll, 
                                 poses[fr].pitch,
                                 poses[fr].yaw],
                    'position': position.tolist()}
                obj_dict['img_name'] = os.path.join(seq_fname[-4:], fname)

                label_file = os.path.join(seq_fname.replace('image', 'label'),
                                          fname.replace('png', 'txt'))
                if os.path.isfile(label_file):
                    num_lines = sum(1 for line in open(label_file, 'r'))
                    with open(label_file, 'r') as f:
                        for line in f:
                            _, tid, obj_class, truncated, occluded, alpha, \
                            bx1, by1, bx2, by2, dz, dy, dx, tx, ty, \
                            tz, rot_y = line.split()
                            obj_dict['tracking_id'].append(tid)
                            obj_dict['ignore'].append(obj_class == 'DontCare')
                            obj_dict['class'].append(obj_class)
                            obj_dict['truncated'].append(float(truncated))
                            obj_dict['occluded'].append(float(occluded))
                            obj_dict['boxes'].append(
                                [int(float(bx1)), int(float(by1)),
                                 int(float(bx2)), int(float(by2))])
                            obj_dict['alpha'].append(float(alpha))
                            # h(z), w(y), l(x)
                            obj_dict['dims'].append(
                                [float(dz), float(dy), float(dx)])
                            # This is the center, bottom (on the ground)
                            obj_dict['trans'].append(
                                [float(tx), float(ty), float(tz)]) 
                            obj_dict['rot_y'].append(float(rot_y))
                            obj_dict['num_boxes'].append(num_lines)
                            obj_dict['pixel'].append(args.min_pixel + 1)
                data_label[obj_dict['img_name']] = obj_dict

            data_label_list.append(data_label)
        with open(os.path.join(DATASET.PKL_FILE), 'wb') as f:
            pickle.dump(data_label_list, f, -1)
        return data_label_list

    def format_data2bdd(self):
        dataset_list = []
        for d_idx in range(len(self.data_idx)):
            dataset_list.append(
                self.get_data2bdd(
                    self.data_idx[d_idx],
                    self.data_label[d_idx],
                    self.box_idx[d_idx],
                    self.det_result[d_idx]))
        print("Saving {} with {} sequences...".format(save_name,
                                                      len(dataset_list)))
        if not args.is_pred: bh.dump_json(save_name, dataset_list)
        print("Saved")

    def get_data2bdd(self, data_idx, data_label, box_idx, det_result):
        seq_data = []
        if data_idx == []:
            print("Empty data")
            return seq_data

        for idx, im_path in enumerate(data_idx):

            check_avail = box_idx[idx]
            obj = data_label[im_path]

            if idx % args.verbose_interval == 0 and idx != 0:
                print('{} images.'.format(idx))

            frame_data = bh.init_frame_format()

            frame_data['name'] = im_path
            del frame_data['url']
            frame_data['videoName'] = obj['vid_name']
            frame_data['resolution']['height'] = im_size_h
            frame_data['resolution']['width'] = im_size_w
            if 'weather' in obj and 'timeofday' in obj:
                frame_data['attributes']['weather'] = obj['weather']
                frame_data['attributes']['timeofday'] = obj['timeofday']
            else:
                del frame_data['attributes']

            frame_data['frameIndex'] = idx
            if 'cali' in obj:
                frame_data['intrinsics']['cali'] = obj['cali']
                frame_data['intrinsics']['focal'] = [obj['cali'][0][0],
                                                     obj['cali'][1][1]]
                frame_data['intrinsics']['center'] = [obj['cali'][0][2],
                                                      obj['cali'][1][2]]
            else:
                frame_data['intrinsics']['cali'] = [
                    [focal, 0, im_size_w // 2, 0],
                    [0, focal, im_size_h // 2, 0],
                    [0, 0, 1, 0]]
                frame_data['intrinsics']['focal'] = [focal, focal]
                frame_data['intrinsics']['center'] = [im_size_w // 2,
                                                      im_size_h // 2]
            frame_data['intrinsics']['fov'] = obj['fov']
            frame_data['intrinsics']['nearClip'] = obj['nearClip']
            # Camera relative rotation from previous frame (in rad)
            # Camera relative location in world coordinate (in meter)
            frame_data['extrinsics']['rotation'] = obj['pose']['rotation']
            frame_data['extrinsics']['location'] = obj['pose']['position']
            frame_data['timestamp'] = obj['timestamp']

            if args.is_pred:
                match_box = det_result[idx]

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
                    prediction_data.append(prediction)

                frame_data['prediction'] = prediction_data

                print("Frame {}, GT: {} Boxes, PD: {} Boxes".format(
                    idx,
                    len(frame_data['labels']),
                    len(frame_data['prediction'])))

                filename = os.path.join(
                                DATASET.PRED_PATH, 
                                os.path.splitext(im_path)[0] + '.json'
                                )
                
                del frame_data['labels']
                if not os.path.exists(os.path.dirname(filename)):
                    os.mkdir(os.path.dirname(filename))
                bh.dump_json(filename, frame_data)

            else:
                frame_data['labels'] = self.get_data_from_id_bdd(obj, check_avail)
                del frame_data['prediction']
                print("Frame {}, GT: {} Boxes".format(
                    idx,
                    len(frame_data['labels'])))

                filename = os.path.join(
                                DATASET.LABEL_PATH, 
                                os.path.splitext(im_path)[0] + '.json'
                                )
                bh.dump_json(filename, frame_data)
                seq_data.append(filename)

        filename = os.path.join(
                        DATASET.LABEL_PATH,
                        frame_data['videoName'] + '_bdd.json'
                        )
        print("Saving {} with {} frames...".format(filename, idx))
        if not args.is_pred: bh.dump_json(filename, seq_data)

        return seq_data

    def get_data_from_id_bdd(self, obj, check_avail):
        frame_data = []
        for idx in check_avail:
            # save data information
            box_data = bh.init_labels_format()
            box_data['id'] = obj['tracking_id'][idx]
            box_data['category'] = obj['class'][idx]

            box_data['attributes']['ignore'] = obj['ignore'][idx]
            box_data['attributes']['occluded'] = obj['occluded'][idx]
            box_data['attributes']['truncated'] = obj['truncated'][idx]

            box_data['box2d']['x1'] = obj['boxes'][idx][0]
            box_data['box2d']['y1'] = obj['boxes'][idx][1]
            box_data['box2d']['x2'] = obj['boxes'][idx][2]
            box_data['box2d']['y2'] = obj['boxes'][idx][3]
            box_data['box2d']['confidence'] = 1.0

            # In camera coordinates
            box_data['box3d']['alpha'] = obj['alpha'][idx]
            box_data['box3d']['orientation'] = obj['rot_y'][idx]
            box_data['box3d']['location'] = obj['trans'][idx]
            box_data['box3d']['dimension'] = obj['dims'][idx]

            if args.set == 'kitti':
                # Align KITTI object location (bottom center of 3D box)
                # with GTA (center of 3D box)
                box_data['box3d']['location'][1] -= obj['dims'][idx][0] / 2.0 

            del box_data['poly2d']
            del box_data['attributes']['trafficLightColor']
            del box_data['attributes']['areaType']
            del box_data['attributes']['laneDirection']
            del box_data['attributes']['laneStyle']
            del box_data['attributes']['laneTypes']
            frame_data.append(box_data)

        return frame_data

    def clean_idx(self, d_idx, data_key, det_result):
        valid_key_list = []
        valid_box_idx = []
        valid_det_list = []
        for i, key in enumerate(data_key):
            val_idx = self.is_valid_idx(d_idx, key)
            if args.is_pred:
                val_det = self.det_valid_idx(det_result[i])

            # NOTE: BN cannot use size = 1 batch. Be careful.
            is_frame_valid = (val_idx.shape[0] > 0) \
                             and (not args.is_pred or val_det.shape[0] > 0)

            is_frame_valid = is_frame_valid or (
                    self.mode == 'test' and self.kitti_task == 'track')

            if is_frame_valid:
                valid_key_list.append(key)
                valid_box_idx.append(val_idx)
                if args.is_pred:
                    if len(val_det) == 0:
                        valid_det_list.append(det_result[i])
                    else:
                        valid_det_list.append(det_result[i][val_det])
            else:
                print(i, key, val_idx, val_det if args.is_pred else '')

        print("Sequence {}: Data length: {}, Valid data length: {}".format(
            d_idx,
            len(data_key),
            len(valid_key_list)))
        return valid_key_list, valid_box_idx, valid_det_list

    def det_valid_idx(self, det_box):
        is_valid = []
        for j, det in enumerate(det_box):
            if (det[2] > det[0] + 2 and det[3] > det[1] + 2) \
                    or (self.mode == 'test' and self.kitti_task == 'track'):
                is_valid.append(j)
        return np.array(is_valid)

    def is_valid_idx(self, d_idx, key):
        obj = self.data_label[d_idx][key]
        is_valid = []
        for j in range(len(obj['class'])):
            if (obj['class'][j] in DATASET.CLASS_LIST \
                    and obj['dims'][j][0] > 0 \
                    and obj['trans'][j][2] > 0 \
                    and obj['trans'][j][2] < args.max_depth \
                    and obj['pixel'][j] > args.min_pixel \
                    and obj['boxes'][j][2] > obj['boxes'][j][0] + 2 \
                    and obj['boxes'][j][3] > obj['boxes'][j][1] + 2):
                is_valid.append(j)
        return np.array(is_valid)


def main():
    ds = Dataset()
    ds.format_data2bdd()


if __name__ == '__main__':
    main()
