import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import utils.bdd_helper as bh 
import utils.tracking_utils as tu
from utils.config import cfg


def copy_border_reflect(img, p_h, p_w):
    if p_h > 0:
        pad_down = img[-p_h:, :][::-1, :]
        img = np.vstack((img, pad_down))
    if p_w > 0:
        pad_right = img[:, -p_w:][:, ::-1]
        img = np.hstack((img, pad_right))
    return img


def load_label_path(json_path, extn_name):
    assert os.path.isdir(json_path), "Empty path".format(json_path)

    # Load lists of json files
    folders = [os.path.join(json_path, n) for n in
                    sorted(os.listdir(json_path)) if
                    os.path.isdir(os.path.join(json_path, n))]
    paths = [os.path.join(n, fn) 
                for n in folders 
                    for fn in sorted(os.listdir(n))
                        if os.path.isfile(os.path.join(n, fn)) and
                           fn.endswith(extn_name)]
    assert len(paths), "Not label files found in {}".format(json_path)

    return paths


class Dataset(Dataset):
    def __init__(self, json_path, phase, split, use_kitti, percent, 
                    is_tracking, is_normalizing, n_box_limit):
        self.json_path = json_path
        self.phase = phase
        self.split = split
        self.use_kitti = use_kitti
        self.percent = percent
        self.is_tracking = is_tracking
        self.is_normalizing = is_normalizing
        self.n_box_limit = n_box_limit
        assert self.phase in ['train', 'val', 'test']

        if self.use_kitti:
            self.H = cfg.KITTI.H  # 384
            self.W = cfg.KITTI.W  # 1248
            self.focal = cfg.KITTI.FOCAL_LENGTH  # 721
            if is_tracking:
                DATASET = cfg.KITTI.TRACKING
            else:
                DATASET = cfg.KITTI.OBJECT
        else:
            # To make 1080 diviable by 32, we pad the bottom 8 pixels more
            self.H = cfg.GTA.H + 8  # 1080 + 8
            self.W = cfg.GTA.W  # 1920
            self.focal = cfg.GTA.FOCAL_LENGTH  # 935.3
            DATASET = cfg.GTA.TRACKING

        self.LABEL_PATH = DATASET.LABEL_PATH.replace('train', split)
        self.PRED_PATH = DATASET.PRED_PATH.replace('train', split)
        self.IM_PATH = DATASET.IM_PATH.replace('train', split)

        # Load files
        if self.json_path:
            # if train: read train, val; if test: read test
            if not self.use_kitti:
                json_path = self.json_path.replace('train', split)
            else:
                json_path = self.json_path
        else:
            json_path = os.path.join(self.LABEL_PATH, self.split)

        self.label_name = []
        self.seq_len = []
        print("Reading {} ...".format(json_path))
        json_ext = 'bdd.json'
        if os.path.isdir(json_path):
            print('Load lists of json files')
            label_paths = load_label_path(json_path, json_ext)
            for seq_idx, json_name in enumerate(label_paths):
                self.label_name.append(bh.read_labels(json_name))
                self.seq_len.append(len(self.label_name[seq_idx]))
            self.label_name = [frames for seqs in self.label_name for frames in
                               seqs]
        elif json_path.endswith(json_ext):
            print('Load single json file')
            self.label_name = bh.read_labels(json_path)
            self.seq_len.append(len(self.label_name))
        else:
            print('Load single bundled json file')
            self.label_name = bh.read_labels(json_path)
            for seq_idx, frames in enumerate(self.label_name):
                self.seq_len.append(len(frames))
            self.label_name = [frames for seqs in self.label_name for frames in
                               seqs]
        self.db_size = len(self.label_name)

        # Accumulated sequence length
        self.seq_accum = np.cumsum(self.seq_len)
        print("Sequences {} with total {} frames".format(self.seq_len,
                                                         self.db_size))

        # Normalize image or not
        if self.is_normalizing:
            self.get_mean_std(json_path)
        else:
            print("Input images are not normalized")

        # For separate train and val in kitti
        if self.percent != 100: print(
            "Warning: Using a subset of {}% data!".format(self.percent))
        select_idx = int(self.db_size * self.percent / 100.0)  # full

        total_idx = [idx for idx in range(self.db_size)]
        self.data_idx = total_idx[:select_idx]
        self.data_len = len(self.data_idx)

    def __getitem__(self, index):
        idx = self.data_idx[index]
        return self.load_image_sample(idx)

    def __len__(self):
        return self.data_len

    def get_mean_std(self, json_name):
        if 'kitti' in json_name:
            mean = [0.28679871, 0.30261545, 0.32524435]
            std = [0.27106311, 0.27234113, 0.27918578]
        elif 'gta' in json_name:
            mean = [0.34088846, 0.34000116, 0.35496006]
            std = [0.21032437, 0.19707282, 0.18238117]
        else:
            print("Not normalized!!")
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]

        self.mean = np.array(mean).reshape(3, 1, 1)
        self.std = np.array(std).reshape(3, 1, 1)

        print("Using mean: {} std: {} of {}".format(mean, std, json_name))

    def load_image_sample(self, idx):

        frame = json.load(open(self.label_name[idx], 'r'))
        pd_name = self.label_name[idx].replace('data', 'output')
        pd_name = pd_name.replace('label', 'pred')
        if os.path.isfile(pd_name):
            frame_pd = json.load(open(pd_name, 'r'))
        else:
            # No prediction json file found
            frame_pd = {'prediction': []}

        n_box = len(frame['labels'])
        if n_box > self.n_box_limit:
            # print("n_box ({}) exceed the limit {}, clip up to
            # limit.".format(n_box, self.n_box_limit))
            n_box = self.n_box_limit

        # Frame level annotations
        im_path = os.path.join(self.IM_PATH, frame['name'])
        endvid = int(idx + 1 in self.seq_accum)
        cam_rot = np.array(frame['extrinsics']['rotation'])
        cam_loc = np.array(frame['extrinsics']['location'])
        cam_calib = np.array(frame['intrinsics']['cali'])
        cam_focal = np.array(frame['intrinsics']['focal'])
        cam_near_clip = np.array(frame['intrinsics']['nearClip'])
        cam_fov_h = np.array(frame['intrinsics']['fov'])
        pose = tu.Pose(cam_loc, cam_rot, not self.use_kitti)

        # Object level annotations
        if self.phase in ['train', 'val']:
            labels = frame['labels'][:n_box]
            predictions = frame_pd['prediction'][:n_box]

            # Random shuffle data
            np.random.seed(777)
            np.random.shuffle(labels)
        else:
            labels = frame['labels']
            predictions = frame_pd['prediction']


        rois_pd = bh.get_box2d_array(predictions).astype(float)
        rois_gt = bh.get_box2d_array(labels).astype(float)
        tid = bh.get_label_array(labels, ['id'], (0)).astype(int)
        # Dim: H, W, L
        dim = bh.get_label_array(labels, ['box3d', 'dimension'], (0, 3)).astype(
            float)
        # Alpha: -pi ~ pi
        alpha = bh.get_label_array(labels, ['box3d', 'alpha'], (0)).astype(float)
        # Location in cam coord: x-right, y-down, z-front
        location = bh.get_label_array(labels, ['box3d', 'location'],
                                   (0, 3)).astype(float)

        # Center
        # f_x,   s, cen_x, ext_x
        #   0, f_y, cen_y, ext_y
        #   0,   0,     1, ext_z
        ext_loc = np.hstack([location, np.ones([len(location), 1])])  # (B, 4)
        proj_loc = ext_loc.dot(cam_calib.T)  # (B, 4) dot (3, 4).T => (B, 3)
        center_gt = proj_loc[:, :2] / proj_loc[:, 2:3]  # normalize

        if self.phase in ['train', 'val']:
            # For depth training
            #center_pd = center_gt.copy()
            # For center training
            cenx = (rois_gt[:, 0:1] + rois_gt[:, 2:3]) / 2
            ceny = (rois_gt[:, 1:2] + rois_gt[:, 3:4]) / 2
            center_pd = np.concatenate([cenx, ceny], axis=1)
        else:
            center_pd = bh.get_cen_array(predictions)

        # Depth
        depth = np.maximum(0, location[:, 2])

        ignore = bh.get_label_array(labels, ['attributes', 'ignore'], (0)).astype(
            int)
        # Get n_box_limit batch
        rois_gt = np.vstack([rois_gt, np.zeros([self.n_box_limit, 5])])[
                  :self.n_box_limit]
        if self.phase in ['train', 'val']:
            rois_pd = rois_gt.copy()
            rois_pd[:, :4] += np.random.rand(rois_gt.shape[0], 4) * 3
        else:
            rois_pd = np.vstack([rois_pd, np.zeros([self.n_box_limit, 5])])[
                      :self.n_box_limit]
        tid = np.hstack([tid, np.zeros(self.n_box_limit)])[:self.n_box_limit]
        alpha = np.hstack([alpha, np.zeros(self.n_box_limit)])[
                :self.n_box_limit]
        depth = np.hstack([depth, np.zeros(self.n_box_limit)])[
                :self.n_box_limit]
        center_pd = np.vstack([center_pd, np.zeros([self.n_box_limit, 2])])[
                    :self.n_box_limit]
        center_gt = np.vstack([center_gt, np.zeros([self.n_box_limit, 2])])[
                    :self.n_box_limit]
        dim = np.vstack([dim, np.zeros([self.n_box_limit, 3])])[
              :self.n_box_limit]
        ignore = np.hstack([ignore, np.zeros(self.n_box_limit)])[
                 :self.n_box_limit]

        # objects center in the world coordinates
        loc_gt = tu.point3dcoord(center_gt, depth, cam_calib, pose)

        # Load images
        img = cv2.imread(im_path)
        assert img is not None, "Cannot read {}".format(im_path)

        h, w, _ = img.shape
        p_h = self.H - h
        p_w = self.W - w
        assert p_h >= 0, "target hight - image hight = {}".format(p_h)
        assert p_w >= 0, "target width - image width = {}".format(p_w)
        img = copy_border_reflect(img, p_h, p_w)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_patch = np.rollaxis(img, 2, 0)
        img_patch = img_patch.astype(float) / 255.0

        # Normalize
        if self.is_normalizing:
            img_patch = (img_patch - self.mean) / self.std

        bin_cls = np.zeros((self.n_box_limit, 2))
        bin_res = np.zeros((self.n_box_limit, 2))

        for i in range(n_box):
            if alpha[i] < np.pi / 6. or alpha[i] > 5 * np.pi / 6.:
                bin_cls[i, 0] = 1
                bin_res[i, 0] = alpha[i] - (-0.5 * np.pi)

            if alpha[i] > -np.pi / 6. or alpha[i] < -5 * np.pi / 6.:
                bin_cls[i, 1] = 1
                bin_res[i, 1] = alpha[i] - (0.5 * np.pi)

        box_info = {
            'im_path': im_path,
            'rois_pd': torch.from_numpy(rois_pd).float(),
            'rois_gt': torch.from_numpy(rois_gt).float(),
            'dim_gt': torch.from_numpy(dim).float(),
            'bin_cls_gt': torch.from_numpy(bin_cls).long(),
            'bin_res_gt': torch.from_numpy(bin_res).float(),
            'alpha_gt': torch.from_numpy(alpha).float(),
            'depth_gt': torch.from_numpy(depth).float(),
            'cen_pd': torch.from_numpy(center_pd).float(),
            'cen_gt': torch.from_numpy(center_gt).float(),
            'loc_gt': torch.from_numpy(loc_gt).float(),
            'tid_gt': torch.from_numpy(tid).int(),
            'ignore': torch.from_numpy(ignore).int(),
            'n_box': n_box,
            'endvid': endvid,
            'cam_calib': torch.from_numpy(cam_calib).float(),
            'cam_rot': torch.from_numpy(pose.rotation).float(),
            'cam_loc': torch.from_numpy(pose.position).float(),
        }

        return torch.from_numpy(img_patch).float(), box_info
