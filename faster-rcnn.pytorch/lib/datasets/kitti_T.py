# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Li Bin
# --------------------------------------------------------

import os
import datasets.imdb as imdb
import numpy as np
import cv2
import pickle as cPickle
import uuid
from datasets.imdb import imdb
from glob import glob

from datasets.kitti_eval import kitti_eval


class kitti(imdb):
    def __init__(self, image_set, devkit_path=None, result_path=None,
                 mask_size=-1, binary_thresh=None):
        super(kitti, self).__init__('kitti_' + image_set)  # set self.name
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path

        self._data_path = os.path.join(self._devkit_path, image_set, 'image_02')
        self._classes = ('__background__', 'Car', 'Truck', 'Tram')
        self._class_to_ind = dict(zip(self._classes, range(len(self._classes))))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'
        assert os.path.exists(
            self._devkit_path), 'Kitti path does not exist: {}'.format(
            self._devkit_path)
        assert os.path.exists(
            self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index[0],
                                  index[1] + self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(
            image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        import os
        imglist = sorted(glob(os.path.join(self._data_path, '*', '*.png')))
        to_del = []
        if self._image_set == 'training':
            for a in imglist:
                b = a.replace('image', 'label')
                b = b.replace('png', 'txt')
                if not os.path.exists(b):
                    print(a)
                    to_del.append(a)
            for a in to_del:
                imglist.remove(a)
        imgset = [(s.split('/')[-2], s.split('/')[-1].replace('.png', '')) for s
                  in imglist]
        return imgset

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join('kitti-T')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        gt_roidb = [self._load_kitti_annotation(index) for index in
                    self._image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'testing':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from TXT file in the kitti format.
        """
        if self._image_set == 'testing':
            track, frame = index
            imagename = os.path.join(self._devkit_path, 'testing/image_02',
                                     track, frame + '.png')
            width = 1242
            height = 375
            print(imagename)
            return {'image': imagename,
                    'height': height,
                    'width': width,
                    'flipped': False,
                    'is_train': False,
                    'gt_overlaps': np.zeros((1, 1)),
                    'gt_classes': np.zeros((1, 1))}

        track, frame = index
        filename = os.path.join(self._devkit_path, 'training/label_02', track,
                                frame + '.txt')
        imagename = os.path.join(self._devkit_path, 'training/image_02', track,
                                 frame + '.png')
        img = cv2.imread(imagename)
        print(imagename)
        width = img.shape[0]
        height = img.shape[1]
        f = open(filename)
        lines = f.readlines()
        num_objs = 0
        for l in lines:
            str_cls = l.split()[1:]
            if str(str_cls[1]) in self._classes or str(str_cls[1]) == 'Van':
                num_objs = num_objs + 1
        num_objs = num_objs
        #        print 'num_objs',num_objs
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ix = 0
        for line in lines:
            data = line.split()[1:]
            if str(data[1]) == 'Van':
                data[1] = 'Car'
            if str(data[1]) not in self._classes:
                continue
            x1 = int(float(data[5]))
            y1 = int(float(data[6]))
            x2 = int(float(data[7]))
            y2 = int(float(data[8]))
            cls = self._class_to_ind[data[1]]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix = ix + 1
        #      print "aaa",gt_classes
        #       a
        # overlaps = scipy.sparse.csr_matrix(overlaps)
        # print img.height,img.width
        return {'boxes': boxes,
                'image': imagename,
                'height': height,
                'width': width,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'max_classes': overlaps.argmax(axis=1),
                'max_overlaps': overlaps.max(axis=1),
                'flipped': False,
                'seg_areas': seg_areas,
                'is_train': True}

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(self._devkit_path, 'training/label_2',
                                '{:s}.txt')
        imageset = self._image_index

        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True  # if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)

            # for every class, run a kitti_eval
            rec, prec, ap = kitti_eval(
                filename, annopath, imagesetfile, cls, cachedir,
                ovthresh=0.7,
                use_07_metric=use_07_metric)

            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):

        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)

        # should use clean up funcion!!!

    #        if self.config['matlab_eval']:
    #            self._do_matlab_eval(output_dir)
    #        if self.config['cleanup']:
    #            for cls in self._classes:
    #                if cls == '__background__':
    #                    continue
    #                filename = self._get_voc_results_file_template().format(
    #                cls)
    #                os.remove(filename)

    def _write_voc_results_file(self, all_boxes):
        # print self.classes
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]

                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))


if __name__ == '__main__':
    pass
