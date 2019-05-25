# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------


import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid

import time
import math
from glob import glob

from datasets.imdb import imdb
import datasets.ds_utils as ds
from model.utils.config import cfg


class gta_det(imdb):
    def __init__(self, image_set, data_path):  # image_set: train/test
        if cfg.BINARY_CLASS:
            self.CLASS_PARSE_DICT = {
                'Compacts': 'foreground',
                'Sedans': 'foreground',
                'SUVs': 'foreground',
                'Coupes': 'foreground',
                'Muscle': 'foreground',
                'Sports Classics': 'foreground',
                'Sports': 'foreground',
                'Super': 'foreground',
                # 8: 'Motorcycles',
                'Off-road': 'foreground',
                'Industrial': 'foreground',
                'Utility': 'foreground',
                'Vans': 'foreground',
                'Service': 'foreground',  # usually taxi
                'Emergency': 'foreground',  # usually police car
                'Military': 'foreground',
                'Commercial': 'foreground'
            }
        else:
            self.CLASS_PARSE_DICT = {
                'Compacts': 'Compacts',
                'Sedans': 'Sedans',
                'SUVs': 'SUVs',
                'Coupes': 'Coupes',
                'Muscle': 'Muscle',
                'Sports Classics': 'Sports Classics',
                'Sports': 'Sports',
                'Super': 'Other',
                # 8: 'Motorcycles',
                'Off-road': 'Off-road',
                'Industrial': 'Other',
                'Utility': 'Other',  # usally truck
                'Vans': 'Vans',
                'Service': 'Service',  # usually taxi
                'Emergency': 'Other',  # usually police car
                'Military': 'Military',
                'Commercial': 'Commercial'
            }

        self.LEGAL_CLASSES = tuple(self.CLASS_PARSE_DICT.keys())
        self.LEGAL_PARSED_CLASSES = tuple(set(self.CLASS_PARSE_DICT.values()))

        self.HASH_TO_CLASS = {-2137348917: 4, -2124201592: 1, -2095439403: 1,
                              -2076478498: 1, -2072933068: 2,
                              -2045594037: 1, -1995326987: 1, -1934452204: 3,
                              -1903012613: 1, -1894894188: 3,
                              -1883869285: 3, -1883002148: 1, -1809822327: 3,
                              -1800170043: 1, -1775728740: 1,
                              -1743316013: 1, -1705304628: 4, -1700801569: 0,
                              -1696146015: 3, -1685021548: 1,
                              -1683328900: 1, -1661854193: 3, -1651067813: 1,
                              -1622444098: 3, -1543762099: 1,
                              -1477580979: 1, -1461482751: 3, -1450650718: 3,
                              -1403128555: 1, -1346687836: 1,
                              -1323100960: 0, -1311240698: 3, -1297672541: 3,
                              -1289722222: 1, -1255452397: 1,
                              -1207771834: 1, -1205801634: 1, -1193103848: 1,
                              -1189015600: 0, -1177863319: 3,
                              -1150599089: 1, -1137532101: 1, -1130810103: 3,
                              -1122289213: 1, -1098802077: 0,
                              -1089039904: 1, -1045541610: 3, -1041692462: 1,
                              -956048545: 1, -947761570: 0,
                              -825837129: 1, -810318068: 1, -808831384: 1,
                              -808457413: 1, -789894171: 1, -784816453: 0,
                              -746882698: 1, -713569950: 2, -685276541: 1,
                              -682211828: 1, -624529134: 1, -599568815: 1,
                              -591610296: 3, -537896628: 3, -511601230: 1,
                              -442313018: 1, -431692672: 3, -394074634: 1,
                              -391594584: 1, -377465520: 3, -344943009: 3,
                              -326143852: 1, -310465116: 1, -304802106: 1,
                              -227741703: 1, -142942670: 3, -120287622: 1,
                              -119658072: 1, -89291282: 1, -16948145: 1,
                              -14495224: 1, -5153954: 1, 48339065: 0,
                              65402552: 1, 75131841: 1, 80636076: 1,
                              92612664: 3, 108773431: 3, 117401876: 1,
                              142944341: 1, 384071873: 3, 408192225: 1,
                              418536135: 3, 464687292: 1, 469291905: 1,
                              475220373: 4, 486987393: 3, 499169875: 3,
                              516990260: 0, 523724515: 1, 569305213: 4,
                              627094268: 1, 699456151: 3, 723973206: 1,
                              767087018: 1, 841808271: 3, 850565707: 1,
                              850991848: 0, 873639469: 3, 884422927: 1,
                              886934177: 1, 887537515: 0, 904750859: 0,
                              914654722: 3, 970598228: 3, 989381445: 0,
                              1011753235: 3, 1032823388: 3, 1039032026: 3,
                              1069929536: 1, 1078682497: 1, 1123216662: 1,
                              1126868326: 3, 1147287684: 3, 1162065741: 1,
                              1171614426: 0, 1177543287: 1, 1221512915: 1,
                              1269098716: 1, 1283517198: 2, 1337041428: 1,
                              1348744438: 1, 1349725314: 3, 1353720154: 4,
                              1373123368: 1, 1491375716: 3, 1507916787: 1,
                              1518533038: 0, 1531094468: 1, 1645267888: 1,
                              1723137093: 1, 1737773231: 3, 1739845664: 1,
                              1747439474: 0, 1762279763: 1, 1770332643: 1,
                              1777363799: 1, 1830407356: 1, 1876516712: 0,
                              1909141499: 1, 1912215274: 1, 1917016601: 4,
                              1923400478: 1, 1938952078: 4, 1951180813: 0,
                              2006918058: 1, 2016857647: 3, 2046537925: 1,
                              2053223216: 4, 2072687711: 3, 2112052861: 2,
                              2132890591: 1, 2136773105: 1}
        # for skip=30
        self.HASH_PARSE_DICT = {914654722: 67, -1311240698: 76,
                                -1627000575: 104, -14495224: 66, -682211828: 26,
                                1951180813: 10,
                                -1934452204: 40, -1193103848: 73,
                                1917016601: 125, 1739845664: 135, 873639469: 36,
                                2016857647: 30,
                                -142942670: 8, -1205801634: 150, 108773431: 17,
                                -344943009: 69, 1069929536: 114,
                                1723137093: 70,
                                -591610296: 110, 850565707: 45, -1150599089: 81,
                                -599568815: 138, 499169875: 64,
                                142944341: 106,
                                723973206: 99, -1651067813: 12, 1032823388: 19,
                                -947761570: 143, 2072687711: 33,
                                -808831384: 16,
                                1830407356: 5, 741586030: 144, 1011753235: 149,
                                -233098306: 53, 1177543287: 55,
                                1337041428: 39,
                                -120287622: 147, -1683328900: 133,
                                2112052861: 27, -431692672: 50, 486987393: 0,
                                1349725314: 7,
                                -685276541: 29, 1147287684: 124, -1045541610: 9,
                                -1289722222: 112, -1685021548: 6,
                                699456151: 56,
                                886934177: 68, -1297672541: 82,
                                -1346687836: 129, 80636076: 97, 1737773231: 79,
                                65402552: 105,
                                -784816453: 108, 464687292: 3, 736902334: 127,
                                384071873: 47, 2046537925: 113,
                                -304802106: 90,
                                418536135: 11, -2137348917: 117,
                                -1041692462: 34, 1221512915: 118, 48339065: 123,
                                1039032026: 95,
                                1269098716: 42, 408192225: 148, 850991848: 22,
                                1912215274: 52, 569305213: 121,
                                1348744438: 1,
                                -1903012613: 23, -1461482751: 41,
                                -119658072: 136, -1130810103: 38, 904750859: 2,
                                -810318068: 137,
                                884422927: 78, -624529134: 48, -1255452397: 43,
                                -1403128555: 145, 1123216662: 18,
                                1171614426: 128,
                                1923400478: 74, 1373123368: 77,
                                -1883002148: 109, 2053223216: 57,
                                516990260: 139,
                                -2072933068: 154,
                                -808457413: 84, 989381445: 131,
                                -1207771834: 132, 1162065741: 51,
                                -377465520: 65,
                                1762279763: 93,
                                970598228: 59, 1777363799: 89, 1353720154: 142,
                                -5153954: 44, -956048545: 14,
                                -713569950: 32,
                                -1137532101: 71, 1126868326: 86,
                                1876516712: 111, -1177863319: 13,
                                887537515: 126,
                                1747439474: 116,
                                -746882698: 49, -1987130134: 152,
                                1078682497: 25, -1543762099: 120,
                                -2124201592: 91,
                                -1809822327: 103, 1938952078: 153,
                                841808271: 75, -1894894188: 80, 475220373: 61,
                                -1883869285: 92,
                                -1775728740: 83, 1886712733: 140,
                                -1696146015: 4, -1450650718: 63, -310465116: 85,
                                -391594584: 54,
                                2006918058: 58, 1518533038: 134, 1645267888: 98,
                                -511601230: 72, -825837129: 96,
                                -1622444098: 60,
                                75131841: 62, -1122289213: 35, 1531094468: 88,
                                -1800170043: 102, 1941029835: 115,
                                -1705304628: 146,
                                -1477580979: 94, -1189015600: 151,
                                -1743316013: 21, -2095439403: 100,
                                -16948145: 119,
                                -2076478498: 130, -1700801569: 122,
                                -1089039904: 37, 1770332643: 141,
                                -789894171: 28,
                                -89291282: 46,
                                2136773105: 20, 1507916787: 101,
                                -1995326987: 31, -394074634: 87, -227741703: 24,
                                1909141499: 107,
                                767087018: 15}
        # for skip=5
        # self.HASH_PARSE_DICT = {914654722: 75, -1311240698: 83, 
        # -1627000575: 107, -14495224: 74, -682211828: 36, 
        # 1951180813: 22,
        #                    -1934452204: 48, -1193103848: 80, 1917016601: 
        #                    127, 1739845664: 136, 873639469: 45, 
        #                    2016857647: 40,
        #                    -142942670: 8, -1205801634: 150, 108773431: 26, 
        #                    -344943009: 76, 1069929536: 118, 
        #                    1723137093: 81,
        #                    -1477580979: 99, -591610296: 114, 850565707: 55,
        #                    -16948145: 18, -599568815: 17, 
        #                    499169875: 73,
        #                    142944341: 109, 723973206: 104, -1651067813: 13,
        #                    1032823388: 28, -947761570: 144, 
        #                    2072687711: 43,
        #                    1491375716: 145, -808831384: 25, 1830407356: 6, 
        #                    741586030: 147, 1011753235: 143, 
        #                    -233098306: 62,
        #                    1177543287: 64, 1337041428: 49, -120287622: 149,
        #                    -1683328900: 134, 2112052861: 10, 
        #                    -431692672: 60,
        #                    486987393: 1, 1349725314: 7, -685276541: 39, 
        #                    -1045541610: 9, -1289722222: 116, 
        #                    -1685021548: 4,
        #                    699456151: 65, 886934177: 77, -1297672541: 111, 
        #                    -1346687836: 121, 80636076: 102, 
        #                    1737773231: 86,
        #                    65402552: 108, -784816453: 112, 464687292: 3, 
        #                    736902334: 30, 384071873: 59, 2046537925: 
        #                    117,
        #                    -304802106: 97, 418536135: 21, -2137348917: 122,
        #                    -1041692462: 88, 1221512915: 14, 
        #                    48339065: 125,
        #                    1039032026: 101, 1269098716: 51, 850991848: 33, 
        #                    1912215274: 61, 569305213: 123, 
        #                    1348744438: 0,
        #                    -1903012613: 34, -1461482751: 50, -119658072: 
        #                    137, -1130810103: 47, 904750859: 2, 
        #                    -810318068: 138,
        #                    884422927: 85, -624529134: 58, -1255452397: 53, 
        #                    1123216662: 27, 1171614426: 129, 
        #                    1923400478: 82,
        #                    1373123368: 84, -1883002148: 113, 2053223216: 
        #                    19, 516990260: 139, -2072933068: 152, 
        #                    -1137532101: 78,
        #                    989381445: 130, -1207771834: 133, 1162065741: 
        #                    56, -377465520: 72, 1762279763: 100, 
        #                    970598228: 69,
        #                    1777363799: 96, 1353720154: 142, -5153954: 54, 
        #                    -956048545: 23, -713569950: 42, 
        #                    -808457413: 91,
        #                    1126868326: 93, 1876516712: 115, -1177863319: 
        #                    16, 887537515: 128, 1747439474: 120, 
        #                    -746882698: 52,
        #                    -1987130134: 151, 1078682497: 35, -2124201592: 
        #                    98, -1809822327: 106, 1938952078: 146, 
        #                    841808271: 37,
        #                    -1894894188: 87, 475220373: 70, -1883869285: 12,
        #                    -1775728740: 90, 1886712733: 140, 
        #                    -1696146015: 5,
        #                    -1450650718: 71, -310465116: 92, -391594584: 63,
        #                    2006918058: 67, 1518533038: 135, 
        #                    1645267888: 103,
        #                    -511601230: 79, -825837129: 15, -1622444098: 68,
        #                    75131841: 20, -1122289213: 44, 
        #                    1531094468: 95,
        #                    -1800170043: 126, 1941029835: 119, -1705304628: 
        #                    148, -1543762099: 11, -1189015600: 132,
        #                    -1743316013: 32, -2095439403: 105, -1150599089: 
        #                    89, -2076478498: 131, -1700801569: 124,
        #                    -1089039904: 46, 1770332643: 141, -789894171: 
        #                    38, -89291282: 57, 2136773105: 31, 
        #                    1507916787: 66,
        #                    -1995326987: 41, -394074634: 94, -227741703: 29,
        #                    1909141499: 110, 767087018: 24}

        assert image_set in ['train', 'val', 'test']
        imdb.__init__(self, 'gta_det_' + image_set)
        # name, paths
        self._image_set = image_set
        self._data_path = data_path

        self._classes = ('__background__',) + self.LEGAL_PARSED_CLASSES

        self._class_to_ind = dict(
            list(zip(self.classes, list(range(self.num_classes)))))
        self._read_dataset()
        self._data_path = self._get_ann_file()
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self.set_proposal_method('gt')


    def _get_ann_file(self):
        if cfg.USE_DEBUG_SET:
            return osp.join(self._data_path, 'train')  # train
        else:
            return osp.join(self._data_path, cfg.ANNO_PATH)  # train or val

    def _read_dataset(self):
        ann_file = self._get_ann_file()
        jsonfiles = sorted(glob(osp.join(ann_file, 'label', '*.json')))
        self.dataset = []
        self.endvid = []
        for jf in jsonfiles:
            dataset = [json.load(open(it)) for it in json.load(open(jf))]
            self.dataset += dataset
            endvid = [False] * len(dataset)
            endvid[-1] = True
            self.endvid += endvid

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        return list(range(len(self.dataset)))  # dataset is saved as a list

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return osp.join(self._data_path, 'image', 
                    self.dataset[self._image_index[i]]['name'])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')

        gt_roidb = [self._load_gta_annotation(index)
                    for index in self._image_index]

        return gt_roidb

    def _load_gta_annotation(self, index):
        """
        Loads GTA bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        width = 1920
        height = 1080

        info = self.dataset[self.image_id_at(index)]
        labels = info['labels']  # a list of dict
        # get the kitti part out and insert the tracking id
        boxes = ds.get_box2d_array(labels).astype(float)[:, :4]
        tid = ds.get_label_array(labels, ['id'], (0)).astype(int)
        num_objs = len(tid)
        #gt_classes = ds.get_label_array(labels, ['class'], (0))
        gt_classes = np.array(['foreground']*num_objs)
        # actually just one single value,
        ignore = ds.get_label_array(labels, 
                            ['attributes', 'ignore'], (0)).astype(int)
        cam_calib = np.array(info['intrinsics']['cali'])
        location = ds.get_label_array(labels, ['box3d', 'location'],
                                   (0, 3)).astype(float)
        ext_loc = np.hstack([location, np.ones([len(location), 1])])  # (B, 4)
        proj_loc = ext_loc.dot(cam_calib.T)  # (B, 4) dot (3, 4).T => (B, 3)
        center = proj_loc[:, :2] / proj_loc[:, 2:3]  # normalize

        seg_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
                    (boxes[:, 3] - boxes[:, 1] + 1)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        endvid = np.zeros((num_objs), dtype=np.uint16) 
        # pad to make it consistent
        if self.endvid[self.image_id_at(index)]:
            endvid += 1

        for ix in range(num_objs):
            cls = self._class_to_ind[gt_classes[ix].strip()]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        ds.validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        info_set = {'width': width,
                    'height': height,
                    'boxes': boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps': overlaps,
                    'flipped': False,
                    'seg_areas': seg_areas,
                    'ignore': ignore,
                    'end_vid': endvid,
                    'center': center
                    }
        return info_set


    def _get_widths(self):
        return [r['width'] for r in self.roidb]

    def append_flipped_images(self):
        raise NotImplementedError(
            'The function is not done yet. (Need to process pose transform)')
        num_images = self.num_images
        widths = self._get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'width': widths[i],
                     'height': self.roidb[i]['height'],
                     'boxes': boxes,
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'flipped': True,
                     'seg_areas': self.roidb[i]['seg_areas']}
            # insert the remaining
            for k, v in list(self.roidb[i].items()):
                if k not in list(entry.keys()):
                    entry[k] = v
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

        # TODO: rewrite functions to eval detection
        # def _print_detection_eval_metrics(self, coco_eval):
        #     IoU_lo_thresh = 0.5
        #     IoU_hi_thresh = 0.95
        #
        #     def _get_thr_ind(coco_eval, thr):
        #         ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
        #                        (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        #         iou_thr = coco_eval.params.iouThrs[ind]
        #         assert np.isclose(iou_thr, thr)
        #         return ind
        #
        #     ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        #     ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        #     # precision has dims (iou, recall, cls, area range, max dets)
        #     # area range index 0: all area ranges
        #     # max dets index 2: 100 per image
        #     precision = \
        #         coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        #     ap_default = np.mean(precision[precision > -1])
        #     print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
        #            '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
        #     print('{:.1f}'.format(100 * ap_default))
        #     for cls_ind, cls in enumerate(self.classes):
        #         if cls == '__background__':
        #             continue
        #         # minus 1 because of __background__
        #         precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 
        #         1), :, cls_ind - 1, 0, 2]
        #         ap = np.mean(precision[precision > -1])
        #         print('{:.1f}'.format(100 * ap))
        #
        #     print('~~~~ Summary metrics ~~~~')
        #     coco_eval.summarize()
        #
        # def _do_detection_eval(self, res_file, output_dir):
        #     ann_type = 'bbox'
        #     coco_dt = self._COCO.loadRes(res_file)
        #     coco_eval = COCOeval(self._COCO, coco_dt)
        #     coco_eval.params.useSegm = (ann_type == 'segm')
        #     coco_eval.evaluate()
        #     coco_eval.accumulate()
        #     self._print_detection_eval_metrics(coco_eval)
        #     eval_file = osp.join(output_dir, 'detection_results.pkl')
        #     with open(eval_file, 'wb') as fid:
        #         pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        #     print('Wrote COCO eval results to: {}'.format(eval_file))
        #
        # def _coco_results_one_category(self, boxes, cat_id):
        #     results = []
        #     for im_ind, index in enumerate(self.image_index):
        #         dets = boxes[im_ind].astype(np.float)
        #         if dets == []:
        #             continue
        #         scores = dets[:, -1]
        #         xs = dets[:, 0]
        #         ys = dets[:, 1]
        #         ws = dets[:, 2] - xs + 1
        #         hs = dets[:, 3] - ys + 1
        #         results.extend(
        #             [{'image_id': index,
        #               'category_id': cat_id,
        #               'bbox': [xs[k], ys[k], ws[k], hs[k]],
        #               'score': scores[k]} for k in range(dets.shape[0])])
        #     return results
        #
        # def _write_coco_results_file(self, all_boxes, res_file):
        #     # [{"image_id": 42,
        #     #   "category_id": 18,
        #     #   "bbox": [258.15,41.29,348.26,243.78],
        #     #   "score": 0.236}, ...]
        #     results = []
        #     for cls_ind, cls in enumerate(self.classes):
        #         if cls == '__background__':
        #             continue
        #         print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
        #                                                          self.num_classes - 1))
        #         coco_cat_id = self._class_to_coco_cat_id[cls]
        #         results.extend(self._coco_results_one_category(all_boxes[
        #         cls_ind],
        #                                                        coco_cat_id))
        #     print('Writing results json to {}'.format(res_file))
        #     with open(res_file, 'w') as fid:
        #         json.dump(results, fid)
        #
        # def evaluate_detections(self, all_boxes, output_dir):
        #     res_file = osp.join(output_dir, ('detections_' +
        #                                      self._image_set +
        #                                      self._year +
        #                                      '_results'))
        #     if self.config['use_salt']:
        #         res_file += '_{}'.format(str(uuid.uuid4()))
        #     res_file += '.json'
        #     self._write_coco_results_file(all_boxes, res_file)
        #     # Only do evaluation on non-test sets
        #     if self._image_set.find('test') == -1:
        #         self._do_detection_eval(res_file, output_dir)
        #     # Optionally cleanup results json file
        #     if self.config['cleanup']:
        #         os.remove(res_file)
