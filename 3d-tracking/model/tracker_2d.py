import cv2
import numpy as np

import utils.tracking_utils as tu
from utils.config import cfg
from model.tracker_model import KalmanBoxTracker


class Tracker2D:

    def __init__(self,
                 dataset,
                 max_depth=150,
                 max_age=1,
                 min_hits=3,
                 affinity_threshold=0.3,
                 deep_sort=False,
                 kf2d=False,
                 verbose=False,
                 visualize=False):

        self.dataset = dataset
        self.max_depth = max_depth
        self.max_age = max_age
        self.min_hits = min_hits
        self.affinity_threshold = affinity_threshold
        self.trackers = []
        self.frame_count = 0

        self.current_frame = {}

        self.n_gt = 0
        self.n_FP = 0
        self.n_FN = 0
        self.n_mismatch = 0
        self.det_dim = 5  # [x1, y1, x2, y2, conf]
        self.feat_dim = 128  # feature dim = 128
        self.det_thresh = 0.9

        DATASET = cfg.GTA if dataset == 'gta' else cfg.KITTI
        self.H = DATASET.H
        self.W = DATASET.W
        self.FOCAL = DATASET.FOCAL_LENGTH

        self.kf2d = kf2d
        self.deep_sort = deep_sort
        self.verbose = verbose
        self.visualize = visualize

        self.iou_affinity_weight = 0.7 if deep_sort else 1.0
        self.feat_affinity_weight = 1.0 - self.iou_affinity_weight

        if verbose:
            print("IOU weight: {}, Feat weight: {}".format(
                self.iou_affinity_weight,
                self.feat_affinity_weight))

    def update(self, data):
        # frame information here
        ret = []
        self.frame_count += 1
        self.cam_rot = data['cam_rot'].squeeze() # In rad
        self.cam_coord = data['cam_loc'].squeeze()
        if self.frame_count == 1:
            self.init_coord = self.cam_coord.copy()
        self.cam_coord -= self.init_coord
        self.cam_calib = data['cam_calib'].squeeze()
        self.cam_pose = tu.Pose(self.cam_coord, self.cam_rot)

        # process information
        dets, feats, dims, alphas, single_depths, cens, roty, world_coords = \
            self.process_info(
                data,
                det_thresh=self.det_thresh,
                max_depth=self.max_depth,
                _nms=self.dataset=='gta',
                _valid=self.dataset=='gta',
                )

        # save to current frame
        self.current_frame = {
            'bbox': dets,
            'feat': feats,
            'dim': dims,
            'alpha': alphas, # in deg
            'roty': roty, # in rad
            'depth': single_depths,
            'center': cens,
            'location': world_coords,
            'n_obj': len(alphas),
        }

        # get predicted locations from existing trackers.
        trk_dim = self.feat_dim + self.det_dim
        trks = np.zeros((len(self.trackers), trk_dim))

        # Prediction
        for t, trk in enumerate(trks):
            if self.kf2d:
                pos = self.trackers[t].predict()[0]
            else:
                pos = self.trackers[t].predict_no_effect()[0]
            trk[:self.det_dim] = [pos[0], pos[1], pos[2], pos[3], 1.0]
            trk[self.det_dim:] = self.trackers[t].feat

        # Association
        self.affinity = np.zeros((len(dets), len(trks)), dtype=np.float32)

        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                self.affinity[
                    d, t] += self.iou_affinity_weight * tu.compute_iou(det[:4],
                                                                       trk[:4])
        if self.deep_sort and len(dets) * len(trks) > 0:
            self.affinity += self.feat_affinity_weight * tu.compute_cos_dis(
                feats, trks[:, 5:])

        matched, unmatched_dets, unmatched_trks = \
            tu.associate_detections_to_trackers(
                dets, trks, self.affinity,
                self.affinity_threshold)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t in unmatched_trks:
                trk.lost = True
                continue

            d = matched[np.where(matched[:, 1] == t)[0], 0]
            trk.update(dets[d, :][0])
            trk.det = dets[d, :][0]
            trk.feat = self.current_frame['feat'][d[0]]
            trk.dim = self.current_frame['dim'][d[0]]
            trk.alpha = self.current_frame['alpha'][d[0]]
            trk.depth = self.current_frame['depth'][d[0]]
            trk.cen = self.current_frame['center'][d[0]]
            trk.rot = self.current_frame['roty'][d[0]]

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            trk.det = dets[i, :]
            trk.feat = self.current_frame['feat'][i]
            trk.dim = self.current_frame['dim'][i]
            trk.alpha = self.current_frame['alpha'][i]
            trk.depth = self.current_frame['depth'][i]
            trk.cen = self.current_frame['center'][i]
            trk.rot = self.current_frame['roty'][i]
            self.trackers.append(trk)

        # Check if boxes are correct
        if self.visualize:
            img = cv2.imread(data['im_path'][0])
            h, w, _ = img.shape
            img = cv2.putText(img, str(self.frame_count), (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (200, 200, 200), 2)
            
            for idx, trk in enumerate(self.trackers):
                box_color = (0, 150, 150) if trk.lost else (0, 255, 0)
                box_bold = 2 if trk.lost else 4
                box = trk.get_state()[0].astype('int')
                print(trk.id+1, 
                    'Lost' if trk.lost else 'Tracked', 
                    #trk.aff_value, 
                    trk.depth, 
                    )
                if trk.depth < 0 or trk.depth > self.max_depth:
                    continue
                img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), box_color, box_bold-1)
                img = cv2.putText(img, str(int(trk.id+1)), (int(box[0]), int(box[1])+20),
                        cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 255), 2)
                img = cv2.putText(img, str(int(trk.depth)), (int(box[2])-14, int(box[3])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8,
                        (0, 0, 255), 2)
                xc = int(trk.cen[0])
                yc = int(trk.cen[1])
                img = cv2.rectangle(img, (xc-1, yc-1),(xc+1, yc+1), (0, 0, 255), box_bold)

            for idx, (det, cen) in enumerate(zip(dets, cens)):
                det = det.astype('int')
                cen = cen.astype('int')
                img = cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), (255, 0, 0), 2)
                img = cv2.rectangle(img, (cen[0]-1, cen[1]-1), (cen[0]+1, cen[1]+1), (255, 0, 0), 4)

            key = 0
            while(key not in [ord('q'), ord(' '), 27]):
                cv2.imshow('preview', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
                key = cv2.waitKey(1)

            if key == 27:
                cv2.destroyAllWindows()
                exit()

        # Get output returns and remove dead tracklet
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            trk_box = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                    trk.hit_streak >= self.min_hits or self.frame_count <=
                    self.min_hits):
                # +1 as MOT benchmark requires positive
                height = float(trk.det[3] - trk.det[1])
                width = float(trk.det[2] - trk.det[0])
                hypo = {'height': height,
                        'width': width,
                        'trk_box': trk_box.tolist(),
                        'det_box': trk.det.tolist(),
                        'id': trk.id + 1,
                        'x': int(trk.cen[0]),
                        'y': int(trk.cen[1]),
                        'dim': trk.dim.tolist(),
                        'alpha': trk.alpha.item(),
                        'roty': trk.rot.item(),
                        'depth': trk.depth.item() 
                        }
                ret.append(hypo)
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return ret

    # UTILS
    def process_info(self, data, det_thresh=0.3, match_thres=0.5, max_depth=150,
                     _nms=True, _valid=True):
        """return the valid, matched index of bbox, and corresponding
        tracking ids, and corresponding info"""
        bbox = data['rois_pd']
        if len(bbox) > 0:
            if _nms:
                keep = tu.nms_cpu(bbox, 0.3)
            else:
                keep = np.arange(0, bbox.shape[0])
            bbox = bbox[keep]
            res_feat = data['feature'][keep]
            res_dim = data['dim_pd'][keep]
            res_alpha = data['alpha_pd'][keep]
            res_depth = data['depth_pd'][keep]
            res_cen = data['center_pd'][keep]

        gt_boxes = data['rois_gt']
        dim_gt = data['dim_gt']
        alpha_gt = data['alpha_gt']
        depth_gt = data['depth_gt']
        cen_gt = data['center_gt']
        ignores = data['ignore']
        tracking_ids = data['tid_gt']

        # Ignore if gt is max_depth meters away
        ignores[depth_gt > max_depth] = 1

        # Prune bbox if prediction is max_depth meters away
        if len(bbox) > 0:
            bbox = bbox[res_depth < max_depth]
            res_feat = res_feat[res_depth < max_depth]
            res_dim = res_dim[res_depth < max_depth]
            res_alpha = res_alpha[res_depth < max_depth]
            res_cen = res_cen[res_depth < max_depth]
            res_depth = res_depth[res_depth < max_depth]

        else:
            # print('*** Pure detecton, do not get pose information here!')
            res_feat = np.zeros([bbox.shape[0], self.feat_dim])
            res_dim = np.ones([bbox.shape[0], 3])
            res_alpha = np.ones([bbox.shape[0]])
            res_depth = np.ones([bbox.shape[0]])
            res_cen = np.zeros([bbox.shape[0], 2])

        # (x1, y1, x2, y2, conf)
        if gt_boxes.shape[0] > 0:
            gt_boxes = gt_boxes[:, 0:4]

        self.frame_annotation = tu.build_frame_annotation(
                                    gt_boxes, 
                                    ignores, 
                                    tracking_ids,
                                    dim_gt, 
                                    alpha_gt,   
                                    depth_gt,
                                    cen_gt,
                                    self.cam_calib, 
                                    self.cam_rot,
                                    self.cam_coord)

        gt_boxes_ignored = [gb for i, gb in enumerate(gt_boxes) if ignores[i]]


        if _valid:
            valid_bbox_ind = []
            for i in range(bbox.shape[0]):
                box = bbox[i, :4]
                score = bbox[i, 4]
                if score > det_thresh:  # if valid
                    save = True
                    for bg in gt_boxes_ignored:
                        if tu.compute_iou(box, bg) > match_thres:
                            save = False
                            break
                    if save:
                        valid_bbox_ind.append(i)
        else:
            valid_bbox_ind = np.arange(0, bbox.shape[0])

        valid_bbox = bbox[valid_bbox_ind]
        valid_feat = res_feat[valid_bbox_ind]
        valid_dim = res_dim[valid_bbox_ind]
        valid_alpha = res_alpha[valid_bbox_ind].reshape(-1, 1)
        valid_depth = res_depth[valid_bbox_ind].reshape(-1, 1)
        valid_cen = res_cen[valid_bbox_ind]

        valid_rots = np.zeros_like(valid_alpha)
        for idx, (alpha, det, center) in enumerate(
                        zip(valid_alpha, valid_bbox, valid_cen)):
            valid_rots[idx] = tu.alpha2rot_y(
                                    alpha, 
                                    center[0] - self.W//2,
                                    FOCAL_LENGTH=self.cam_calib[0][0])

        # now build the world 3d coordinate
        valid_worldcoords = tu.point3dcoord(valid_cen, valid_depth, self.cam_calib, self.cam_pose)

        return valid_bbox, valid_feat, valid_dim, valid_alpha, valid_depth, \
               valid_cen, valid_rots, valid_worldcoords

