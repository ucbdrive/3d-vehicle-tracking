import cv2
import numpy as np

import torch

import utils.tracking_utils as tu
from utils.config import cfg
from model.motion_model import LSTMKF, LSTM
from model.tracker_model import KalmanBox3dTracker, LSTM3dTracker, LSTMKF3dTracker
np.set_printoptions(precision=3, suppress=True)

class Tracker3D:

    def __init__(self,
                 dataset,
                 max_depth=150,
                 max_age=1,
                 min_hits=3,
                 affinity_threshold=0.3,
                 use_occ=False,
                 deep_sort=False,
                 kf3d=False,
                 lstm3d=False,
                 lstmkf3d=False,
                 device='cuda',
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
        self.feat_dim = 128 + 7  # feature dim = 128 , add 5 (depth, roty...)
        self.occ_min_depth = 0.15
        self.occ_max_depth = max_depth
        self.occ_iou_thresh = 0.7
        self.det_thresh = 0.9

        DATASET = cfg.GTA if dataset == 'gta' else cfg.KITTI
        self.H = DATASET.H
        self.W = DATASET.W
        self.FOCAL = DATASET.FOCAL_LENGTH

        self.kf3d = kf3d
        self.use_occ = use_occ
        self.deep_sort = deep_sort
        self.lstm3d = lstm3d
        self.lstmkf3d = lstmkf3d
        self.device = device
        self.verbose = verbose
        self.visualize = visualize
        if self.lstm3d:
            MODEL = LSTM
            ckpt_name = './checkpoint/803_kitti_300_linear.pth'
        elif self.lstmkf3d:
            MODEL = LSTMKF
            ckpt_name = './checkpoint/723_linear.pth'

        if self.lstm3d or self.lstmkf3d:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.lstm = MODEL(1, 64, 128, 2, 3).to(device)
            self.lstm.eval()
            ckpt = torch.load(ckpt_name)
            if verbose: print(ckpt_name, ckpt['epoch'])
            self.lstm.load_state_dict(ckpt['state_dict'])
            if verbose: print(
                "=> Successfully loaded checkpoint {}".format(ckpt_name))
            del ckpt

        self.coord_3d_affinity_weight = 0.7 if deep_sort else 1.0
        self.feat_affinity_weight = 1.0 - self.coord_3d_affinity_weight

        if verbose:
            print("3D Projected IOU weight: {}, Feat weight: {}".format(
                self.coord_3d_affinity_weight,
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
                _center=self.dataset=='kitti'
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

        # Prediction
        # get predicted locations from existing trackers.
        trk_locs = np.zeros((len(self.trackers), 3))
        trk_dets = np.zeros((len(self.trackers), 5))
        trk_dims = np.zeros((len(self.trackers), 3))
        trk_rots = np.zeros((len(self.trackers), 1))
        trk_feats = np.zeros((len(self.trackers), self.feat_dim))
        for t in range(len(trk_locs)):
            trk_locs[t] = self.trackers[t].predict().squeeze()
            trk_dets[t] = self.trackers[t].det
            trk_dims[t] = self.trackers[t].dim
            trk_rots[t] = self.trackers[t].rot
            trk_feats[t] = self.trackers[t].feat
            trk_cen = tu.projection3d(self.cam_calib, self.cam_pose, trk_locs[t:t+1])
            self.trackers[t].cen = trk_cen.squeeze()

        # Generate 2D boxes from 3D estimated location
        trkboxes, trkdepths, trkpoints = tu.construct2dlayout(trk_locs, trk_dims, trk_rots,
                                             self.cam_calib,
                                             self.cam_pose)
        detboxes, detdepths, detpoints = tu.construct2dlayout(world_coords, dims, roty,
                                             self.cam_calib,
                                             self.cam_pose)

        # Association
        idxes_order = np.argsort(trkdepths)
        boxes_order = []
        for idx in idxes_order:
            if self.use_occ:
                # Check if trk box has occluded by others
                if boxes_order != []:
                    # Sort boxes
                    box = trkboxes[idx]
                    ious = []
                    for bo in boxes_order:
                        ious.append(tu.compute_iou(bo, box))
                    # Check if occluded
                    self.trackers[idx].occ = (max(ious) > self.occ_iou_thresh)
            boxes_order.append(trkboxes[idx])

        trk_depths_order = np.array(trkdepths)[idxes_order]
        trk_feats_order = trk_feats[idxes_order]
        trk_dim_order = trk_dims[idxes_order]

        coord_affinity = np.zeros((len(detboxes), len(boxes_order)),
                             dtype=np.float32)
        feat_affinity = np.zeros((len(detboxes), len(boxes_order)),
                             dtype=np.float32)

        if self.use_occ:
            for d, det in enumerate(detboxes):
                if len(boxes_order) != 0:
                    coord_affinity[d, :] = \
                        tu.compute_boxoverlap_with_depth(
                            dets[d],
                            [det[0], det[1], det[2], det[3], 1.0],
                            detdepths[d],
                            dims[d],
                            trk_dets[idxes_order],
                            boxes_order,
                            trk_depths_order,
                            trk_dim_order,
                            H=self.H,
                            W=self.W)
        else:
            for d, det in enumerate(detboxes):
                for t, trk in enumerate(boxes_order):
                    coord_affinity[d, t] += tu.compute_iou(trk, det[:4])

        # Filter out those are not overlaped at all
        location_mask = (coord_affinity>0)

        if self.deep_sort and len(detboxes) * len(boxes_order) > 0:
            feat_affinity += location_mask * \
                             tu.compute_cos_dis(feats, trk_feats_order)

        self.affinity = self.coord_3d_affinity_weight * coord_affinity + \
                        self.feat_affinity_weight * feat_affinity

        # Assignment
        matched, unmatched_dets, unmatched_trks = \
            tu.associate_detections_to_trackers(
                detboxes, boxes_order, self.affinity,
                self.affinity_threshold)

        # update matched trackers with assigned detections
        for t, trkidx in enumerate(idxes_order):
            if t in unmatched_trks:
                self.trackers[trkidx].lost = True
                self.trackers[trkidx].aff_value *= 0.9
                continue

            d = matched[np.where(matched[:, 1] == t)[0], 0]
            if self.kf3d:
                self.trackers[trkidx].update(world_coords[d[0]])
            elif self.lstm3d or self.lstmkf3d:
                self.trackers[trkidx].update(world_coords[d[0]])
            self.trackers[trkidx].lost = False
            self.trackers[trkidx].aff_value = self.affinity[d, t].item()
            self.trackers[trkidx].det = dets[d, :][0]
            self.trackers[trkidx].trk_box = boxes_order[t]
            feat_alpha = 1 - feat_affinity[d, t].item()
            self.trackers[trkidx].feat += feat_alpha * (self.current_frame['feat'][d[0]] - self.trackers[trkidx].feat)
            self.trackers[trkidx].dim = self.current_frame['dim'][d[0]]
            self.trackers[trkidx].alpha = self.current_frame['alpha'][d[0]]
            self.trackers[trkidx].depth = self.current_frame['depth'][d[0]]
            self.trackers[trkidx].cen = self.current_frame['center'][d[0]]
            self.trackers[trkidx].rot = self.current_frame['roty'][d[0]]

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            if self.kf3d:
                trk = KalmanBox3dTracker(world_coords[i])
            elif self.lstm3d:
                trk = LSTM3dTracker(self.device,
                                    self.lstm,
                                    world_coords[i])
            elif self.lstmkf3d:
                trk = LSTMKF3dTracker(self.device,
                                    self.lstm,
                                    world_coords[i])
            trk.det = dets[i, :]
            trk.trk_box = detboxes[i]
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
            _h, _w, _ = img.shape
            img = cv2.putText(img, str(self.frame_count), (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1,
                        (200, 200, 200), 2)

            lost_color = (0, 150, 150)
            occ_color = (150, 0, 150)
            trk_color = (0, 255, 0)
            det_color = (255, 0, 0)
            gt_color = (0, 0, 255)
            cb = 5
            for idx, (box, po, trk) in enumerate(zip(trkboxes, trkpoints, self.trackers)):
                box_color = lost_color if trk.lost else trk_color
                box_color = occ_color if trk.occ else box_color
                box_bold = 2 if trk.lost or trk.occ else 4
                box = box.astype('int')
                print(trk.id+1, 
                    'Lost' if trk.lost else 'Tracked', 
                    '{:2d}'.format(trk.time_since_update),
                    '{:.02f} {:.02f}'.format(trk.aff_value, trkdepths[idx]), 
                    trk.get_history()[-1].flatten(), 
                    trk.get_state()
                    )
                if trkdepths[idx] < 0 or trkdepths[idx] > self.max_depth:
                    continue
                '''
                for (ii,jj) in po:
                    img = cv2.line(img, (int(ii[0]), int(ii[1])),
                            (int(jj[0]), int(jj[1])), box_color, box_bold)
                #'''
                img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), box_color, box_bold-1)
                img = cv2.rectangle(img, (int(trk.cen[0])-cb, int(trk.cen[1])-cb), 
                                        (int(trk.cen[0])+cb, int(trk.cen[1])+cb), box_color, box_bold)

                img = cv2.putText(img, '{}'.format(trk.id+1), (int(box[0]), int(box[1])+20),
                        cv2.FONT_HERSHEY_COMPLEX, 1, box_color, box_bold)
                img = cv2.putText(img, '{:.02f}'.format(trk.aff_value), (int(box[0]-14), int(box[3])+20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, box_color, box_bold)
                img = cv2.putText(img, str(int(trkdepths[idx])), (int(box[2])-14, int(box[3])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, box_color, box_bold)

            if len(data['alpha_gt']) > 0:
                valid_rots = np.zeros_like(data['alpha_gt'])[:, np.newaxis]
                for idx, (alpha, det, center) in enumerate(
                                zip(data['alpha_gt'], data['rois_gt'], data['center_gt'])):
                    valid_rots[idx] = tu.deg2rad(tu.alpha2rot_y(
                                            alpha, 
                                            center[0] - self.W//2, #cam_calib[0][2],
                                            FOCAL_LENGTH=self.cam_calib[0][0]))
                loc_gt = tu.point3dcoord(data['center_gt'], data['depth_gt'], self.cam_calib, self.cam_pose)

                if self.dataset == 'kitti':
                    loc_gt[:, 2] += data['dim_gt'][:, 0] / 2 
                bbgt, depgt, ptsgt = tu.construct2dlayout(loc_gt, data['dim_gt'], valid_rots,
                                                 self.cam_calib,
                                                 self.cam_pose)
                for idx, (tid, boxgt, cengt) in enumerate(zip(data['tid_gt'], data['rois_gt'], data['center_gt'])):
                    detgt = boxgt.astype('int')
                    cengt = cengt.astype('int')
                    img = cv2.rectangle(img, (detgt[0], detgt[1]), (detgt[2], detgt[3]), gt_color, 2)
                    img = cv2.rectangle(img, (cengt[0]-cb, cengt[1]-cb), (cengt[0]+cb, cengt[1]+cb), gt_color, 4)

                    '''
                    for (ii,jj) in ptsgt[idx]:
                        img = cv2.line(img, (int(ii[0]), int(ii[1])),
                                (int(jj[0]), int(jj[1])), gt_color, 2)
                    #'''

            for idx, (det, detbox, detpo, cen) in enumerate(zip(dets, detboxes, detpoints, cens)):
                det = det.astype('int')
                detbox = detbox.astype('int')
                cen = cen.astype('int')
                '''
                for (ii,jj) in detpo:
                    img = cv2.line(img, (int(ii[0]), int(ii[1])),
                            (int(jj[0]), int(jj[1])), det_color, 2)
                #'''
                #img = cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), det_color, 2)
                img = cv2.rectangle(img, (detbox[0], detbox[1]), (detbox[2], detbox[3]), det_color, 2)
                img = cv2.rectangle(img, (cen[0]-cb, cen[1]-cb), (cen[0]+cb, cen[1]+cb), det_color, 4)
                img = cv2.putText(img, str(int(detdepths[idx])), (int(detbox[2])-14, int(detbox[3])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, det_color, 4)

            key = 0
            while(key not in [ord('q'), ord(' '), 27]):
                _f = 0.5 if _h > 600 else 1.0
                cv2.imshow('preview', cv2.resize(img, (0, 0), fx=_f, fy=_f))
                key = cv2.waitKey(1)

            if key == 27:
                cv2.destroyAllWindows()
                exit()


        # Get output returns and remove dead tracklet
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if self.kf3d:
                dep_ = tu.worldtocamera(trk.kf.x[:3].T,
                                        self.cam_pose)[0, 2]
            elif self.lstm3d or self.lstmkf3d:
                dep_ = tu.worldtocamera(trk.x[:3][np.newaxis],
                                        self.cam_pose)[0, 2]
            if (trk.time_since_update < 1) and not trk.occ and (
                    trk.hit_streak >= self.min_hits or self.frame_count <=
                    self.min_hits):
                # +1 as MOT benchmark requires positive
                height = float(trk.det[3] - trk.det[1])
                width = float(trk.det[2] - trk.det[0])
                hypo = {'height': height,
                        'width': width,
                        'trk_box': trk.trk_box.tolist(),
                        'det_box': trk.det.tolist(),
                        'id': trk.id + 1,
                        'x': int(trk.cen[0]),
                        'y': int(trk.cen[1]),
                        'dim': trk.dim.tolist(),
                        'alpha': trk.alpha.item(),
                        'roty': trk.rot.item(),
                        'depth': float(dep_) 
                        }
                ret.append(hypo)
            i -= 1
            # remove dead tracklet
            if dep_ <= self.occ_min_depth or dep_ >= self.occ_max_depth or \
                    (trk.time_since_update > self.max_age and not trk.occ):
                self.trackers.pop(i)

        return ret


    # UTILS
    def process_info(self, data, det_thresh=0.3, match_thres=0.5, max_depth=150,
                     _nms=False, _valid=False, _center=False):
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
        num_boxes = len(gt_boxes)
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
            valid_rots[idx] = tu.deg2rad(tu.alpha2rot_y(
                                    alpha, 
                                    center[0] - self.W//2, #cam_calib[0][2],
                                    FOCAL_LENGTH=self.cam_calib[0][0]))

        # NOTE: pre-processing for RRC detection
        if len(valid_bbox) > 0 and _center:
            _hor = (valid_bbox[:, 2] + 10 < self.W) | (valid_bbox[:, 0] > 10)
            _ver = valid_bbox[:, 3] + 10 < self.H
            valid_cen[_hor, 0] = (valid_bbox[_hor, 0] + valid_bbox[_hor, 2]) / 2
            valid_cen[_ver, 1] = (valid_bbox[_ver, 1] + valid_bbox[_ver, 3]) / 2

        # now build the world 3d coordinate
        valid_worldcoords = tu.point3dcoord(valid_cen, valid_depth, self.cam_calib, self.cam_pose)

        # NOTE: Adding information to feature comparison
        if len(valid_feat):
            valid_feat = np.hstack([valid_feat, 
                                    valid_dim,
                                    valid_cen / np.array([[self.W, self.H]]), 
                                    valid_rots, 
                                    valid_depth / 10])

        return valid_bbox, valid_feat, valid_dim, valid_alpha, valid_depth, \
               valid_cen, valid_rots, valid_worldcoords

