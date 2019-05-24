import numba
import numpy as np
import os
from numpy.linalg import inv

import cv2
import sklearn.metrics.pairwise as skp
from sklearn.utils.linear_assignment_ import linear_assignment
import torch
import utm
from shapely.geometry import Polygon


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, verbose=False):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            if verbose: print("Avg: {} | Sum: {} | Count: {}".format(
                self.avg, self.sum, self.count))
            self.avg = self.sum / self.count
        else:
            print("Not update! count equals 0")

class Pose:
    ''' Calibration matrices in KITTI
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord: 
        front x, left y, up z

        world coord (GTA):
        right x, front y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    '''
    def __init__(self, position, rotation, is_gta=False):
        # relative position to the 1st frame: (X, Y, Z)
        # relative rotation to the previous frame: (r_x, r_y, r_z)
        self.position = position
        if rotation.shape == (3, 3):
            # rotation matrices already
            self.rotation = rotation
        else:
            # rotation vector 
            self.rotation = angle2rot(rotation)
            if is_gta:
                magic_rot = angle2rot(np.array([np.pi / 2, 0, 0]), inverse=True)
            else:
                magic_rot = angle2rot(np.array([np.pi / 2, -np.pi / 2, 0]), inverse=True)
            self.rotation = self.rotation.dot(magic_rot)

# Functions from kio_slim
class KittiPoseParser:
    def __init__(self, fields=None):
        self.latlon = None
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.position = None
        self.rotation = None
        if fields is not None:
            self.set_oxt(fields)

    def set_oxt(self, fields):
        fields = [float(f) for f in fields]
        self.latlon = fields[:2]
        location = utm.from_latlon(*self.latlon)
        self.position = np.array([location[0], location[1], fields[2]])

        self.roll = fields[3]
        self.pitch = fields[4]
        self.yaw = fields[5]
        rotation = angle2rot(np.array([self.roll, self.pitch, self.yaw]))
        magic_rot = angle2rot(np.array([np.pi / 2, -np.pi / 2, 0]), inverse=True)
        self.rotation = rotation.dot(magic_rot.T)

# Same as angle2rot from kio_slim
def angle2rot(rotation, inverse=False):
    return rotate(np.eye(3), rotation, inverse=inverse)

def read_oxts(oxts_dir, seq_idx):
    oxts_path = os.path.join(oxts_dir, '{:04d}.txt'.format(seq_idx))
    fields = [line.strip().split() for line in open(oxts_path, 'r')]
    return fields

def read_calib(seq_idx, calib_dir='cali', cam=2):
    fields = [line.split() for line in
              open(os.path.join(calib_dir, '{:04d}.txt'.format(seq_idx)))]
    return np.asarray(fields[cam][1:], dtype=np.float32).reshape(3, 4)

def point3dcoord_torch(point, depth, projection, position, rotation):
    corners = projection[:, 0:3].inverse().mm(
        torch.cat([point, point.new_ones((1, point.shape[1]))]))
    assert abs(corners[2, 0] - 1) < 0.01
    corners_global = rotation.mm(corners * depth).view(3, -1) + position
    return corners_global


def point3dcoord(points, depths, projection, pose):
    """
    project point to 3D world coordinate

    point: (N, 2), N points on X-Y image plane
    projection: (3, 4), projection matrix
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    assert points.shape[1] == 2, (
        "Shape ({}) not fit".format(points.shape))
    corners = imagetocamera(points, depths, projection)
    corners_global = cameratoworld(corners, pose)
    return corners_global


def boxto3dcoord(box, depth, projection, pose):
    """
    project a box center to 3D world coordinate

    box: (5,), N boxes on X-Y image plane
    projection: (3, 4), projection matrix
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    corners_global: (3, 1), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    x1, y1, x2, y2 = box[:4]
    points = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2]])
    return point3dcoord(points, depth, projection, pose)

def projection3d(projection, pose, corners_global):
    """
    project 3D point in world coordinate to 2D image plane

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    projection: (3, 4), projection matrix
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    point: (N, 2), N points on X-Y image plane
    """
    corners = worldtocamera(corners_global, pose)
    corners = cameratoimage(corners, projection)
    return corners 

def cameratoimage(corners, projection, invalid_value=-1000):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera plane
    projection: (3, 4), projection matrix

    points: (N, 2), N points on X-Y image plane
    """
    assert corners.shape[1] == 3, "Shape ({}) not fit".format(corners.shape)

    points = np.hstack([corners, np.ones((corners.shape[0], 1))]).dot(
                projection.T)

    # [x, y, z] -> [x/z, y/z]
    mask = points[:, 2:3] > 0
    points = (points[:, :2] / points[:, 2:3]) * mask + invalid_value * (1-mask)

    return points

def imagetocamera(points, depth, projection):
    """
    points: (N, 2), N points on X-Y image plane
    depths: (N,), N depth values for points
    projection: (3, 4), projection matrix

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    assert points.shape[1] == 2, "Shape ({}) not fit".format(points.shape)

    corners = np.hstack([points, np.ones((points.shape[0], 1))]).dot(
                inv(projection[:, 0:3]).T)
    assert np.allclose(corners[:, 2], 1)
    corners *= depth.reshape(-1, 1)

    return corners

def worldtocamera(corners_global, pose):
    """
    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3,), translation of world coordinates

    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    """
    assert corners_global.shape[1] == 3, (
        "Shape ({}) not fit".format(corners_global.shape))
    corners = (corners_global - pose.position[np.newaxis]).dot(pose.rotation)
    return corners

def cameratoworld(corners, pose):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3), translation of world coordinates

    corners_global: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    """
    assert corners.shape[1] == 3, (
        "Shape ({}) not fit".format(corners.shape))
    corners_global = corners.dot(pose.rotation.T) + \
                     pose.position[np.newaxis]
    return corners_global


def compute_boxoverlap_with_depth(det, detbox, detdepth, detdim,
                                  trkdet, trkedboxes, trkeddepths, trkdims,
                                  H=1080, W=1920):
    iou_2d = np.zeros((len(trkeddepths)))
    
    # Sum up all the available region of each tracker
    for i in range(len(trkedboxes)):
        iou_2d[i] = compute_iou(trkedboxes[i], detbox)
    depth_weight = np.exp(-abs(trkeddepths - detdepth)/(detdepth + 1e-6))

    #print(iou_2d, depth_weight, iou_2d*depth_weight)
    # Calculate the IOU
    iou_2d *= depth_weight
    # Check if depth difference is within the diagonal distance of two cars
    iou_2d *= (detdim[2] + detdim[1] + trkdims[:, 2] + trkdims[:, 1]) > \
                abs(detdepth - trkeddepths)
    return iou_2d

def compute_boxoverlap_with_depth_draw(det, detbox, detdepth, detdim,
                                  trkdet, trkedboxes, trkeddepths, trkdims,
                                  H=1080, W=1920):
    overlap = np.zeros((len(trkeddepths)))
    valid_trkbox = np.zeros((len(trkeddepths)))
    iou_2d = np.zeros((len(trkeddepths)))
    same_layer = np.zeros((len(trkeddepths)))
    # Find where the DOI is in the tracking depth order
    idx = np.searchsorted(trkeddepths, detdepth)

    boxesbefore = trkedboxes[:idx]
    boxesafter = trkedboxes[idx:]
    before = np.zeros((H, W), dtype=np.uint8)
    after = np.zeros((H, W), dtype=np.uint8)
    now = np.zeros((H, W), dtype=np.uint8)
    # Plot 2D bounding box according to the depth order
    for idx, box in enumerate(boxesbefore):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)
        before = cv2.rectangle(before, (x1, y1), (x2, y2), idx + 1, -1)
    for idx, box in enumerate(reversed(boxesafter)):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)
        after = cv2.rectangle(after, (x1, y1), (x2, y2),
                              len(trkedboxes) - idx, -1)

    x1, y1, x2, y2, _ = detbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    now = cv2.rectangle(now, (x1, y1), (x2, y2), 1, -1)

    currentpixels = np.where(now == 1)
    pixelsbefore = before[currentpixels]
    pixelsafter = after[currentpixels]

    # Sum up all the available region of each tracker
    for i in range(len(trkedboxes)):
        overlap[i] = np.sum(pixelsbefore == (i + 1)) + np.sum(
            pixelsafter == (i + 1))
        iou_2d[i] = compute_iou(trkdet[i], det)
        same_layer[i] = np.sum((abs(trkeddepths[i] - trkeddepths) < 1)) > 1
        valid_trkbox[i] = np.sum(before == (i+1)) + np.sum(after == (i+1))
    trkedboxesarr = np.array(trkedboxes).astype('float')
    overlap = overlap.astype('float')

    # Calculate the IOU
    #trkareas = (trkedboxesarr[:, 2] - trkedboxesarr[:, 0]) * (
    #        trkedboxesarr[:, 3] - trkedboxesarr[:, 1])
    trkareas = valid_trkbox
    trkareas += (x2 - x1) * (y2 - y1)
    trkareas -= overlap
    occ_iou = overlap / (trkareas + (trkareas == 0).astype(int))
    occ_iou[occ_iou > 1.0] = 1.0

    #print(occ_iou, iou_2d, same_layer)
    occ_iou += same_layer * (iou_2d - occ_iou)
    # Check if depth difference is within the diagonal distance of two cars
    occ_iou *= (detdim[2] + detdim[1] + trkdims[:, 2] + trkdims[:, 1]) > \
                abs(detdepth - trkeddepths)
    return occ_iou


def construct2dlayout(trks, dims, rots, cam_calib, pose, cam_near_clip=0.15):
    depths = []
    boxs = []
    points = []
    corners_camera = worldtocamera(trks, pose)
    for corners, dim, rot in zip(corners_camera, dims, rots):
        # in camera coordinates
        points3d = computeboxes(rot, dim, corners)
        depths.append(corners[2])
        projpoints = draw_box(cam_calib, pose, points3d, cam_near_clip)
        points.append(projpoints)
        if projpoints == []:
            box = np.array([-1000, -1000, -1000, -1000])
            boxs.append(box)
            depths[-1] = -10
            continue
        projpoints = np.vstack(projpoints)[:, :2]
        projpoints = projpoints.reshape(-1,2)
        minx = projpoints[:,0].min()
        maxx = projpoints[:,0].max()
        miny = projpoints[:,1].min()
        maxy = projpoints[:,1].max()
        box = np.array([minx, miny, maxx, maxy])
        boxs.append(box)
    return boxs, depths, points


def computeboxes(roty, dim, loc):
    '''Get 3D bbox vertex in camera coordinates 
    Input:
        roty: (1,), object orientation, -pi ~ pi
        box_dim: a tuple of (h, w, l)
        loc: (3,), box 3D center
    Output:
        vertex: numpy array of shape (8, 3) for bbox vertex
    '''
    roty = roty[0]
    face_idx = np.array([[1, 2, 6, 5],  # front face
                         [2, 3, 7, 6],  # left face
                         [3, 4, 8, 7],  # back face
                         [4, 1, 5, 8],
                         [1, 6, 5, 2]], dtype=np.int32) - 1  # right
    R = np.array([[+np.cos(roty), 0, +np.sin(roty)],
                  [0, 1, 0],
                  [-np.sin(roty), 0, +np.cos(roty)]])
    corners = get_vertex(dim)
    corners = corners.dot(R.T) + loc
    return corners


def get_vertex(box_dim):
    '''Get 3D bbox vertex (used for the upper volume iou calculation)
    Input:
        box_dim: a tuple of (h, w, l)
    Output:
        vertex: numpy array of shape (8, 3) for bbox vertex
    '''
    h, w, l = box_dim
    corners = np.array(
        [[l / 2,  l / 2, -l / 2, -l / 2,  l / 2,  l / 2, -l / 2, -l / 2],
         [h / 2,  h / 2,  h / 2,  h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
         [w / 2, -w / 2, -w / 2,  w / 2,  w / 2, -w / 2, -w / 2,  w / 2]])
    return corners.T


def draw_3d_cube(frame,
                 points_camera,
                 tid,
                 cam_calib,
                 cam_pose,
                 cam_near_clip=0.15,
                 cam_field_of_view=60.0,
                 line_color=(0, 255, 0),
                 line_width=3,
                 corner_info=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1

    if corner_info:
        cp1 = cameratoimage(points_camera[0:1], cam_calib)[0]
        if cp1 is None:
            return frame
        before = is_before_clip_plane_camera(points_camera[0:1], 
                                            cam_near_clip)[0]
        if not before:
            return frame

        x1 = int(cp1[0])
        y1 = int(cp1[1])
        cv2.putText(
            frame, str(int(tid) % 1000), (x1, y1), font, font_size,
            line_color, 2, cv2.LINE_AA)

    projpoints = draw_box(cam_calib, cam_pose, points_camera, cam_near_clip)

    for p1, p2 in projpoints:
        cv2.line(frame, 
                (int(p1[0]), int(p1[1])), 
                (int(p2[0]), int(p2[1])),
                line_color,
                line_width)

    return frame


def draw_box(cam_calib, cam_pose, points3d, cam_near_clip=0.15):
    '''Get 3D bbox vertex in camera coordinates 
    Input:
        cam_calib: (3, 4), projection matrix
        cam_pose: a class with position, rotation of the frame
            rotation:  (3, 3), rotation along camera coordinates
            position:  (3), translation of world coordinates
        points3d: (8, 3), box 3D center in camera coordinates
        cam_near_clip: in meter, distance to the near plane
    Output:
        points: numpy array of shape (8, 2) for bbox in image coordinates
    '''
    lineorder = np.array([[1, 2, 6, 5],  # front face
                          [2, 3, 7, 6],  # left face
                          [3, 4, 8, 7],  # back face
                          [4, 1, 5, 8],
                          [1, 6, 5, 2]], dtype=np.int32) - 1  # right

    points = []

    # In camera coordinates
    cam_dir = np.array([0, 0, 1])
    center_pt = cam_dir * cam_near_clip

    for i in range(len(lineorder)):
        for j in range(4):
            p1 = points3d[lineorder[i, j]].copy()
            p2 = points3d[lineorder[i, (j + 1) % 4]].copy()

            before1 = is_before_clip_plane_camera(p1[np.newaxis], 
                                                cam_near_clip)[0]
            before2 = is_before_clip_plane_camera(p2[np.newaxis], 
                                                cam_near_clip)[0]

            inter = get_intersect_point(
                center_pt, cam_dir, p1, p2)

            if not (before1 or before2):
                # print("Not before 1 or 2")
                continue
            elif before1 and before2:
                # print("Both 1 and 2")
                vp1 = p1
                vp2 = p2
            elif before1 and not before2:
                # print("before 1 not 2")
                vp1 = p1
                vp2 = inter
            elif before2 and not before1:
                # print("before 2 not 1")
                vp1 = inter
                vp2 = p2

            cp1 = cameratoimage(vp1[np.newaxis], cam_calib)[0]
            cp2 = cameratoimage(vp2[np.newaxis], cam_calib)[0]
            points.append((cp1, cp2))
    return points



def build_frame_annotation(gt_boxes,
                           gt_boxes_ignored,
                           gt_tracking_ids,
                           gt_dim,
                           gt_alpha,
                           gt_depth,
                           gt_center,
                           cam_calib,
                           cam_rot,
                           cam_loc):
    frame_annotation = []
    assert len(gt_boxes) == len(gt_boxes_ignored) == len(gt_tracking_ids), \
        "gt_boxes: {}, gt_ignore: {}, gt_tr_id: {}".format(
            len(gt_boxes),
            len(gt_boxes_ignored),
            len(gt_tracking_ids))
    for box, ignore, tid, dim, alpha, depth, center in zip(gt_boxes,
                                                           gt_boxes_ignored,
                                                           gt_tracking_ids,
                                                           gt_dim,
                                                           gt_alpha,
                                                           gt_depth,
                                                           gt_center):
        x1, y1, x2, y2 = box.astype(float)
        height = float(y2 - y1)
        width = float(x2 - x1)
        assert (height >= 0) and (width >= 0)
        anno = {'dco': float(ignore),
                'box': [x1, y1, x2, y2],
                'height': height,
                'width': width,
                'id': int(tid),
                'x': float(x1 / 2 + x2 / 2),
                'y': float(y1 / 2 + y2 / 2),
                'xc': center[0].astype(float),  # 3D projected center
                'yc': center[1].astype(float),
                'dim': [d.astype(float) for d in dim],
                'alpha': alpha.astype(float),
                'depth': depth.astype(float),
                'cam_calib': cam_calib.astype(float).tolist(),
                'cam_rot': cam_rot.astype(float).tolist(),
                'cam_loc': cam_loc.astype(float).tolist()
                }
        frame_annotation.append(anno)

    return frame_annotation


def associate_detections_to_trackers(detections, trackers, affinity_mat,
                                     affinity_threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0): return np.empty((0, 2), dtype=int), np.arange(
            len(detections)), np.empty((0, 2), dtype=int)

    matched_indices = linear_assignment(-affinity_mat)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if affinity_mat[m[0], m[1]] < affinity_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def box_filter(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    #print('Area', areas)
    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ovr[inter >= thresh * areas[order[1:]]] = 1
        if np.any(inter >= thresh * areas[i]):
            print("Delete mis detection")
            keep = keep[:-1]
        #print(keep, i, inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_cpu(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def compute_cos_dis(featA, featB):
    return np.exp(-skp.pairwise_distances(featA, featB))


def compute_cos_sim(featA, featB):
    sim = np.dot(featA, featB.T)
    sim /= np.linalg.norm(featA, axis=1).reshape(featA.shape[0], 1)
    sim /= np.linalg.norm(featB, axis=1).reshape(featB.shape[0], 1).T
    return sim


@numba.jit(nopython=True, nogil=True)
def compute_iou(boxA, boxB):
    if boxA[0] > boxB[2] or boxB[0] > boxA[2] or boxA[1] > boxB[3] \
            or boxB[1] > boxA[3]:
        return 0
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def compute_iou_arr(boxA, boxB):
    """
    compute IOU in batch format without for-loop
    NOTE: only work in normalized coordinate (x, y in [0, 1])
    boxA: a array of box with shape [N1, 4]
    boxB: a array of box with shape [N2, 4]

    return a array of IOU with shape [N1, N2]
    """
    boxBt = boxB.transpose()

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[:, 0:1], boxBt[0:1, :])
    yA = np.maximum(boxA[:, 1:2], boxBt[1:2, :])
    xB = np.minimum(boxA[:, 2:3], boxBt[2:3, :])
    yB = np.minimum(boxA[:, 3:4], boxBt[3:4, :])

    # compute the area of intersection rectangle
    x_diff = np.maximum(xB - xA, 0)
    y_diff = np.maximum(yB - yA, 0)
    interArea = (x_diff + 1) * (y_diff + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[:, 2:3] - boxA[:, 0:1] + 1) * \
               (boxA[:, 3:4] - boxA[:, 1:2] + 1)
    boxBArea = (boxBt[2:3, :] - boxBt[0:1, :] + 1) * \
               (boxBt[3:4, :] - boxBt[1:2, :] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_iou(box_t1, box_t2, thres):
    """
    Input:
        box_t1: (N1, 4)
        box_t2: (N2, 4)
        thres: single value
    Output:
        iou: Float (N1, N2)
        idx: Long (N1, 1)
        valid: Float (N1, 1)
    """
    # Get IOU
    iou_tensor = compute_iou_arr(box_t1, box_t2)  # [x1, y1, x2, y2]

    # Select index
    val = np.max(iou_tensor, axis=1)
    idx = iou_tensor.argmax(axis=1)

    # Matched index
    valid = (val > thres).reshape(-1, 1)

    return iou_tensor, idx, valid


def matching(box_t1, box_t2, thres=0.85):
    """
    Match input order by box IOU, select matched feature and box at time t2.
    The match policy is as follows:
        time t1: matched - USE
                 not matched - DUPLICATE ITSELF
        time t2: matched - USE
                 not matched - DISCARD
    Use idx as a match order index to swap their order
    Use valid as a valid threshold to keep size by duplicating t1 itself

    Input:
        box_t1: (N1, 4)
        box_t2: (N2, 4)

    Inter:
        iou: Float
        matches: Float
        idx: Long
        valid: Float

    Output:
        new_box_t2: (N1, 4)
    """
    # Get IOU
    iou, idx, valid = get_iou(box_t1, box_t2, thres)

    # Select features
    new_box_t2 = box_t2[idx] * valid + box_t1 * (1 - valid)

    return new_box_t2, idx, valid


@numba.jit()
def rad2deg(rad):
    return rad * 180.0 / np.pi

@numba.jit()
def deg2rad(deg):
    return deg / 180.0 * np.pi

@numba.jit()
def rot_y2alpha(rot_y, x, FOCAL_LENGTH):
    """
    Get alpha by rotation_y - theta + 180
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    alpha : Observation angle of object, ranging [-pi..pi]
    """
    alpha = rad2deg(rot_y - np.arctan2(x, FOCAL_LENGTH)) + 180
    alpha = alpha % 360 - 180
    return alpha

@numba.jit()
def alpha2rot_y(alpha, x, FOCAL_LENGTH):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = rad2deg(alpha + np.arctan2(x, FOCAL_LENGTH)) - 180
    rot_y = rot_y % 360 - 180
    return rot_y


@numba.jit(nopython=True, nogil=True)
def rot_axis(angle, axis):
    # RX = np.array([ [1,             0,              0],
    #                 [0, np.cos(gamma), -np.sin(gamma)],
    #                 [0, np.sin(gamma),  np.cos(gamma)]])
    #
    # RY = np.array([ [ np.cos(beta), 0, np.sin(beta)],
    #                 [            0, 1,            0],
    #                 [-np.sin(beta), 0, np.cos(beta)]])
    #
    # RZ = np.array([ [np.cos(alpha), -np.sin(alpha), 0],
    #                 [np.sin(alpha),  np.cos(alpha), 0],
    #                 [            0,              0, 1]])
    cg = np.cos(angle)
    sg = np.sin(angle)
    if axis == 0:  # X
        v = [0, 4, 5, 7, 8]
    elif axis == 1:  # Y
        v = [4, 0, 6, 2, 8]
    else:  # Z
        v = [8, 0, 1, 3, 4]
    RX = np.zeros(9, dtype=numba.float64)
    RX[v[0]] = 1.0
    RX[v[1]] = cg
    RX[v[2]] = -sg
    RX[v[3]] = sg
    RX[v[4]] = cg
    return RX.reshape(3, 3)


@numba.jit(nopython=True, nogil=True)
def rotate(vector, angle, inverse=False):
    """
    Rotation of x, y, z axis
    Forward rotate order: Z, Y, X
    Inverse rotate order: X^T, Y^T,Z^T
    Input:
        vector: vector in 3D coordinates
        angle: rotation along X, Y, Z (raw data from GTA)
    Output:
        out: rotated vector
    """
    gamma, beta, alpha = angle[0], angle[1], angle[2]

    # Rotation matrices around the X (gamma), Y (beta), and Z (alpha) axis
    RX = rot_axis(gamma, 0)
    RY = rot_axis(beta, 1)
    RZ = rot_axis(alpha, 2)

    # Composed rotation matrix with (RX, RY, RZ)
    if inverse:
        return np.dot(np.dot(np.dot(RX.T, RY.T), RZ.T), vector)
    else:
        return np.dot(np.dot(np.dot(RZ, RY), RX), vector)


@numba.jit(nopython=True, nogil=True)
def rotate_(a, theta):
    """
    rotate a theta from vector a
    """
    d0 = np.cos(theta[2]) * (np.cos(theta[1]) * a[0]
                             + np.sin(theta[1]) * (
                                     np.sin(theta[0]) * a[1] + np.cos(
                                 theta[0]) * a[2])) \
         - np.sin(theta[2]) * (
                 np.cos(theta[0]) * a[1] - np.sin(theta[0]) * a[2])
    d1 = np.sin(theta[2]) * (np.cos(theta[1]) * a[0]
                             + np.sin(theta[1]) * (
                                     np.sin(theta[0]) * a[1] + np.cos(
                                 theta[0]) * a[2])) \
         + np.cos(theta[2]) * (
                 np.cos(theta[0]) * a[1] - np.sin(theta[0]) * a[2])
    d2 = -np.sin(theta[1]) * a[0] + np.cos(theta[1]) * \
         (np.sin(theta[0]) * a[1] + np.cos(theta[0]) * a[2])

    vector = np.zeros(3, dtype=numba.float64)
    vector[0] = d0
    vector[1] = d1
    vector[2] = d2

    return vector



def cal_3D_iou(vol_box_pd, vol_box_gt):
    vol_inter = intersect_bbox_with_yaw(vol_box_pd, vol_box_gt)
    vol_gt = intersect_bbox_with_yaw(vol_box_gt, vol_box_gt)
    vol_pd = intersect_bbox_with_yaw(vol_box_pd, vol_box_pd)

    return get_vol_iou(vol_pd, vol_gt, vol_inter)


def intersect_bbox_with_yaw(box_a, box_b):
    """
    A simplified calculation of 3d bounding box intersection.
    It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.
    :param box_a, box_b: obstacle bounding boxes for comparison
    :return: intersection volume (float)
    """
    # height (Z) overlap
    min_h_a = np.min(box_a[2])
    max_h_a = np.max(box_a[2])
    min_h_b = np.min(box_b[2])
    max_h_b = np.max(box_b[2])
    max_of_min = np.max([min_h_a, min_h_b])
    min_of_max = np.min([max_h_a, max_h_b])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        # print("Z = 0")
        return 0.

    # oriented XY overlap
    # TODO: Check if the order of 3D box is correct
    xy_poly_a = Polygon(zip(*box_a[0:2, 0:4]))
    xy_poly_b = Polygon(zip(*box_b[0:2, 0:4]))
    xy_intersection = xy_poly_a.intersection(xy_poly_b).area
    if xy_intersection == 0:
        # print("XY = 0")
        return 0.

    return z_intersection * xy_intersection


@numba.jit(nopython=True, nogil=True)
def get_vol_iou(vol_a, vol_b, vol_intersect):
    union = vol_a + vol_b - vol_intersect
    return vol_intersect / union if union else 0.



@numba.jit(nopython=True)
def get_intersect_point(center_pt, cam_dir, vertex1, vertex2):
    # get the intersection point of two 3D points and a plane
    c1 = center_pt[0]
    c2 = center_pt[1]
    c3 = center_pt[2]
    a1 = cam_dir[0]
    a2 = cam_dir[1]
    a3 = cam_dir[2]
    x1 = vertex1[0]
    y1 = vertex1[1]
    z1 = vertex1[2]
    x2 = vertex2[0]
    y2 = vertex2[1]
    z2 = vertex2[2]

    k_up = abs(a1 * (x1 - c1) + a2 * (y1 - c2) + a3 * (z1 - c3))
    k_down = abs(a1 * (x1 - x2) + a2 * (y1 - y2) + a3 * (z1 - z2))
    if k_up > k_down:
        k = 1
    else:
        k = k_up / k_down
    inter_point = (1 - k) * vertex1 + k * vertex2
    return inter_point


def is_before_clip_plane_world(points_world, cam_pose, cam_near_clip=0.15):
    """
    points_world: (N, 3), N points on X(right)-Y(front)-Z(up) world coordinate (GTA)
                    or X(front)-Y(left)-Z(up) velodyne coordinates (KITTI)
    pose: a class with position, rotation of the frame
        rotation:  (3, 3), rotation along camera coordinates
        position:  (3,), translation of world coordinates
    cam_near_clip: scalar, the near projection plane

    is_before: (N,), bool, is the point locate before the near clip plane
    """
    return worldtocamera(points_world, cam_pose)[:, 2] > cam_near_clip

def is_before_clip_plane_camera(points_camera, cam_near_clip=0.15):
    """
    points_camera: (N, 3), N points on X(right)-Y(down)-Z(front) camera coordinate
    cam_near_clip: scalar, the near projection plane

    is_before: bool, is the point locate before the near clip plane
    """
    return points_camera[:, 2] > cam_near_clip

if __name__ == '__main__':
    cam_rotation = np.array([45.0, 45.0, 45.0])
    theta = (np.pi / 180.0) * cam_rotation
    vector = np.array([1., 2., 3.])
    print(rotate(vector, theta))
    print(rotate_(vector, theta))
    print(
        np.allclose(rotate(rotate(vector, theta, inverse=True), theta), vector))
    print(np.allclose(rotate_(rotate(vector, theta, inverse=True), theta),
                      vector))
