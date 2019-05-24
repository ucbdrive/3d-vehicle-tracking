# coding: utf-8
import sys
import cv2
import json
import argparse
import numpy as np
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns

from utils.config import cfg
import utils.tracking_utils as tu
import utils.plot_utils as pu


def parse_args():
    parser = argparse.ArgumentParser(
                    description='Monocular 3D Tracking Visualizer',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='gta', choices=['gta', 'kitti'])
    parser.add_argument('split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--session', default='616',
                        help='Name of the session, to separate exp')
    parser.add_argument('--epoch', default='030',
                        help='number of total epochs to run')
    parser.add_argument('--box_key', default='det_box',
                        choices=['det_box', 'trk_box'],
                        help='using detection or tracked boxes')
    parser.add_argument('--fps', default=7, type=int,
                        help='fps to store video')
    parser.add_argument('--json_path', dest='json_path',
                        default='./data/gta5_tracking/',
                        help='path to json files')
    parser.add_argument('--setting', default=None,
                        help='Setting string to separate results. e.g., '
                             '_aff02_hit2')
    parser.add_argument('--select_seq', nargs='*', default=0, type=int,
                        help='Number of the selected sequences')
    parser.add_argument('--is_save', default=False, action='store_true',
                        help='whether to merge two video or not')
    parser.add_argument('--is_merge', default=False, action='store_true',
                        help='whether to merge two video or not')
    parser.add_argument('--draw_3d', default=False, action='store_true',
                        help='draw 3D box')
    parser.add_argument('--draw_2d', default=False, action='store_true',
                        help='draw 2D box')
    parser.add_argument('--draw_bev', default=False, action='store_true',
                        help='draw Birds eye view')
    args = parser.parse_args()
    args.select_seq = [args.select_seq] if isinstance(args.select_seq,
                                                      int) else args.select_seq

    print(' '.join(sys.argv))

    return args


args = parse_args()

# Global Variable
sns.set(style="darkgrid")
FONT = cv2.FONT_HERSHEY_SIMPLEX
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
OUTPUT_PATH = cfg.OUTPUT_PATH
FOV_H = 60
NEAR_CLIP = 0.15

if args.dataset == 'gta':
    W = cfg.GTA.W  # 1920
    H = cfg.GTA.H  # 1080
    resW = W // 2
    resH = H // 2
    FOCAL_LENGTH = cfg.GTA.FOCAL_LENGTH  # 935.3074360871937
else:
    W = cfg.KITTI.W # 1248
    H = cfg.KITTI.H # 384
    resW = W
    resH = H
    FOCAL_LENGTH = cfg.KITTI.FOCAL_LENGTH  # 721.5377


def get_3d_box_from_2d(box_2d_cen, depth, rot_y, dim, cam_calib, cam_pose):
    points_cam = tu.imagetocamera(box_2d_cen, depth, cam_calib)
    vol_box = tu.computeboxes([rot_y], dim, points_cam)
    
    return vol_box


def plot_bev_obj(center, text, rot_y, l, w, plt, color, line_width=1):
    # Calculate length, width of object
    vec_l = [l * np.cos(rot_y), -l * np.sin(rot_y)]
    vec_w = [-w * np.cos(rot_y - np.pi / 2), w * np.sin(rot_y - np.pi / 2)]
    vec_l = np.array(vec_l)
    vec_w = np.array(vec_w)

    # Make 4 points
    p1 = center + 0.5 * vec_l - 0.5 * vec_w
    p2 = center + 0.5 * vec_l + 0.5 * vec_w
    p3 = center - 0.5 * vec_l + 0.5 * vec_w
    p4 = center - 0.5 * vec_l - 0.5 * vec_w

    # Plot object
    line_style = '-' if 'PD' in text else ':'
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], line_style, c=color,
             linewidth=3 * line_width)
    plt.plot([p1[0], p4[0]], [p1[1], p4[1]], line_style, c=color,
             linewidth=line_width)
    plt.plot([p3[0], p2[0]], [p3[1], p2[1]], line_style, c=color,
             linewidth=line_width)
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], line_style, c=color,
             linewidth=line_width)


def plot_3D_box_pd(info_gt, info_pd, args, session_name):
    if args.draw_bev: print("BEV: {}".format(session_name))
    if args.draw_2d or args.draw_3d: print("3D: {}".format(session_name))

    # Variables
    cam_field_of_view_h = FOV_H
    cam_near_clip = NEAR_CLIP
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

    # set output video
    if args.is_save:
        vid_trk = cv2.VideoWriter('{}_{}_{}_{}.mp4'.format(
            session_name,
            args.box_key,
            'tracking',
            '_'.join([str(n) for n in args.select_seq])),
            FOURCC, args.fps, (resW, resH))
        vid_bev = cv2.VideoWriter('{}_{}_{}_{}.mp4'.format(
            session_name,
            args.box_key,
            'birdsview',
            '_'.join([str(n) for n in args.select_seq])),
            FOURCC, args.fps, (resH, resH))
    else:
        vid_trk = None
        vid_bev = None

    # Iterate through all objects
    for n_seq, (pd_seq, gt_seq) in enumerate(zip(info_pd, info_gt)):

        id_to_color = {}
        cmap = pu.RandomColor(len(pd_seq['frames']))
        np.random.seed(777)

        if n_seq not in args.select_seq:
            continue

        for n_frame, (pd_boxes, gt_boxes) in enumerate(
                zip(pd_seq['frames'], gt_seq['frames'])):
            if n_frame % 100 == 0:
                print(n_frame)

            # Get objects
            if args.draw_3d or args.draw_2d:
                rawimg = cv2.imread(gt_boxes['im_path'][0])
                cv2.putText(rawimg, '{}'.format(n_frame), (0, 30), FONT, 1,
                            (0, 0, 0), 2, cv2.LINE_AA)

            if len(gt_boxes['annotations']) > 0:
                cam_coords = np.array(gt_boxes['annotations'][0]['cam_loc'])
                cam_rotation = np.array(gt_boxes['annotations'][0]['cam_rot'])
                cam_calib = np.array(gt_boxes['annotations'][0]['cam_calib'])

                cam_pose = tu.Pose(cam_coords, cam_rotation)

            for i, hypo in enumerate(pd_boxes['hypotheses']):
                tid = hypo['id']

                # Get information of gt and pd
                box_pd = np.array(hypo[args.box_key]).astype(int)

                h_pd, w_pd, l_pd = hypo['dim']
                depth_pd = hypo['depth']
                alpha_pd = hypo['alpha']
                xc_pd, yc_pd = hypo['x'], hypo['y']

                center_pd = np.hstack(
                    [(xc_pd - W // 2) * depth_pd / FOCAL_LENGTH, depth_pd])

                rot_y_pd = tu.deg2rad(
                    tu.alpha2rot_y(alpha_pd,
                                   xc_pd - W // 2,
                                   FOCAL_LENGTH)
                )  # rad

                vol_box_pd = get_3d_box_from_2d(
                    np.array([[xc_pd, yc_pd]]),
                    np.array([depth_pd]),
                    rot_y_pd,
                    (h_pd, w_pd, l_pd),
                    cam_calib,
                    cam_pose)

                # Get box color
                # color is in BGR format (for cv2), color[:-1] in RGB format
                # (for plt)
                if tid not in list(id_to_color):
                    id_to_color[tid] = [cmap.get_random_color(scale=255), 10]
                else:
                    id_to_color[tid][1] = 10
                color, life = id_to_color[tid]

                # Make rectangle
                if args.draw_3d:
                    # Make rectangle
                    rawimg = tu.draw_3d_cube(rawimg, vol_box_pd, tid,
                                             cam_calib,
                                             cam_pose,
                                             line_color=color)
                if args.draw_2d:
                    text_pd = 'PD:{}° {}m'.format(
                        int(alpha_pd / np.pi * 180),
                        int(depth_pd))
                    cv2.rectangle(rawimg,
                                  (box_pd[0], box_pd[1]),
                                  (box_pd[2], box_pd[3]),
                                  color,
                                  10)
                if args.draw_bev:
                    # Change BGR to RGB
                    color_bev = [c / 255.0 for c in color[::-1]]
                    plot_bev_obj(center_pd, 'PD', rot_y_pd, l_pd, w_pd, plt,
                                 color_bev)

            if args.draw_bev:
                # Make plot
                ax.set_aspect('equal', adjustable='box')
                plt.axis([-60, 60, -10, 100])
                # plt.axis([-80, 80, -10, 150])
                plt.plot([0, 0], [0, 3], 'k-')
                plt.plot([-1, 0], [2, 3], 'k-')
                plt.plot([1, 0], [2, 3], 'k-')

            for tid in list(id_to_color):
                id_to_color[tid][1] -= 1
                if id_to_color[tid][1] < 0:
                    del id_to_color[tid]

            # Plot
            if vid_trk:
                vid_trk.write(cv2.resize(rawimg, (resW, resH)))
            elif args.draw_3d or args.draw_2d:
                #draw_img = rawimg[:, :, ::-1]

                #fig = plt.figure(figsize=(18, 9), dpi=50)
                #plt.imshow(draw_img)
                key = 0
                while(key not in [ord('q'), ord(' '), 27]):
                    cv2.imshow('preview', cv2.resize(rawimg, (resW, resH)))
                    key = cv2.waitKey(1)

                if key == 27:
                    cv2.destroyAllWindows()
                    return

            # Plot
            if vid_bev:
                fig_data = pu.fig2data(plt.gcf())
                vid_bev.write(cv2.resize(fig_data, (resH, resH)))
                plt.clf()
            elif args.draw_bev:
                plt.show()
                plt.clf()

    if args.is_save:
        vid_trk.release()
        vid_bev.release()


def plot_3D_box(info_gt, info_pd, args, session_name):
    if args.draw_bev: print("BEV: {}".format(session_name))
    if args.draw_2d or args.draw_3d: print("3D: {}".format(session_name))

    # Variables
    cam_field_of_view_h = FOV_H
    cam_near_clip = NEAR_CLIP
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

    # set output video
    if args.is_save:
        vid_trk = cv2.VideoWriter('{}_{}_{}_{}.mp4'.format(
            session_name,
            args.box_key,
            'tracking',
            '_'.join([str(n) for n in args.select_seq])),
            FOURCC, args.fps, (resW, resH))
        vid_bev = cv2.VideoWriter('{}_{}_{}_{}.mp4'.format(
            session_name,
            args.box_key,
            'birdsview',
            '_'.join([str(n) for n in args.select_seq])),
            FOURCC, args.fps, (resH, resH))
    else:
        vid_trk = None
        vid_bev = None

    # Iterate through all objects
    for n_seq, (pd_seq, gt_seq) in enumerate(zip(info_pd, info_gt)):

        id_to_color = {}
        cmap = pu.RandomColor(len(gt_seq['frames']))
        np.random.seed(777)

        if n_seq not in args.select_seq:
            continue

        for n_frame, (pd_boxes, gt_boxes) in enumerate(
                zip(pd_seq['frames'], gt_seq['frames'])):
            if n_frame % 100 == 0:
                print(n_frame)

            # Get objects
            if args.draw_3d or args.draw_2d:
                rawimg = cv2.imread(gt_boxes['im_path'][0])
                cv2.putText(rawimg, '{}'.format(n_frame), (0, 30), FONT, 1,
                            (0, 0, 0), 2, cv2.LINE_AA)

            if len(gt_boxes['annotations']) > 0:
                cam_coords = np.array(gt_boxes['annotations'][0]['cam_loc'])
                cam_rotation = np.array(gt_boxes['annotations'][0]['cam_rot'])
                cam_calib = np.array(gt_boxes['annotations'][0]['cam_calib'])

                cam_pose = tu.Pose(cam_coords, cam_rotation)
                boxes_pd = [hypo[args.box_key] for hypo in
                            pd_boxes['hypotheses']]

            for i, anno in enumerate(gt_boxes['annotations']):

                tid = anno['id']
                box_gt = np.array(anno['box']).astype(int)
                h_gt, w_gt, l_gt = anno['dim']
                depth_gt = anno['depth']
                alpha_gt = anno['alpha']
                xc_gt, yc_gt = anno['xc'], anno['yc']

                center_gt = np.hstack(
                    [(xc_gt - W // 2) * depth_gt / FOCAL_LENGTH, depth_gt])

                rot_y_gt = tu.deg2rad(
                    tu.alpha2rot_y(alpha_gt,
                                   xc_gt - W // 2,
                                   FOCAL_LENGTH)
                )  # degree

                vol_box_gt = get_3d_box_from_2d(
                    np.array([[xc_gt, yc_gt]]),
                    np.array([depth_gt]),
                    rot_y_gt,
                    (h_gt, w_gt, l_gt),
                    cam_calib,
                    cam_pose)

                # Match gt and pd
                has_match = len(boxes_pd) != 0
                if has_match:
                    _, idx, valid = tu.matching(
                        np.array(anno['box']).reshape(-1, 4),
                        np.array(boxes_pd).reshape(-1, 5)[:, :4],
                        0.8)
                    has_match = has_match and valid.item()

                    hypo = pd_boxes['hypotheses'][idx[0]]

                    # Get information of gt and pd
                    box_pd = np.array(hypo[args.box_key]).astype(int)

                    h_pd, w_pd, l_pd = hypo['dim']
                    depth_pd = hypo['depth']
                    alpha_pd = hypo['alpha']
                    xc_pd, yc_pd = hypo['xc'], hypo['yc']

                    center_pd = np.hstack(
                        [(xc_pd - W // 2) * depth_pd / FOCAL_LENGTH, depth_pd])

                    rot_y_pd = tu.deg2rad(
                        tu.alpha2rot_y(alpha_pd,
                                       xc_pd - W // 2,
                                       FOCAL_LENGTH)
                    )  # rad

                    vol_box_pd = get_3d_box_from_2d(
                        np.array([[xc_pd, yc_pd]]),
                        np.array([depth_pd]),
                        roy_y_pd,
                        (h_pd, w_pd, l_pd),
                        cam_calib,
                        cam_pose)

                # Get box color
                # color is in BGR format (for cv2), color[:-1] in RGB format
                # (for plt)
                if tid not in list(id_to_color):
                    id_to_color[tid] = [cmap.get_random_color(scale=255), 10]
                else:
                    id_to_color[tid][1] = 10
                color, life = id_to_color[tid]

                # Make rectangle
                if args.draw_3d:
                    # Make rectangle
                    rawimg = tu.draw_3d_cube(rawimg, vol_box_gt, tid,
                                             cam_calib,
                                             cam_pose,
                                             line_color=(color[0],
                                                         color[1] * 0.7,
                                                         color[2] * 0.7),
                                             line_width=2)
                    if has_match:
                        rawimg = tu.draw_3d_cube(rawimg, vol_box_pd, tid,
                                                 cam_calib,
                                                 cam_pose,
                                                 line_color=color)
                if args.draw_2d:
                    text_gt = 'GT:{}° {}m'.format(
                        int(alpha_gt / np.pi * 180),
                        int(depth_gt))
                    cv2.rectangle(rawimg,
                                  (box_gt[0], box_gt[1]),
                                  (box_gt[2], box_gt[3]),
                                  (color[0],
                                   color[1] * 0.7,
                                   color[2] * 0.7),
                                  8)
                    if has_match:
                        text_pd = 'PD:{}° {}m'.format(
                            int(alpha_pd / np.pi * 180),
                            int(depth_pd))
                        cv2.rectangle(rawimg,
                                      (box_pd[0], box_pd[1]),
                                      (box_pd[2], box_pd[3]),
                                      color,
                                      10)
                if args.draw_bev:
                    # Change BGR to RGB
                    color_bev = [c / 255.0 for c in color[::-1]]
                    if has_match:
                        plot_bev_obj(center_pd, 'PD', rot_y_pd, l_pd, w_pd, plt,
                                     color_bev)

                    plot_bev_obj(center_gt, 'GT', rot_y_gt, l_gt, w_gt, plt,
                                 color_bev)

            if args.draw_bev:
                # Make plot
                ax.set_aspect('equal', adjustable='box')
                plt.axis([-60, 60, -10, 100])
                # plt.axis([-80, 80, -10, 150])
                plt.plot([0, 0], [0, 3], 'k-')
                plt.plot([-1, 0], [2, 3], 'k-')
                plt.plot([1, 0], [2, 3], 'k-')

            for tid in list(id_to_color):
                id_to_color[tid][1] -= 1
                if id_to_color[tid][1] < 0:
                    del id_to_color[tid]

            # Plot
            if vid_trk:
                vid_trk.write(cv2.resize(rawimg, (resW, resH)))
            elif args.draw_3d or args.draw_2d:
                #draw_img = rawimg[:, :, ::-1]

                #fig = plt.figure(figsize=(18, 9), dpi=50)
                #plt.imshow(draw_img)
                key = 0
                while(key not in [ord('q'), ord(' '), 27]):
                    cv2.imshow('preview', cv2.resize(rawimg, (resW, resH)))
                    key = cv2.waitKey(1)

                if key == 27:
                    cv2.destroyAllWindows()
                    return

            # Plot
            if vid_bev:
                fig_data = pu.fig2data(plt.gcf())
                vid_bev.write(cv2.resize(fig_data, (resH, resH)))
                plt.clf()
            elif args.draw_bev:
                plt.show()
                plt.clf()

    if args.is_save:
        vid_trk.release()
        vid_bev.release()


def save_vid(info_gt, info_pd, session_name, args):
    # Loop over save_range and plot the BEV
    print("Total {} frames. Now saving...".format(
        sum([len(seq['frames']) for seq in info_pd])))
    # plot_3D_box(info_gt, info_pd, args, session_name)
    plot_3D_box_pd(info_gt, info_pd, args, session_name)
    print("Done!")


def merge_vid(vidname1, vidname2, outputname):
    print("Vertically stack {} and {}, save as {}".format(vidname1, vidname2,
                                                          outputname))

    # Get input video capture
    cap1 = cv2.VideoCapture(vidname1)
    cap2 = cv2.VideoCapture(vidname2)

    # Default resolutions of the frame are obtained.The default resolutions
    # are system dependent.
    # We convert the resolutions from float to integer.
    # https://docs.opencv.org/2.4/modules/highgui/doc
    # /reading_and_writing_images_and_video.html#videocapture-get
    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    fps1 = cap1.get(5)
    num_frames = int(cap1.get(7))

    frame_width2 = int(cap2.get(3))
    frame_height2 = int(cap2.get(4))
    fps2 = cap2.get(5)
    num_frames2 = int(cap2.get(7))

    print(fps1, fps2)
    if fps1 > fps2:
        fps = fps1
    else:
        fps = fps2

    assert frame_height == frame_height2, "Height of frames are not equal. {} " \
                                          "" \
                                          "vs. {}".format(
        frame_height,
        frame_height2)
    assert num_frames == num_frames2, "Number of frames are not equal. {} vs." \
                                      " {}".format(
        num_frames, num_frames2)

    # Set output videowriter
    vidsize = (frame_width + frame_width2, frame_height)
    out = cv2.VideoWriter(outputname + '.mp4', FOURCC, fps, vidsize)

    # Loop over and save
    print("Total {} frames. Now saving...".format(num_frames))
    idx = 0
    while (cap1.isOpened() and cap2.isOpened() and idx < num_frames):
        if idx % 100 == 0:
            print(idx)
        if (idx % (fps / fps1) == 0.0):
            # print(idx, fps/fps2, "1")
            ret1, frame1 = cap1.read()
        if (idx % (fps / fps2) == 0.0):
            # print(idx, fps/fps1, "2")
            ret2, frame2 = cap2.read()
        # print(ret1, ret2)
        if (ret1 == True) and (ret2 == True):
            # cv2.putText(frame1, vidname1, (45, 30), FONT, 1, (0, 0, 0), 2,
            # cv2.LINE_AA)
            # cv2.putText(frame2, vidname2, (45, 30), FONT, 1, (0, 0, 0), 2,
            # cv2.LINE_AA)
            out_frame = np.hstack([frame1, frame2])
            out.write(out_frame)
        idx += 1

    out.release()
    cap1.release()
    cap2.release()

    print("Done!")


def main():
    _METHOD_NAMES = ['none', 'kf2ddeep', 'kf3ddeep', 'lstmdeep', 'lstmoccdeep']
    # _METHOD_NAMES = ['none', 'kf2d', 'kf2ddeep', 'kf3d',
    # 'kf3ddeep', 'lstm', 'lstmdeep', 'occdeep',
    # 'kf3doccdeep', 'lstmoccdeep']
    #_METHOD_NAMES = ['lstm']

    output_path = '{}/{}_{}_{}_{}_set/'.format(OUTPUT_PATH, 
                                            args.session, 
                                            args.epoch, 
                                            args.dataset, 
                                            args.split)

    # Get informations
    for name in _METHOD_NAMES:
        session_name = "{}_{}".format(name, args.setting)
        print("Loading information {} ...".format(session_name))
        info_gt = json.load(open(join(output_path, 'gt.json'), 'r'))
        info_pd = json.load(open(join(output_path, '{ID}_pd.json'.format(
            **{'ID': session_name})), 'r'))
        save_vid(info_gt, info_pd, session_name, args)

        if args.is_merge:
            # Merge two video vertically
            seq_id = '_'.join([str(n) for n in args.select_seq])
            vidname1 = '{}_{}_{}_{}.mp4'.format(session_name, args.box_key,
                                                'tracking', seq_id)
            vidname2 = '{}_{}_{}_{}.mp4'.format(session_name, args.box_key,
                                                'birdsview', seq_id)
            outputname = '{}_{}_compose'.format(vidname1, vidname2)
            merge_vid(vidname1, vidname2, outputname)


if __name__ == '__main__':
    main()
