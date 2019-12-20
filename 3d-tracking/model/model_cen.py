import numpy as np
import torch
import torch.nn as nn

import utils.network_utils as nu
import utils.tracking_utils as tu
from model import dla_up
from lib.model.roi_layers import ROIAlign, ROIPool


class Model(nn.Module):

    def __init__(self, arch_name, roi_name, down_ratio, roi_kernel):
        super(Model, self).__init__()
        self.base = dla_up.__dict__[arch_name](
            pretrained_base='imagenet', down_ratio=down_ratio)

        num_channel = self.base.channels[int(np.log2(down_ratio))]

        # We use roialign with kernel size = 7 in our experiments
        assert ('align' in roi_name or 'pool' in roi_name)
        assert (roi_kernel == 7)

        if 'align' in roi_name:
            print('Using RoIAlign')
            self.roi_pool = ROIAlign(
                                    (roi_kernel, roi_kernel), 
                                    1.0 / down_ratio, 
                                    0)
        elif 'pool' in roi_name:
            print('Using RoIPool')
            self.roi_pool = ROIPool(
                                    (roi_kernel, roi_kernel), 
                                    1.0 / down_ratio)

        self.dim = nn.Sequential(
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, 3, kernel_size=1,
                      stride=1, padding=0, bias=True))  # 3 dim

        self.rot = nn.Sequential(
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, 8, kernel_size=1,
                      stride=1, padding=0, bias=True))  # 1 + 1 + 2

        self.dep = nn.Sequential(
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, 1, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.Sigmoid())

        self.cen = nn.Sequential(
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, 2, kernel_size=1,
                      stride=1, padding=0, bias=True))

        nu.init_module(self.base)
        nu.init_module(self.dim)
        nu.init_module(self.rot)
        nu.init_module(self.dep)
        nu.init_module(self.cen)

    def forward(self, image, box_info, device, phase):

        # for 3D
        rois = box_info['rois_pd']

        # Get box info
        num_imgs = image.size(0)
        n_gt_box = box_info['n_box'].cpu().numpy()
        n_pd_box = torch.sum(rois[:, :, 4] > 0, dim=1).cpu().numpy()

        # Check number of boxes
        num_rois = int(np.sum(n_gt_box))  # get n_gt_box of this frame
        if (n_gt_box == 0).any(): print("GT is empty")
        num_det = int(np.sum(n_pd_box))  # get n_pd_box of this frame
        if (n_pd_box == 0).any(): print("Prediction is empty")

        # Make sure if n_gt_box and n_pd_box are the same during training
        if phase in ['train', 'val']:
            assert (n_pd_box == n_gt_box).any(), \
                "Number of pred. bbox ({}) not equals to gt ({})".format(
                    n_pd_box, n_gt_box)

        # Init 
        image = image.to(device)
        boxes = torch.zeros([num_det, 5]).to(device)
        cen_pd = torch.zeros([num_det, 2]).to(device)
        rois_pd = torch.zeros([num_det, 5]).to(device)
        rois_gt = torch.zeros([num_rois, 5]).to(device)
        dim_gt = torch.zeros([num_rois, 3]).to(device)
        dep_gt = torch.zeros([num_rois]).to(device)
        cen_gt = torch.zeros([num_rois, 2]).to(device)
        loc_gt = torch.zeros([num_rois, 3]).to(device)
        tid_gt = torch.zeros([num_rois]).to(device)
        if phase == 'train':
            bin_gt = torch.zeros([num_rois, 2]).to(device).long()
            res_gt = torch.zeros([num_rois, 2]).to(device)
        else:
            alpha_gt = torch.zeros([num_rois]).to(device)
            ignore = torch.zeros([num_rois]).to(device)

        # Feed valid info to gpu
        sum_gt = 0
        sum_det = 0
        for idx in range(num_imgs):
            if n_pd_box[idx] > 0:
                # indicate which image to get feature
                boxes[sum_det:sum_det + n_pd_box[idx], 0] = idx  
                boxes[sum_det:sum_det + n_pd_box[idx], 1:5] = rois[idx,
                                                              :n_pd_box[idx],
                                                              0:4]  # box
                cen_pd[sum_det:sum_det + n_pd_box[idx]] = box_info['cen_pd'][idx,
                                                       :n_pd_box[idx]]
                rois_pd[sum_det:sum_det + n_pd_box[idx]] = rois[idx,
                                                           :n_pd_box[idx],
                                                           :]  # for tracking
            if n_gt_box[idx] > 0:
                dim_gt[sum_gt:sum_gt + n_gt_box[idx]] = box_info['dim_gt'][idx,
                                                        :n_gt_box[idx]]
                dep_gt[sum_gt:sum_gt + n_gt_box[idx]] = box_info['depth_gt'][
                                                        idx, :n_gt_box[idx]]
                cen_gt[sum_gt:sum_gt + n_gt_box[idx]] = box_info['cen_gt'][idx,
                                                        :n_gt_box[idx]]
                loc_gt[sum_gt:sum_gt + n_gt_box[idx]] = box_info['loc_gt'][idx,
                                                        :n_gt_box[idx]]
                tid_gt[sum_gt:sum_gt + n_gt_box[idx]] = box_info['tid_gt'][idx,
                                                        :n_gt_box[idx]]
                rois_gt[sum_gt:sum_gt + n_gt_box[idx]] = box_info['rois_gt'][
                                                         idx, :n_gt_box[idx]]
                if phase == 'train':
                    bin_gt[sum_gt:sum_gt + n_gt_box[idx]] = box_info[
                                                                'bin_cls_gt'][
                                                            idx, :n_gt_box[idx]]
                    res_gt[sum_gt:sum_gt + n_gt_box[idx]] = box_info[
                                                                'bin_res_gt'][
                                                            idx, :n_gt_box[idx]]
                else:
                    alpha_gt[sum_gt:sum_gt + n_gt_box[idx]] = box_info[
                                                                  'alpha_gt'][
                                                              idx,
                                                              :n_gt_box[idx]]
                    ignore[sum_gt:sum_gt + n_gt_box[idx]] = box_info['ignore'][
                                                            idx, :n_gt_box[idx]]
            sum_gt += n_gt_box[idx]
            sum_det += n_pd_box[idx]

        # Inference of 3D estimation
        img_feat = self.base(image)
        if num_det > 0:
            pooled_feat = self.roi_pool(img_feat, boxes)
            dim = self.dim(pooled_feat).flatten(start_dim=1)
            cen = self.cen(pooled_feat).flatten(start_dim=1) + cen_pd
            orient_ = self.rot(pooled_feat).flatten(start_dim=1)
            # bin 1
            divider1 = torch.sqrt(orient_[:, 2:3] ** 2 + orient_[:, 3:4] ** 2)
            b1sin = orient_[:, 2:3] / divider1
            b1cos = orient_[:, 3:4] / divider1

            # bin 2
            divider2 = torch.sqrt(orient_[:, 6:7] ** 2 + orient_[:, 7:8] ** 2)
            b2sin = orient_[:, 6:7] / divider2
            b2cos = orient_[:, 7:8] / divider2

            rot = torch.cat(
                [orient_[:, 0:2], b1sin, b1cos, orient_[:, 4:6], b2sin, b2cos],
                1)
            dep = nu.get_pred_depth(self.dep(pooled_feat).flatten())

            loc_pd = []
            sum_l = 0
            for l_idx in range(num_imgs):
                if n_pd_box[l_idx] == 0:
                    continue
                cam_calib = box_info['cam_calib'][l_idx]
                position = box_info['cam_loc'][l_idx]
                rotation = box_info['cam_rot'][l_idx]
                loc_pd.append(tu.point3dcoord_torch(
                    cen[sum_l:sum_l + n_pd_box[l_idx]],
                    dep[sum_l:sum_l + n_pd_box[l_idx]],
                    cam_calib, 
                    position,
                    rotation))
                sum_l += n_pd_box[l_idx]
            loc_pd = torch.cat(loc_pd)
        else:
            pooled_feat = image.new_zeros(1, 128, 7, 7)
            dim = image.new_ones(1, 3)
            rot = image.new_ones(1, 8)
            dep = image.new_zeros(1)
            cen = image.new_zeros(1, 2)
            loc_pd = image.new_zeros(1, 3)

        # Pack infos
        box_output = {'rois': rois_pd,
                      'feat': pooled_feat.detach(),
                      'dim': dim.detach(),
                      'rot': rot.detach(),
                      'dep': dep.detach(),
                      'cen': cen.detach(),
                      'loc': loc_pd.detach(),
                      }

        if phase == 'train':
            loss_dim = nu.compute_dim_loss(dim, dim_gt).unsqueeze(0)
            loss_rot = nu.compute_rot_loss(rot, bin_gt, res_gt).unsqueeze(0)
            loss_dep = nu.compute_dep_loss(dep, dep_gt).unsqueeze(0)
            loss_dep += nu.compute_dep_loss(loc_pd, loc_gt).unsqueeze(0)
            loss_cen = nu.compute_cen_loss(cen, cen_gt).unsqueeze(0)
            targets = (loss_dim, loss_rot, loss_dep, loss_cen)
        else:
            targets = (rois_gt,
                       dim_gt,
                       alpha_gt,
                       dep_gt,
                       cen_gt,
                       loc_gt,
                       ignore,
                       tid_gt)

        return box_output, targets
