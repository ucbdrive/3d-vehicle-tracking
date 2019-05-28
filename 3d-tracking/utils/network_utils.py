import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

best_score = 0.0


def compute_cen_loss(output, target):
    return F.l1_loss(output, target)


def compute_dim_loss(output, target):
    return F.l1_loss(output, target)


def compute_dep_loss(output, target):
    return F.l1_loss(output, target)


def compute_rot_loss(output, target_bin, target_res):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 2) [bin1_res, bin2_res]

    loss_bin1 = F.cross_entropy(output[:, 0:2], target_bin[:, 0])
    loss_bin2 = F.cross_entropy(output[:, 4:6], target_bin[:, 1])
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = F.smooth_l1_loss(valid_output1[:, 2],
                                     torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = F.smooth_l1_loss(valid_output1[:, 3],
                                     torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = F.smooth_l1_loss(valid_output2[:, 6],
                                     torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = F.smooth_l1_loss(valid_output2[:, 7],
                                     torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


def linear_motion_loss(outputs, mask):
    #batch_size = outputs.shape[0]
    s_len = outputs.shape[1]

    loss = outputs.new_zeros(1)
    for idx in range(2, s_len, 1):
        # mask loss to valid outputs
        # motion_mask: (B, 1), the mask of current frame
        motion_mask = mask[:, idx].view(mask.shape[0], 1)

        # Loss: |(loc_t - loc_t-1), (loc_t-1, loc_t-2)|_1 for t = [2, s_len]
        # If loc_t is empty, mask it out by motion_mask
        curr_motion = (outputs[:, idx] - outputs[:, idx - 1]) * motion_mask
        past_motion = (outputs[:, idx - 1] - outputs[:, idx - 2]) * motion_mask
        loss += torch.mean(1.0 - F.cosine_similarity(past_motion, curr_motion))
        loss += F.l1_loss(past_motion, curr_motion)
    return loss / (torch.sum(mask))


def compute_depth_errors(gt, pred):
    assert (np.all(np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)))
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    log_diff = (np.log(gt) - np.log(pred))
    rmse_log = np.sqrt(np.square(log_diff).mean())

    scale_invariant = np.sqrt(
        np.square(log_diff).mean() - np.square(log_diff.mean()))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, scale_invariant, a1, a2, a3


def get_volume(dim):
    # dim: (B, 3)
    return np.prod(dim, axis=1)


def compute_cen(center_gt, center_pd, w, h):
    return (1.0 + np.cos((center_gt - center_pd) / np.array([w, h]))) / 2.0


def compute_dim(dim_gt, dim_pd):
    vol_gt = get_volume(dim_gt)
    vol_pd = get_volume(dim_pd)
    return np.minimum((vol_gt / vol_pd), (vol_pd / vol_gt))


def compute_os(alpha_gt, alpha_pd):
    return (1.0 + np.cos(alpha_gt - alpha_pd)) / 2.0


def get_pred_depth(depth):
    return 1. / depth - 1.


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def freeze_model(model):
    for m in model.modules():
        m.eval()
        for p in m.parameters():
            p.requires_grad = False


def freeze_bn(model, freeze_bn_running=True, freeze_bn_affine=False):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if freeze_bn_running:
                m.eval()  # Freezing Mean/Var of BatchNorm2D.
            if freeze_bn_affine:
                # Freezing Weight/Bias of BatchNorm2D.
                for p in m.parameters():
                    p.requires_grad = False


def load_checkpoint(model, ckpt_path, optimizer=None, is_test=False):
    global best_score
    assert os.path.isfile(ckpt_path), (
        "No checkpoint found at '{}'".format(ckpt_path))
    print("=> Loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    if 'best_score' in checkpoint:
        best_score = checkpoint['best_score']
    if 'optimizer' in checkpoint and optimizer is not None:
        print("=> Loading optimizer state")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except (ValueError) as ke:
            print("Cannot load full model: {}".format(ke))
            if is_test: raise ke

    state = model.state_dict()
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except (RuntimeError, KeyError) as ke:
        print("Cannot load full model: {}".format(ke))
        if is_test: raise ke
        state.update(checkpoint['state_dict'])
        model.load_state_dict(state)
    print("=> Successfully loaded checkpoint '{}' (epoch {})"
          .format(ckpt_path, checkpoint['epoch']))
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(state, use_kitti, session, check_freq):
    global best_score
    ckpt_path = os.path.join(state['save_path'],
                             '{}_gta_checkpoint_latest.pth.tar'.format(session))
    if use_kitti: ckpt_path = ckpt_path.replace('gta', 'kitti')
    torch.save(state, ckpt_path)

    if state['best_score'] > best_score and state['phase'] == 'val':
        best_score = state['best_score']
        best_path = ckpt_path.replace('latest', 'best')
        shutil.copyfile(ckpt_path, best_path)
    if state['epoch'] % check_freq == 0:
        history_path = ckpt_path.replace('latest',
                                         '{:03d}'.format(state['epoch']))
        shutil.copyfile(ckpt_path, history_path)


def adjust_learning_rate(args, optimizer, epoch):
    if args.lr_adjust == 'step':
        """Sets the learning rate to the initial LR decayed by 10
        every 30 epochs"""
        lr = args.lr * (args.step_ratio ** (epoch // args.lr_step))
    else:
        raise ValueError()
    print('Epoch [{}] Learning rate: {:0.6f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_module(layer):
    '''
    Initial modules weights and biases
    '''
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data,
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.uniform_()
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)

def init_lstm_module(layer):
    '''
    Initial LSTM weights and biases
    '''
    for name, param in layer.named_parameters():

        if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0) # initializing the lstm bias with zeros
