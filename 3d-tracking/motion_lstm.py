import argparse
import multiprocessing
import os
import pickle
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import tracking_utils as tu
import network_utils as nu
from utils.config import cfg
from plot_utils import plot_3D
from model.motion_model import LSTM
from model.tracker_model import KalmanBox3dTracker


# CUDA_VISIBLE_DEVICES=0 python motion_lstm.py train --pred_next -j 10 -b 20
# CUDA_VISIBLE_DEVICES=0 python motion_lstm.py test --path
# ./output/623_100_kitti_train_set/ --session 803 --start_epoch 300 --num_epochs 1
# --resume --pred_next -b 1 --is_plot -c

print(torch.__version__)
np.random.seed(777)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='RNN depth motion estimation',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('set', choices=['gta', 'kitti'])
    parser.add_argument('phase', choices=['train', 'test'], 
                        help='Which data split to use in testing')
    parser.add_argument('--split', choices=['train', 'val', 'test'], 
                        default='train',
                        help='Which data split to use in testing')
    parser.add_argument('--path', dest='path',
                        help='path of input info for tracking',
                        default='./output/623_100_kitti_train_set/', type=str)
    parser.add_argument('--cache_name', dest='cache_name',
                        help='path of cache file',
                        default='./output/623_100_kitti_train_traj.pkl', type=str)
    parser.add_argument('--session', dest='session', help='session of tracking',
                        default='804', type=str)
    parser.add_argument('--ckpt_path', dest='ckpt_path',
                        help='path of checkpoint file',
                        default='./checkpoint/{}_{}_{:3d}_linear.pth', type=str)
    parser.add_argument('--start_epoch', default=100, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--n_parts', dest='n_parts',
                        help='separate into n_parts of data',
                        default=1, type=int)
    parser.add_argument('--seq_len', dest='seq_len',
                        help='sequence length feed to model',
                        default=10, type=int)
    parser.add_argument('--min_seq_len', dest='min_seq_len',
                        help='minimum available sequence length',
                        default=10, type=int)
    parser.add_argument('--max_depth', dest='max_depth',
                        help='maximum depth in training',
                        default=150, type=int)
    parser.add_argument('--min_depth', dest='min_depth',
                        help='minimum depth in training',
                        default=0, type=int)
    parser.add_argument('--show_freq', dest='show_freq',
                        help='verbose frequence',
                        default=10, type=int)
    parser.add_argument('--feature_dim', dest='feature_dim',
                        help='feature dimension feed into model',
                        default=64, type=int)
    parser.add_argument('--loc_dim', dest='loc_dim',
                        help='output dimension, we model depth here',
                        default=3, type=int)
    parser.add_argument('--hidden_size', dest='hidden_size',
                        help='hidden size of LSTM',
                        default=128, type=int)
    parser.add_argument('--num_layers', dest='num_layers',
                        help='number of layers of LSTM',
                        default=2, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs',
                        help='number of epochs',
                        default=300, type=int)
    parser.add_argument('--num_seq', dest='num_seq',
                        help='number of seq used in predicting next step',
                        default=5, type=int)
    parser.add_argument('--lr', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-step', help='number of steps to decay lr',
                        default=100, type=int)
    parser.add_argument('--step-ratio', dest='step_ratio', default=0.5,
                        type=float)
    parser.add_argument('--depth_weight', dest='depth_weight',
                        help='weight of depth and smooth loss',
                        default=0.9, type=float)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('-b', '--batch_size', default=10, type=int,
                        help='the batch size on each gpu')
    parser.add_argument('--is_plot', dest='is_plot',
                        help='show prediction result',
                        default=False, action='store_true')
    parser.add_argument('--pred_next', dest='pred_next',
                        help='predict next frame result',
                        default=False, action='store_true')
    parser.add_argument('--resume', dest='resume',
                        help='resume model checkpoint',
                        default=False, action='store_true')
    parser.add_argument('--pretrain', dest='pretrain',
                        help='load pretrain depth checkpoint',
                        default=False, action='store_true')
    parser.add_argument('-c', '--cache', dest='cache',
                        help='using cached trajectories',
                        default=False, action='store_true')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(' '.join(sys.argv))

    return args


class Dataset(Dataset):
    def __init__(self, args, paths):
        self.seq_len = args.seq_len
        self.max_depth = args.max_depth
        self.phase = args.phase
        self.pred_next = args.pred_next
        if args.cache:
            print("Loading {} ...".format(args.cache_name))
            data = pickle.load(open(args.cache_name, 'rb'))
            self.tracking_seqs, self.data_key = data
        else:
            self.tracking_seqs = []
            self.data_key = []
            for pidx, path in enumerate(paths):
                print(pidx, path)
                with open(path, 'rb') as f:
                    sequence = pickle.load(f)
                    self.tracking_seqs.append(self.convert_tracking(
                        (frs for seqs in sequence for frs in seqs)))
                    self.data_key.append(self.sample(self.tracking_seqs[pidx],
                                                     seq_len=args.min_seq_len))
        self.seqs_len = [len(keys) for keys in self.data_key]
        self.accum_len = np.cumsum(self.seqs_len) - 1
        self.data_len = sum(self.seqs_len)

        if not args.cache:
            with open(args.cache_name, 'wb') as f:
                pickle.dump([self.tracking_seqs, self.data_key], f)

    def __getitem__(self, index):
        seq = np.sum(self.accum_len < index)
        fr = index - (self.accum_len[seq] if seq > 0 else 0)
        key = self.data_key[seq][fr]
        return self.dataloader(seq, key)

    def __len__(self):
        return self.data_len

    def dataloader(self, seq, key):
        # Dataloading
        trajectory_seq = self.tracking_seqs[seq][key]
        traj_len = len(trajectory_seq)

        # Random sample data
        if self.phase == 'train':
            idx = np.random.randint(traj_len - self.seq_len)
        if self.phase == 'test':
            idx = 0
        upper_idx = self.seq_len + idx
        data_seq = trajectory_seq[idx:upper_idx]

        # Get data
        rois = np.array([fr[0] for fr in data_seq])
        feat = np.array([fr[1] for fr in data_seq])
        depth_pd = np.array([fr[2] for fr in data_seq])
        alpha_pd = np.array([fr[3] for fr in data_seq])
        dim_pd = np.array([fr[4] for fr in data_seq])
        cen_pd = np.array([fr[5] for fr in data_seq])
        depth_gt = np.array([fr[6] for fr in data_seq])
        alpha_gt = np.array([fr[7] for fr in data_seq])
        dim_gt = np.array([fr[8] for fr in data_seq])
        cen_gt = np.array([fr[9] for fr in data_seq])
        cam_calib = np.array([fr[10].reshape(3, 4) for fr in data_seq])
        cam_loc = np.array([fr[11].flatten() for fr in data_seq])
        cam_rot = np.array([fr[12].flatten() for fr in data_seq])
        pose = [tu.Pose(fr[11].flatten(), fr[12].flatten()) for fr in data_seq]

        objs_pd = np.array([tu.point3dcoord(
            cen_pd[i], 
            depth_pd[i], 
            cam_calib[i], 
            pose[i]) for i in range(len(depth_pd))]).reshape(-1, 3)

        # objects center in the world coordinates
        objs_gt = np.array([tu.point3dcoord(
            cen_gt[i], 
            depth_gt[i], 
            cam_calib[i], 
            pose[i]) for i in range(len(depth_gt))]).reshape(-1, 3)

        # Predict next
        if self.pred_next:
            seq_len = self.seq_len - 1
            rois = rois[:-1]
            feat = feat[:-1]
            depth_pd = depth_pd[:-1]
            alpha_pd = alpha_pd[:-1]
            dim_pd = dim_pd[:-1]
            cen_pd = cen_pd[:-1]
            depth_gt = depth_gt[1:]
            alpha_gt = alpha_gt[1:]
            dim_gt = dim_gt[1:]
            cen_gt = cen_gt[1:]
            cam_rot = cam_rot[1:]
            cam_loc = cam_loc[1:]
        else:
            seq_len = self.seq_len

        # Padding
        valid_mask = np.zeros(seq_len)
        valid_mask[:np.sum(rois[:, 4] > 0)] = 1
        rois = np.vstack([rois, np.zeros([seq_len, 5])])[:seq_len]
        feat = np.vstack([feat, np.zeros([seq_len, 128])])[:seq_len]
        # feat = np.vstack([feat, np.zeros([seq_len, 128, 7, 7])])[:seq_len]
        objs_pd = np.vstack([objs_pd, np.zeros([self.seq_len, 3])])[
                  :self.seq_len]
        objs_gt = np.vstack([objs_gt, np.zeros([self.seq_len, 3])])[
                  :self.seq_len]
        cam_rot = np.vstack([cam_rot, np.zeros([seq_len, 3])])[:seq_len]
        cam_loc = np.vstack([cam_loc, np.zeros([seq_len, 3])])[:seq_len]
        cen_pd = np.vstack([cen_pd, np.zeros([seq_len, 2])])[:seq_len]
        cen_gt = np.vstack([cen_gt, np.zeros([seq_len, 2])])[:seq_len]
        dim_pd = np.vstack([dim_pd, np.zeros([seq_len, 3])])[:seq_len]
        dim_gt = np.vstack([dim_gt, np.zeros([seq_len, 3])])[:seq_len]
        depth_pd = np.hstack([depth_pd, np.zeros([seq_len])])[:seq_len]
        depth_gt = np.hstack([depth_gt, np.zeros([seq_len])])[:seq_len]
        alpha_pd = np.hstack([alpha_pd, np.zeros([seq_len])])[:seq_len]
        alpha_gt = np.hstack([alpha_gt, np.zeros([seq_len])])[:seq_len]

        # Torch tensors
        traj_out = {
            'inputs': torch.from_numpy(feat).float(),
            'loc_gt': torch.from_numpy(objs_gt).float(),
            'loc_pd': torch.from_numpy(objs_pd).float(),
            'depth_gt': torch.from_numpy(depth_gt).float(),
            'depth_pd': torch.from_numpy(depth_pd).float(),
            'alpha_gt': torch.from_numpy(alpha_gt).float(),
            'alpha_pd': torch.from_numpy(alpha_pd).float(),
            'dim_gt': torch.from_numpy(dim_gt).float(),
            'dim_pd': torch.from_numpy(dim_pd).float(),
            'cen_gt': torch.from_numpy(cen_gt).float(),
            'cen_pd': torch.from_numpy(cen_pd).float(),
            'cam_rot': torch.from_numpy(cam_rot).float(),
            'cam_loc': torch.from_numpy(cam_loc).float(),
            'valid_mask': torch.from_numpy(valid_mask).float()
        }
        return traj_out

    def convert_tracking_gt(self, data):
        tracking_dict = {}
        for fr_idx, frame in enumerate(data):
            for box_idx in range(len(frame['tid_gt'])):
                tid = int(frame['tid_gt'][box_idx])
                # If not ignore
                # Get rois, feature, depth, depth_gt, cam_rot, cam_trans
                tid_data = {
                        'depth_gt': frame['depth_gt'][box_idx],
                        'alpha_gt': frame['alpha_gt'][box_idx],
                        'dim_gt': frame['dim_gt'][box_idx],
                        'center_gt': frame['center_gt'][box_idx],
                        'loc_gt': frame['loc_gt'][box_idx],
                        'cam_calib': frame['cam_calib'],
                        'cam_rot': frame['cam_rot'],
                        'cam_loc': frame['cam_loc'],
                        'fr_idx': fr_idx}

                if frame['ignore'][box_idx] \
                    or frame['depth_gt'][box_idx] > self.max_depth:
                    tracking_dict[tid] = tid_data

                if tid not in tracking_dict:
                    tracking_dict[tid] = [tid_data]
                elif fr_idx == tracking_dict[tid][-1][-1] + 1:
                    tracking_dict[tid].append(tid_data)

        return tracking_dict

    def convert_tracking(self, data):
        tracking_dict = {}
        for fr_idx, frame in enumerate(data):
            if len(frame['rois_gt']) == 0 or len(frame['rois_pd']) == 0:
                print("Skip")
                continue
            iou, idx, valid = tu.get_iou(frame['rois_gt'], frame['rois_pd'],
                                         0.85)
            valid = valid.flatten()
            rois_pd = frame['rois_pd'][idx]  # [valid]
            depth_pd = frame['depth_pd'][idx]  # [valid]
            center_pd = frame['center_pd'][idx]  # [valid]
            alpha_pd = frame['alpha_pd'][idx]  # [valid]
            dim_pd = frame['dim_pd'][idx]  # [valid]
            feature = frame['feature'][idx]  # [valid]
            # feature = frame['feature'].reshape(-1, 128, 7, 7)[idx]#[valid]
            for box_idx in range(len(frame['tid_gt'])):
                tid = int(frame['tid_gt'][box_idx])
                # If not ignores
                # Get rois, feature, depth, depth_gt, cam_rot, cam_trans
                if not frame['ignore'][box_idx] \
                        and valid[box_idx] \
                        and rois_pd[box_idx][4] != 0 \
                        and frame['depth_gt'][box_idx] < self.max_depth:

                    tid_data = [rois_pd[box_idx],
                               feature[box_idx],
                               depth_pd[box_idx],
                               alpha_pd[box_idx],
                               dim_pd[box_idx],
                               center_pd[box_idx],
                               frame['depth_gt'][box_idx],
                               frame['alpha_gt'][box_idx],
                               frame['dim_gt'][box_idx],
                               frame['center_gt'][box_idx],
                               frame['cam_calib'],
                               frame['cam_loc'],
                               frame['cam_rot'],
                               fr_idx,
                                               ]
                    # If tid not in the existed dict
                    # Or box_idx is not consecutive
                    if tid not in tracking_dict:
                        tracking_dict[tid] = [tid_data]
                    elif fr_idx <= tracking_dict[tid][-1][-1] + 5:
                        tracking_dict[tid].append(tid_data)

        return tracking_dict

    def sample(self, data, seq_len=10):
        datakey = []
        for key in list(data):
            if len(data[key]) > seq_len:
                datakey.append(key)
        return datakey


def train(args):
    model = LSTM(args.batch_size,
                 args.feature_dim,
                 args.hidden_size,
                 args.num_layers,
                 args.loc_dim).to(args.device)
    model = model.train()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True
    )

    if args.resume:
        nu.load_checkpoint(model, 
                        args.ckpt_path.format(args.session, args.set, args.start_epoch),
                        optimizer=optimizer)

    if args.path.endswith('_bdd_roipool_output.pkl'):
        paths = [args.path]
        dsize = len(paths)
        n_parts = 1
    else:
        paths = sorted(
            [os.path.join(args.path, n) for n in os.listdir(args.path)
             if n.endswith('_bdd_roipool_output.pkl')])
        dsize = len(paths) // args.n_parts
        n_parts = args.n_parts

    print("Total {} sequences separate into {} parts".format(dsize, n_parts))

    dataset = Dataset(args, paths[:dsize])

    print("Number of trajectories to train: {}".format(dataset.__len__()))

    # Data loading code
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # Start iterations
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        losses = tu.AverageMeter()
        pred_losses = tu.AverageMeter()
        refine_losses = tu.AverageMeter()
        lin_losses = tu.AverageMeter()
        nu.adjust_learning_rate(args, optimizer, epoch)

        for part in range(n_parts):

            for iters, traj_out in enumerate(iter(train_loader)):

                # Initial
                inputs = traj_out['inputs'].to(args.device)
                loc_gt = traj_out['loc_gt'].to(args.device)[:, 1:]
                loc_obs = 0.05 *traj_out['loc_pd'].to(args.device) + \
                            0.95 * traj_out['loc_gt'].to(args.device)
                depth_gt = traj_out['depth_gt'].to(args.device)
                depth_pd = traj_out['depth_pd'].to(args.device)
                alpha_gt = traj_out['alpha_gt'].to(args.device)
                alpha_pd = traj_out['alpha_pd'].to(args.device)
                cen_gt = traj_out['cen_gt'].to(args.device)
                cen_pd = traj_out['cen_pd'].to(args.device)
                dim_gt = traj_out['dim_gt'].to(args.device)
                dim_pd = traj_out['dim_pd'].to(args.device)
                cam_loc = traj_out['cam_loc'].to(args.device)
                cam_rot = traj_out['cam_rot'].to(args.device)
                valid_mask = traj_out['valid_mask'].to(args.device)

                loc_preds = []
                loc_refines = []

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                hidden_predict = model.init_hidden(args.device)  # None
                hidden_refine = model.init_hidden(args.device)  # None


                # Generate a history of location
                vel_history = loc_obs.new_zeros(args.num_seq, loc_obs.shape[0], 3)
                for i in range(valid_mask.shape[1]):

                    # Force input update or use predicted
                    if i == 0:
                        loc_refine = loc_obs[:, i]
                    # Update history
                    loc_pred, hidden_predict = model.predict(vel_history,
                                                             loc_refine,
                                                             hidden_predict)
                    vel_history = torch.cat([vel_history[1:], (loc_obs[:, i+1] - loc_pred).unsqueeze(0)], dim=0)
                    loc_refine, hidden_refine = model.refine(loc_pred,
                                                             loc_obs[:,
                                                             i + 1],
                                                             hidden_refine)
                    print(vel_history[:, 0, :].cpu().detach().numpy(), 
                        loc_pred[0, :].cpu().detach().numpy(), 
                        loc_refine[0, :].cpu().detach().numpy(), 
                        loc_gt[0, i, :].cpu().detach().numpy())
                    # Predict residual of depth
                    loc_preds.append(loc_pred)
                    loc_refines.append(loc_refine)

                loc_preds = torch.cat(loc_preds, dim=1).view(
                    valid_mask.shape[0], -1, 3)
                loc_refines = torch.cat(loc_refines, dim=1).view(
                    valid_mask.shape[0], -1, 3)

                loc_preds = loc_preds * valid_mask.unsqueeze(2)
                loc_refines = loc_refines * valid_mask.unsqueeze(2)

                # Cost functions
                pred_loss = F.l1_loss(loc_preds,
                                      loc_gt * valid_mask.unsqueeze(2),
                                      reduction='sum') / torch.sum(
                    valid_mask)
                refine_loss = F.l1_loss(loc_refines,
                                        loc_gt * valid_mask.unsqueeze(2),
                                        reduction='sum') / torch.sum(
                    valid_mask)
                dep_loss = (pred_loss + refine_loss)
                linear_loss = nu.linear_motion_loss(loc_preds, valid_mask)
                linear_loss += nu.linear_motion_loss(loc_refines, valid_mask)

                loss = (args.depth_weight * dep_loss +
                        (1.0 - args.depth_weight) * linear_loss)
                # / torch.sum(valid_mask)

                # Clear the states of model parameters each time
                optimizer.zero_grad()

                # BP loss
                loss.backward()

                # Clip if the gradients explode
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()

                # Updates
                losses.update(loss.data.cpu().numpy().item(),
                              int(torch.sum(valid_mask)))
                pred_losses.update(pred_loss.data.cpu().numpy().item(),
                                   int(torch.sum(valid_mask)))
                refine_losses.update(refine_loss.data.cpu().numpy().item(),
                                     int(torch.sum(valid_mask)))
                lin_losses.update(linear_loss.data.cpu().numpy().item(),
                                  int(torch.sum(valid_mask)))

                # Verbose
                if iters % args.show_freq == 0 and iters != 0:
                    print('[{NAME} - {SESS}] Epoch: [{EP}][{IT}/{TO}] '
                          'Loss {loss.val:.4f} ({loss.avg:.3f}) '
                          'P-Loss {pred.val:.2f} ({pred.avg:.2f}) '
                          'R-Loss {refine.val:.2f} ({refine.avg:.2f}) '
                          'S-Loss {smooth.val:.2f} ({smooth.avg:.2f}) '.format(
                        NAME=args.set.upper(),
                        SESS=args.session, EP=epoch,
                        IT=iters, TO=len(train_loader),
                        loss=losses, pred=pred_losses,
                        refine=refine_losses, smooth=lin_losses))
                    print("PD: {pd} OB: {obs} RF: {ref} GT: {gt}".format(
                        pd=loc_preds.cpu().data.numpy().astype(int)[0, 2],
                        obs=loc_obs.cpu().data.numpy().astype(int)[0, 3],
                        ref=loc_refines.cpu().data.numpy().astype(int)[0, 2],
                        gt=loc_gt.cpu().data.numpy().astype(int)[0, 2]))

                    if args.is_plot:
                        plot_3D('{}_{}'.format(epoch, iters), args.session,
                                cam_loc.cpu().data.numpy()[0],
                                loc_gt.cpu().data.numpy()[0],
                                loc_preds.cpu().data.numpy()[0],
                                loc_refines.cpu().data.numpy()[0])

            # Save
            if epoch != args.start_epoch:
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'session': args.session},
                           args.ckpt_path.format(args.session, args.set, epoch))

            print(
                "Epoch [{}] Loss: {:.3f} P-Loss: {:.3f} R-Loss: {:.3f} "
                "S-Loss: {:.3f} ".format(
                    epoch,
                    losses.avg,
                    pred_losses.avg,
                    refine_losses.avg,
                    lin_losses.avg))

            if n_parts != 1:
                dataset = Dataset(args,
                                  paths[part * dsize:part * dsize + dsize])

                print("Number of trajectories to train: {}".format(
                    dataset.__len__()))

                # Data loading code
                train_loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)


def test(args):
    model = LSTM(args.batch_size,
                 args.feature_dim,
                 args.hidden_size,
                 args.num_layers,
                 args.loc_dim).to(args.device)
    model = model.eval()

    if args.resume:
        nu.load_checkpoint(model, 
                        args.ckpt_path.format(args.session, args.set, args.start_epoch),
                        is_test=True)

    if args.path.endswith('_bdd_roipool_output.pkl'):
        paths = [args.path]
        dsize = len(paths)
        n_parts = 1
    else:
        paths = sorted(
            [os.path.join(args.path, n) for n in os.listdir(args.path)
             if n.endswith('_bdd_roipool_output.pkl')])
        dsize = len(paths) // args.n_parts
        n_parts = args.n_parts

    print("Total {} sequences separate into {} parts".format(dsize, n_parts))

    dataset = Dataset(args, paths[:dsize])

    print("Number of trajectories to test: {}".format(dataset.__len__()))

    # Data loading code
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # Start iterations
    losses = tu.AverageMeter()
    pred_losses = tu.AverageMeter()
    refine_losses = tu.AverageMeter()
    lin_losses = tu.AverageMeter()
    losses_kf = tu.AverageMeter()
    pred_losses_kf = tu.AverageMeter()
    refine_losses_kf = tu.AverageMeter()
    lin_losses_kf = tu.AverageMeter()

    epoch = args.start_epoch

    for part in range(n_parts):
        for iters, traj_out in enumerate(iter(test_loader)):

            # Initial
            inputs = traj_out['inputs'].to(args.device)
            loc_gt = traj_out['loc_gt'].to(args.device)[:, 1:]
            loc_obs = traj_out['loc_pd'].to(args.device)
            depth_gt = traj_out['depth_gt'].to(args.device)
            depth_pd = traj_out['depth_pd'].to(args.device)
            alpha_gt = traj_out['alpha_gt'].to(args.device)
            alpha_pd = traj_out['alpha_pd'].to(args.device)
            cen_gt = traj_out['cen_gt'].to(args.device)
            cen_pd = traj_out['cen_pd'].to(args.device)
            dim_gt = traj_out['dim_gt'].to(args.device)
            dim_pd = traj_out['dim_pd'].to(args.device)
            cam_loc = traj_out['cam_loc'].to(args.device)
            cam_rot = traj_out['cam_rot'].to(args.device)
            valid_mask = traj_out['valid_mask'].to(args.device)

            loc_preds = []
            loc_refines = []
            loc_preds_kf = []
            loc_refines_kf = []

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            hidden_predict = model.init_hidden(args.device)  # None
            hidden_refine = model.init_hidden(args.device)  # None


            # Generate a history of location
            vel_history = loc_obs.new_zeros(args.num_seq, loc_obs.shape[0], 3)
            for i in range(valid_mask.shape[1]):

                # Force input update or use predicted
                if i == 0:
                    loc_refine = loc_obs[:, i]
                    trk = KalmanBox3dTracker(loc_refine.cpu().detach().numpy())
                # Update history
                vel_history = torch.cat([vel_history[1:], (loc_obs[:, i] - loc_refine).unsqueeze(0)], dim=0)
                loc_pred, hidden_predict = model.predict(vel_history,
                                                         loc_refine,
                                                         hidden_predict)
                loc_refine, hidden_refine = model.refine(loc_pred,
                                                         loc_obs[:,
                                                         i + 1],
                                                         hidden_refine)

                loc_pred_kf = trk.predict().squeeze()
                trk.update(loc_obs[:, i+1].cpu().detach().numpy())
                loc_refine_kf = trk.get_state()[:3]

                # Predict residual of depth
                loc_preds.append(loc_pred)
                loc_refines.append(loc_refine)
                loc_preds_kf.append(loc_pred_kf)
                loc_refines_kf.append(loc_refine_kf)

            loc_preds = torch.cat(loc_preds, dim=1).view(valid_mask.shape[0],
                                                         -1, 3)
            loc_refines = torch.cat(loc_refines, dim=1).view(
                valid_mask.shape[0], -1, 3)
            loc_preds_kf = valid_mask.new(loc_preds_kf).view(valid_mask.shape[0],
                                                         -1, 3)
            loc_refines_kf = valid_mask.new(loc_refines_kf).view(
                valid_mask.shape[0], -1, 3)

            loc_preds = loc_preds * valid_mask.unsqueeze(2)
            loc_refines = loc_refines * valid_mask.unsqueeze(2)
            loc_preds_kf = loc_preds_kf * valid_mask.unsqueeze(2)
            loc_refines_kf = loc_refines_kf * valid_mask.unsqueeze(2)

            # Cost functions
            pred_loss = F.l1_loss(loc_preds, loc_gt * valid_mask.unsqueeze(2),
                                  reduction='sum') / torch.sum(valid_mask)
            refine_loss = F.l1_loss(loc_refines,
                                    loc_gt * valid_mask.unsqueeze(2),
                                    reduction='sum') / torch.sum(
                valid_mask)
            pred_loss_kf = F.l1_loss(loc_preds_kf, loc_gt * valid_mask.unsqueeze(2),
                                  reduction='sum') / torch.sum(valid_mask)
            refine_loss_kf = F.l1_loss(loc_refines_kf,
                                    loc_gt * valid_mask.unsqueeze(2),
                                    reduction='sum') / torch.sum(
                valid_mask)
            dep_loss = pred_loss + refine_loss
            linear_loss = nu.linear_motion_loss(loc_preds, valid_mask)
            linear_loss += nu.linear_motion_loss(loc_refines, valid_mask)
            dep_loss_kf = pred_loss_kf + refine_loss_kf
            linear_loss_kf = nu.linear_motion_loss(loc_preds_kf, valid_mask)
            linear_loss_kf += nu.linear_motion_loss(loc_refines_kf, valid_mask)

            loss = (args.depth_weight * dep_loss +
                    (1.0 - args.depth_weight) * linear_loss)  # /
            # torch.sum(valid_mask)
            loss_kf = (args.depth_weight * dep_loss_kf +
                    (1.0 - args.depth_weight) * linear_loss_kf)  # /
            # torch.sum(valid_mask)

            # Updates
            losses.update(loss.data.cpu().numpy().item(),
                          int(torch.sum(valid_mask)))
            pred_losses.update(pred_loss.data.cpu().numpy().item(),
                               int(torch.sum(valid_mask)))
            refine_losses.update(refine_loss.data.cpu().numpy().item(),
                                 int(torch.sum(valid_mask)))
            lin_losses.update(linear_loss.data.cpu().numpy().item(),
                              int(torch.sum(valid_mask)))
            # Updates
            losses_kf.update(loss_kf.data.cpu().numpy().item(),
                          int(torch.sum(valid_mask)))
            pred_losses_kf.update(pred_loss_kf.data.cpu().numpy().item(),
                               int(torch.sum(valid_mask)))
            refine_losses_kf.update(refine_loss_kf.data.cpu().numpy().item(),
                                 int(torch.sum(valid_mask)))
            lin_losses_kf.update(linear_loss_kf.data.cpu().numpy().item(),
                              int(torch.sum(valid_mask)))

            # Verbose
            if iters % args.show_freq == 0 and iters != 0:
                print('[{NAME} - {SESS}] Epoch: [{EP}][{IT}/{TO}] '
                      'Loss {loss.val:.4f} ({loss.avg:.3f}) '
                      'P-Loss {pred.val:.2f} ({pred.avg:.2f}) '
                      'R-Loss {refine.val:.2f} ({refine.avg:.2f}) '
                      'S-Loss {smooth.val:.2f} ({smooth.avg:.2f}) \n'
                      'Loss {loss_kf.val:.4f} ({loss_kf.avg:.3f}) '
                      'P-Loss {pred_kf.val:.2f} ({pred_kf.avg:.2f}) '
                      'R-Loss {refine_kf.val:.2f} ({refine_kf.avg:.2f}) '
                      'S-Loss {smooth_kf.val:.2f} ({smooth_kf.avg:.2f}) '.format(
                    NAME=args.set.upper(),
                    SESS=args.session, EP=epoch,
                    IT=iters, TO=len(test_loader),
                    loss=losses, pred=pred_losses,
                    refine=refine_losses, smooth=lin_losses,
                    loss_kf=losses_kf, pred_kf=pred_losses_kf,
                    refine_kf=refine_losses_kf, smooth_kf=lin_losses_kf))
                print("PD: {pd} OB: {obs} RF: {ref} GT: {gt} \n"
                      "PDKF: {pdkf} OBKF: {obs} RFKF: {refkf} GT: {gt}".format(
                    pd=loc_preds.cpu().data.numpy().astype(int)[0, 0],
                    obs=loc_obs.cpu().data.numpy().astype(int)[0, 1],
                    ref=loc_refines.cpu().data.numpy().astype(int)[0, 0],
                    pdkf=loc_preds_kf.cpu().data.numpy().astype(int)[0, 0],
                    refkf=loc_refines_kf.cpu().data.numpy().astype(int)[0, 0],
                    gt=loc_gt.cpu().data.numpy().astype(int)[0, 0]))

                if args.is_plot:
                    plot_3D('{}_{}'.format(epoch, iters), args.session,
                            cam_loc.cpu().data.numpy()[0],
                            loc_gt.cpu().data.numpy()[0],
                            loc_preds.cpu().data.numpy()[0],
                            loc_refines.cpu().data.numpy()[0])
                    plot_3D('{}_{}_kf'.format(epoch, iters), args.session,
                            cam_loc.cpu().data.numpy()[0],
                            loc_gt.cpu().data.numpy()[0],
                            loc_preds_kf.cpu().data.numpy()[0],
                            loc_refines_kf.cpu().data.numpy()[0])

            print('[{NAME} - {SESS}] Epoch: [{EP}][{IT}/{TO}] '
                  'Loss {loss.val:.4f} ({loss.avg:.3f}) '
                  'P-Loss {pred.val:.2f} ({pred.avg:.2f}) '
                  'R-Loss {refine.val:.2f} ({refine.avg:.2f}) '
                  'S-Loss {smooth.val:.2f} ({smooth.avg:.2f}) \n'
                  'Loss {loss_kf.val:.4f} ({loss_kf.avg:.3f}) '
                  'P-Loss {pred_kf.val:.2f} ({pred_kf.avg:.2f}) '
                  'R-Loss {refine_kf.val:.2f} ({refine_kf.avg:.2f}) '
                  'S-Loss {smooth_kf.val:.2f} ({smooth_kf.avg:.2f}) '.format(
                NAME=args.set.upper(),
                SESS=args.session, EP=epoch,
                IT=iters, TO=len(test_loader),
                loss=losses, pred=pred_losses,
                refine=refine_losses, smooth=lin_losses,
                loss_kf=losses_kf, pred_kf=pred_losses_kf,
                refine_kf=refine_losses_kf, smooth_kf=lin_losses_kf))
            print("PD: {pd} OB: {obs} RF: {ref} GT: {gt} \n"
                  "PDKF: {pdkf} OBKF: {obs} RFKF: {refkf} GT: {gt}".format(
                pd=loc_preds.cpu().data.numpy().astype(int)[0, 0],
                obs=loc_obs.cpu().data.numpy().astype(int)[0, 1],
                ref=loc_refines.cpu().data.numpy().astype(int)[0, 0],
                pdkf=loc_preds_kf.cpu().data.numpy().astype(int)[0, 0],
                refkf=loc_refines_kf.cpu().data.numpy().astype(int)[0, 0],
                gt=loc_gt.cpu().data.numpy().astype(int)[0, 0]))

        if n_parts != 1:
            dataset = Dataset(args, paths[part * dsize:part * dsize + dsize])

            print(
                "Number of trajectories to test: {}".format(dataset.__len__()))

            # Data loading code
            test_loader = DataLoader(
                dataset,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, drop_last=True)


if __name__ == '__main__':
    args = parse_args()
    assert args.pred_next is True, "Need to predict next frame for motion " \
                                   "estimation in new setting"
    torch.set_num_threads(multiprocessing.cpu_count())
    cudnn.benchmark = True

    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        assert args.batch_size == 1
        test(args)
