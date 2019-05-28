import argparse
import multiprocessing
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils.network_utils as nu
import utils.tracking_utils as tu
from utils.config import cfg
from loader.dataset import Dataset

# System verbose
print(torch.__version__)
print(' '.join(sys.argv))


def parse_args():
    parser = argparse.ArgumentParser(description='Monocular 3D Estimation',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('set', choices=['gta', 'kitti'])
    parser.add_argument('phase', choices=['train', 'test'])
    parser.add_argument('--data_split', choices=['train', 'val', 'test'], 
                        default='val',
                        help='Which data split to use in testing')
    parser.add_argument('--session', default='100',
                        help='Name of the session, to separate exp')
    parser.add_argument('--track_name', default=None,
                        help='Name of the result for tracking')
    parser.add_argument('--is_tracking', action='store_true', default=False,
                        help='using KITTI detection or tracking dataset')
    parser.add_argument('--use_tfboard', action='store_true', default=False,
                        help='log results using tfboard')
    parser.add_argument('--has_val', action='store_true', default=False,
                        help='training with validation set')
    parser.add_argument('--is_normalizing', action='store_true', default=False,
                        help='normalize input image by mean and var')
    parser.add_argument('--percent', default=100, choices=[1, 10, 25, 50, 100],
                        type=int,
                        help='how much data used in training [1, 10, 25, 50, '
                             '100]')
    parser.add_argument('--adaptBN', action='store_true', default=False,
                        help='use adaptBN before testing')
    parser.add_argument('--roi_name', default='roialign',
                        choices=['roipool', 'roialign'],
                        help='roipooling methods: roipool, roialign (default: '
                             'roialign)')
    parser.add_argument('--down_ratio', default=8, type=int,
                        help='pooling downsample ratio. default: 8')
    parser.add_argument('--roi_kernel', default=7, type=int,
                        help='roi kernel size. default: 7')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='dla34up',
                        choices=['dla34', 'dla34up'],
                        help='model architecture: dla34, dla34up (default: '
                             'dla34up)')
    parser.add_argument('--n_box_limit', default=20, type=int, metavar='N',
                        help='number of boxes to use in one frame (default: '
                             '30)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--optim', type=str, default='adam',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-step', default=30, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--check-freq', default=1, type=int,
                        help='saving ckpt frequency (default: 1)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--json_path', dest='json_path', default=None,
                        help='path to json files')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='the batch size on each gpu')
    parser.add_argument('--lr-adjust', dest='lr_adjust',
                        choices=['step'], default='step')
    parser.add_argument('--step-ratio', dest='step_ratio', default=0.5,
                        type=float)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--depth_weight', dest='depth_weight', default=0.1,
                        type=float)
    parser.add_argument('--min_depth', dest='min_depth', default=0.0,
                        type=float)
    parser.add_argument('--max_depth', dest='max_depth', default=150.0,
                        type=float)
    parser.add_argument('--note', dest='note',
                        default='Tell me before kill it, thanks.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def run_training(model, args):

    if not os.path.isdir(cfg.CHECKPOINT_PATH):
        os.mkdir(cfg.CHECKPOINT_PATH)

    cudnn.benchmark = True

    if args.has_val:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Data loading code
    train_loader = {
        phase: DataLoader(
            Dataset(args.json_path, 
                        phase,
                        phase,
                        args.set == 'kitti',
                        args.percent,
                        args.is_tracking,
                        args.is_normalizing,
                        args.n_box_limit
                        ),
            batch_size=args.batch_size,
            shuffle=(phase == 'train'),
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)
        for phase in phases
    }

    # Optimizer
    if args.adaptBN:
        model_param = list()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                model_param.append({'params': list(m.parameters())})
        lr = 0.0
    else:
        # model_param = model.parameters()
        model_param = filter(lambda p: p.requires_grad, model.parameters())
        lr = args.lr

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model_param, lr,
                                    momentum=args.momentum, nesterov=True,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model_param, lr,
                                     weight_decay=args.weight_decay,
                                     amsgrad=True)

    # optionally resume from a checkpoint
    if args.resume:
        nu.load_checkpoint(model, args.resume, optimizer=optimizer)

    # switch to train mode
    if args.adaptBN:
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                for p in m.parameters():
                    p.requires_grad = False

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")
    else:
        logger = None

    for epoch in range(args.start_epoch, args.epochs):
        # Resume with normal lr adjust 
        # not to over suppress the lr due to large epoch
        # if args.optim == 'sgd':
        nu.adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        for phase in phases:
            if phase == 'train':
                model.train()
                # nu.freeze_model(model.module.base)
                # nu.freeze_model(model.module.rot)
                # nu.freeze_model(model.module.dep)
                # nu.freeze_model(model.module.dim)
                train_model(args, train_loader[phase], model, optimizer, epoch,
                            phase, logger)
                acc = 0.0
            else:
                model.eval()
                acc = val_model(args, train_loader[phase], model, epoch, phase,
                                logger)

            # Save checkpoint
            nu.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': acc,
                'phase': phase,
                'save_path': cfg.CHECKPOINT_PATH,
            }, args.set == 'kitti', args.session, args.check_freq)

    if args.use_tfboard:
        logger.close()

    print("Training finished!!")


def train_model(args, train_loader, model, optimizer, epoch, phase, logger):
    batch_time = tu.AverageMeter()
    data_time = tu.AverageMeter()
    losses = tu.AverageMeter()
    losses_dim = tu.AverageMeter()
    losses_rot = tu.AverageMeter()
    losses_dep = tu.AverageMeter()
    losses_cen = tu.AverageMeter()
    data_size = len(train_loader)

    end = time.time()
    for i, (image, box_info) in enumerate(iter(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()

        # track history if only in train
        _, losses_ = model(image, box_info, args.device, phase)

        loss_dim = torch.mean(losses_[0])
        loss_rot = torch.mean(losses_[1])
        loss_dep = torch.mean(losses_[2])
        loss_cen = torch.mean(losses_[3])

        loss = loss_dim + loss_rot + loss_dep * args.depth_weight + loss_cen

        # measure accuracy and record loss
        losses.update(loss.cpu().data.numpy().item())
        losses_dim.update(loss_dim.cpu().data.numpy().item())
        losses_rot.update(loss_rot.cpu().data.numpy().item())
        losses_dep.update(loss_dep.cpu().data.numpy().item())
        losses_cen.update(loss_cen.cpu().data.numpy().item())

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if args.use_tfboard:
            loss_info = {
                'L_all': losses.avg,
                'L_dim': losses_dim.avg,
                'L_rot': losses_rot.avg,
                'L_dep': losses_dep.avg,
                'L_cen': losses_cen.avg,
            }
            logger.add_scalars("loss_{}/".format(args.session),
                               loss_info,
                               epoch * data_size + i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{NAME} - {SESS} - {PHASE}][{EP}][{IT}/{TO}] '
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) '
                  'Data {data_time.val:.2f} ({data_time.avg:.2f}) '
                  'Loss {loss.val:.3f} ({loss.avg:.3f}) '
                  'Dim {dim.val:.3f} ({dim.avg:.3f}) '
                  'Alpha {alpha.val:.3f} ({alpha.avg:.3f}) '
                  'Depth {depth.val:.3f} ({depth.avg:.3f}) '
                  'Center {center.val:.3f} ({center.avg:.3f}) '.format(
                NAME=args.set.upper(),
                PHASE=phase, SESS=args.session, EP=epoch,
                IT=i, TO=data_size, batch_time=batch_time,
                data_time=data_time, loss=losses, dim=losses_dim,
                alpha=losses_rot, depth=losses_dep, center=losses_cen))


def val_model(args, val_loader, model, epoch, phase, logger):
    aos_meter = tu.AverageMeter()
    dim_meter = tu.AverageMeter()
    cen_meter = tu.AverageMeter()
    dm = []
    name_line = "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, " \
                "{:>10}, {:>10}, {:>10}".format(
        'abs_rel', 'sq_rel', 'rms', 'log_rms', \
        'a1', 'a2', 'a3', 'AOS', 'DIM', 'CEN')

    for i, (image, box_info) in enumerate(iter(val_loader)):

        with torch.no_grad():
            box_output, targets = model(image, box_info, args.device, 'test')

        rois_gt, \
        dim_gt_, \
        alpha_gt_, \
        dep_gt_, \
        cen_gt_, \
        loc_gt_, \
        ignore_, \
        tid_gt = targets

        box_gt = rois_gt.cpu().data.numpy()
        box_pd = box_output['rois'].cpu().data.numpy()
        dim_gt = dim_gt_.cpu().data.numpy()
        dim_pd = box_output['dim'].cpu().data.numpy()
        alpha_gt = alpha_gt_.cpu().data.numpy()
        alpha_pd = nu.get_alpha(box_output['rot'].cpu().data.numpy())
        depth_gt = dep_gt_.cpu().data.numpy()
        depth_pd = box_output['dep'].cpu().data.numpy()
        center_gt = cen_gt_.cpu().data.numpy()
        center_pd = box_output['cen'].cpu().data.numpy()

        if len(box_gt) > 0:
            iou, idx, valid = tu.get_iou(box_gt, box_pd[:, :4], 0.85)
        else:
            valid = np.array([False])
        if valid.any():
            box_pd_v = box_pd[idx]
            alpha_pd_v = alpha_pd[idx]
            dim_pd_v = dim_pd[idx]
            depth_pd_v = depth_pd[idx]
            center_pd_v = center_pd[idx]

            aos_meter.update(np.mean(nu.compute_os(alpha_gt, alpha_pd_v)),
                             alpha_gt.shape[0])
            dim_meter.update(np.mean(nu.compute_dim(dim_gt, dim_pd_v)),
                             dim_gt.shape[0])
            w = (box_pd_v[:, 2:3] - box_pd_v[:, 0:1] + 1)
            h = (box_pd_v[:, 3:4] - box_pd_v[:, 1:2] + 1)
            cen_meter.update(
                np.mean(nu.compute_cen(center_gt, center_pd_v, w, h)),
                center_gt.shape[0])

            # Avoid zero in calculating a1, a2, a3
            mask = np.logical_and(depth_gt > args.min_depth,
                                  depth_gt < args.max_depth)
            mask = np.logical_and(mask, depth_pd_v > 0)
            if mask.any():
                dm.append(
                    nu.compute_depth_errors(depth_gt[mask], depth_pd_v[mask]))
            else:
                print("Not a valid depth range in GT")

        if i % 100 == 0 and i != 0:
            depth_metrics = np.mean(dm, axis=0)
            data_line = "{:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, " \
                        "{:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, " \
                        "{:10.3f}".format(
                depth_metrics[0].mean(), depth_metrics[1].mean(), \
                depth_metrics[2].mean(), depth_metrics[3].mean(), \
                depth_metrics[5].mean(), depth_metrics[6].mean(), \
                depth_metrics[7].mean(), \
                aos_meter.avg, dim_meter.avg, cen_meter.avg)
            print(i)
            print(name_line)
            print(data_line)

    print("Validation Result:")
    depth_metrics = np.mean(dm, axis=0)
    data_line = "{:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, " \
                "{:10.3f}, {:10.3f}, {:10.3f}, " \
                "{:10.3f}".format(
        depth_metrics[0].mean(), depth_metrics[1].mean(), \
        depth_metrics[2].mean(), depth_metrics[3].mean(), \
        depth_metrics[5].mean(), depth_metrics[6].mean(), \
        depth_metrics[7].mean(), \
        aos_meter.avg, dim_meter.avg, cen_meter.avg)
    print(name_line)
    print(data_line)

    return depth_metrics[
               5].mean() + aos_meter.avg + dim_meter.avg + cen_meter.avg


def test_model(model, args):
    assert args.batch_size == 1
    # get data list for tracking
    tracking_list_seq = []
    tracking_list = []
    batch_time = tu.AverageMeter()
    data_time = tu.AverageMeter()

    # resume from a checkpoint
    nu.load_checkpoint(model, args.resume, is_test=True)

    cudnn.benchmark = True

    dataset = Dataset(args.json_path, 
                        'test',
                        args.data_split,
                        args.set == 'kitti',
                        args.percent,
                        args.is_tracking,
                        args.is_normalizing,
                        args.n_box_limit
                        )

    print("Number of image to test: {}".format(dataset.__len__()))

    # Data loading code
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model.eval()

    end = time.time()
    for i, (image, box_info) in enumerate(iter(test_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        with torch.no_grad():
            box_output, targets = model(image, box_info, args.device, 'test')

        batch_time.update(time.time() - end)

        rois_gt, \
        dim_gt_, \
        alpha_gt_, \
        dep_gt_, \
        cen_gt_, \
        loc_gt_, \
        ignore_, \
        tid_gt = targets

        cam_calib = box_info['cam_calib'].cpu().data.numpy().reshape(3, 4)
        cam_rot = box_info['cam_rot'].cpu().data.numpy()
        cam_loc = box_info['cam_loc'].cpu().data.numpy()
        box_gt = rois_gt.cpu().data.numpy()
        box_pd = box_output['rois'].cpu().data.numpy()
        dim_gt = dim_gt_.cpu().data.numpy()
        dim_pd = box_output['dim'].cpu().data.numpy()
        alpha_gt = alpha_gt_.cpu().data.numpy()
        alpha_pd = nu.get_alpha(box_output['rot'].cpu().data.numpy())
        depth_gt = dep_gt_.cpu().data.numpy()
        depth_pd = box_output['dep'].cpu().data.numpy()
        center_gt = cen_gt_.cpu().data.numpy()
        center_pd = box_output['cen'].cpu().data.numpy()
        loc_gt = loc_gt_.cpu().data.numpy()
        loc_pd = box_output['loc'].cpu().data.numpy()

        feature = F.normalize(
            F.avg_pool2d(box_output['feat'], (7, 7)).view(-1, 128))
        # feature = box_output['feat']
        feature_np = feature.cpu().data.numpy()

        tracking_list.append({
            'im_path': box_info['im_path'],
            'endvid': box_info['endvid'].cpu().data.numpy(),
            'rois_pd': box_pd,
            'rois_gt': box_gt,
            'feature': feature_np,
            'dim_pd': dim_pd,
            'alpha_pd': alpha_pd,
            'depth_pd': depth_pd,
            'center_pd': center_pd,
            'loc_pd': loc_pd,
            'dim_gt': dim_gt,
            'alpha_gt': alpha_gt,
            'depth_gt': depth_gt,
            'center_gt': center_gt,
            'loc_gt': loc_gt,
            'cam_calib': cam_calib,
            'cam_rot': cam_rot,
            'cam_loc': cam_loc,
            'ignore': ignore_.cpu().data.numpy(),
            'tid_gt': tid_gt.cpu().data.numpy(),
        })

        if box_info['endvid'].cpu().data.numpy().any() \
            or i == len(test_loader):
            tracking_list_seq.append(tracking_list)
            tracking_list = []

        if i % 100 == 0 and i != 0:
            print(i)
        end = time.time()


    if args.track_name is None:
        trk_name = os.path.join(cfg.OUTPUT_PATH,
                                '{}_{}_{}_bdd_roipool_output.pkl'.format(
                                    args.session,
                                    str(args.start_epoch).zfill(3),
                                    args.set))
    else:
        trk_name = os.path.join(cfg.OUTPUT_PATH, args.track_name)

    with open(trk_name, 'wb') as f:
        print("Saving {} with total {} sequences...".format(trk_name, len(
            tracking_list_seq)))
        pickle.dump(tracking_list_seq, f)


def main():
    torch.set_num_threads(multiprocessing.cpu_count())
    args = parse_args()
    if args.set == 'gta':
        from model.model import Model
    elif args.set == 'kitti':
        from model.model_cen import Model
    else:
        raise ValueError("Model not found")

    model = Model(args.arch,
                  args.roi_name,
                  args.down_ratio,
                  args.roi_kernel)
    model = nn.DataParallel(model)
    model = model.to(args.device)

    if args.phase == 'train':
        run_training(model, args)
    elif args.phase == 'test':
        test_model(model, args)


if __name__ == '__main__':
    main()
