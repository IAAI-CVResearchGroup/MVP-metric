from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from torchreid import data_manager
from torchreid.dataset_loader import ImageDataset, VideoDataset
from torchreid import transforms as T
from torchreid import models
from torchreid.losses import CrossEntropyLabelSmooth, TripletLoss, KALoss, BA_TripletLoss, LiftedLoss
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.gpu_eval import gpu_evaluate
from torchreid.utils.torchtools import count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.samplers import RandomIdentitySampler
from torchreid.optimizers import init_optim


parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--seq-len', type=int, default=15,
                    help="number of images to sample in a tracklet")
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=500, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=10, type=int,
                    help="test batch size (number of tracklets)")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[300, 400], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
parser.add_argument('--lambda-xent', type=float, default=1,
                    help="weight to balance cross entropy loss")
parser.add_argument('--lambda-htri', type=float, default=1,
                    help="weight to balance hard triplet loss")
parser.add_argument('--label-smooth', action='store_true',
                    help="use label smoothing regularizer in cross entropy loss")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])
# Miscs
parser.add_argument('--print-freq', type=int, default=10,
                    help="print frequency")
parser.add_argument('--seed', type=int, default=1,
                    help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--vis-ranked-res', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")
parser.add_argument('--same-margin', type=float, default=-1,
                    help="margin for same class of our loss")
parser.add_argument('--use-auto-samemargin', action='store_true', help="use auto grad margin")



args = parser.parse_args(['--dataset', 'mars',
                          '--seq-len', '10', 
                          '--test-batch', '25', 
                          '--gpu-devices','1',
                          '--margin', '200.0',
                          '--same-margin', '200.0', 
                          '--max-epoch', '200',
                          '--use-auto-samemargin', 
                          '--eval-step', '2'
                         ])  


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_vidreid_dataset(root=args.root, name=args.dataset)

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    # decompose tracklets into images for image-based training
    new_train = []
    for img_paths, pid, camid in dataset.train:
        for img_path in img_paths:
            new_train.append((img_path, pid, camid))

    trainloader = DataLoader(
        ImageDataset(new_train, transform=transform_train),
        sampler=RandomIdentitySampler(new_train, args.train_batch, args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=args.seq_len, sample='evenly', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='evenly', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.3f} M".format(count_num_param(model)))

    criterion_xent = nn.CrossEntropyLoss()
    criterion_htri = TripletLoss(margin=args.margin)
    criterion_KA = KALoss(margin=args.margin, same_margin = args.same_margin, use_auto_samemargin = args.use_auto_samemargin)        
    cirterion_lifted = LiftedLoss(margin=args.margin)
    cirterion_batri = BA_TripletLoss(margin=args.margin)
    
    if args.use_auto_samemargin == True:
        G_params = [{'params': model.parameters(), 'lr': args.lr }, {'params': criterion_KA.auto_samemargin, 'lr': args.lr}]
    else :
        G_params = [para for _, para in model.named_parameters()]
    
    optimizer = init_optim(args.optim, G_params, args.lr, args.weight_decay)

    if args.load_weights:
        # load pretrained weights but ignore layers that don't match in size
        if check_isfile(args.load_weights):
            checkpoint = torch.load(args.load_weights)
            pretrain_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume:
        if check_isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            rank1 = checkpoint['rank1']
            print("Loaded checkpoint from '{}'".format(args.resume))
            print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, rank1))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        distmat = test(model, queryloader, galleryloader, args.pool, use_gpu, return_distmat=True)
        if args.vis_ranked_res:
            visualize_ranked_results(
                distmat, dataset,
                save_dir=osp.join(args.save_dir, 'ranked_results'),
                topk=20,
            )
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        adjust_learning_rate(optimizer, epoch)
        
        train(epoch, model, cirterion_batri, cirterion_lifted, criterion_xent, criterion_htri, criterion_KA, optimizer, trainloader, use_gpu)
        
        train_time += round(time.time() - start_train_time)
        
        scheduler.step()
        
        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, args.pool, use_gpu)
            is_best = rank1 > best_rank1
            
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, cirterion_batri, cirterion_lifted, criterion_xent, criterion_htri, criterion_KA, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_xent = AverageMeter()
    losses_kapos = AverageMeter()
    losses_kaneg = AverageMeter()
    losses_batri = AverageMeter()
    losses_lifted = AverageMeter()
    losses_htri = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        data_time.update(time.time() - end)
        
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        
        torch.cuda.empty_cache()
        outputs, features = model(imgs)

        xent_loss = criterion_xent(outputs, pids)
        losses_xent.update(xent_loss.item(), pids.size(0))
        
        
        htri_loss = criterion_htri(features, pids)
        losses_htri.update(htri_loss.item(), pids.size(0))
        
        
        lifted_loss = cirterion_lifted(features, pids)
        losses_lifted.update(lifted_loss.item(), pids.size(0))
        
        
        batri_loss = cirterion_batri(features, pids)
        losses_batri.update(batri_loss.item(), pids.size(0))
        
        
        loss_KA_pos = criterion_KA(features, features, pids, pids, mode = 'pos')
        losses_kapos.update(loss_KA_pos.item(), pids.size(0))
        
        loss_KA_neg = criterion_KA(features, features, pids, pids, mode = 'neg')
        losses_kaneg.update(loss_KA_neg.item(), pids.size(0))
        
       
        loss = xent_loss + (1.0 / 20) * batri_loss
        losses.update(loss.item(), pids.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

       

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'.format(
                   epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                   data_time=data_time),end='')
            print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss = losses), end='')
            print('xent_loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss = losses_xent), end='')
            print('htri_loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss = losses_htri), end='')
            print('lift_loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss = losses_lifted), end='')
            print('batri_loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss = losses_batri), end='')
            print('loss_KA_pos {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss = losses_kapos),end='')
            print('loss_KA_neg {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss = losses_kaneg),end='')
            print()
            sys.stdout.flush()
            
            
        
        end = time.time()


def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()
            b, s, c, h, w = imgs.size()
            imgs = imgs.view(b*s, c, h, w)
            
            end = time.time()
            outputs, features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.view(b, s, -1)
            if pool == 'avg':
                features = torch.mean(features, 1)
            else:
                features, _ = torch.max(features, 1)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()
            b, s, c, h, w = imgs.size()
            imgs = imgs.view(b*s, c, h, w)
            
            end = time.time()
            torch.cuda.empty_cache()
            outputs, features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.view(b, s, -1)
            if pool == 'avg':
                features = torch.mean(features, 1)
            else:
                features, _ = torch.max(features, 1)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch*args.seq_len))

    sys.stdout.flush()
    print("Computing CMC and mAP")
    cmc, mAP = gpu_evaluate(qf.cuda(), gf.cuda(), q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")
    sys.stdout.flush()

    if return_distmat:
        return distmat
    return cmc[0]

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 200 epochs"""
    lr = args.lr * pow(0.95, epoch)
    optimizer.param_groups[0]['lr'] = lr
    if args.use_auto_samemargin == True:
        optimizer.param_groups[1]['lr'] = lr * 100.0
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        print ('Learning Rate: {lr:.6f}'.format(lr=param_group['lr']))

if __name__ == '__main__':
    main()
