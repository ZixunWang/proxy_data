import os
import sys
import time
from copy import deepcopy

import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import utils
from analysis import plot_corr, plot_1strank_base_acc
from build_proxy_data import *
from nats_bench import create
from nas_bench.utils import get_dataset, get_split, get_net_201, get_rank_201, NUM_BENCH_201


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')


def train(net, train_loader, criterion, optimizer):
    net.train()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    for data, target in train_loader:
        n = data.size(0)
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        out, logits = net(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, (1, 5))
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
    return top1.avg


def infer(net, test_loader):
    net.eval()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            n = data.size(0)
            data = data.cuda()
            target = target.cuda()
            
            out, logits = net(data)
            prec1, prec5 = utils.accuracy(logits, target, (1, 5))
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
    return top1.avg


def main(bench, dataset_name, sampler, net_name, ratio, average_times, root, batch_size, learning_rate, momentum, weight_decay, epoch, resume):
    logging.info(f'''
        dataset_name: {dataset_name},
        sampler: {sampler},
        net_name: {net_name},
        ratio: {ratio},
        average_times: {average_times},
        root: {root},
        batch_size: {batch_size}.
        learning_rate: {learning_rate}.
        momentum: {momentum},
        weight_decay: {weight_decay}'''
    )
    train_data, test_data, xshape, class_num = get_dataset(dataset_name, root)
    # split_info = get_split(dataset_name)
    # train_indices = split_info.train
    train_indices = np.arange(len(train_data))
    if sampler == 'random':
        # sampler = random_sampler(train_indices, ratio)
        from torch.utils.data.sampler import SubsetRandomSampler
        sampler = SubsetRandomSampler(np.load('./result/cifar10_random_indices.npy'))  # fixed
    elif sampler == 'low entropy':
        sampler = low_entropy_sampler(dataset_name, net_name, train_indices, ratio=ratio)
    elif sampler == 'high entropy':
        sampler = high_entropy_sampler(datset_name, net_name, train_indices, ratio=ratio)
    elif sampler == 'tail entropy':
        sampler = tail_entropy_sampler(dataset_name, net_name, triain_indices, ratio=ratio)
    else:
        pass # todo: other selected method

    logging.info(f'gpu count: {torch.cuda.device_count()}')
    dataloader = DataLoader(
        train_data,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
    testloader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
    bench_api = create(None, 'tss', fast_mode=True, verbose=False)
    if resume is not None and os.path.exists(resume):
        state = torch.load(resume)
        logging.info(f'resume from {resume}')
    else:
        state = {'scores': {}}
    
    scores = []
    scores_total = np.load(f'./result/201_{dataset_name}_tss_test-accuracy_200epoch.npy')
    indices = np.load('./result/tss_100_sample_model.npy')
    scores_base = scores_total[indices]

    for i, cell in enumerate(indices):
        if cell in state['scores']:
            scores.append(state['scores'][cell])
            continue
        start = time.time()
        cell_score = []
        for t in range(average_times):
            net = get_net_201(dataset_name, 'tss', cell, bench_api)
            net = net.cuda()
            net = nn.DataParallel(net)
            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = torch.optim.SGD(
                net.parameters(),
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epoch,
                eta_min=0
            )
            logging.info(f'{i} iter, cell - {cell}, {t}th calc:')
            cur_score = []
            for e in range(1, epoch+1):
                lr = scheduler.get_last_lr()[0]
                logging.info(f'epoch {e}, learning rate {lr}')
                train_acc = train(net, dataloader, criterion, optimizer)
                logging.info(f'top1 train acc: {train_acc}')
                test_acc = infer(net, testloader)
                cur_score.append(test_acc)
                logging.info(f'top1 test acc: {test_acc}')
                scheduler.step()
            cell_score.append(cur_score)

        logging.info(f'cell time consuming: {time.time()-start} s')
        scores.append(cell_score)
        if resume is not None:
            state['scores'][cell] = cell_score
            torch.save(state, resume)
            logging.info(f'state dict restores into {resume}')
    scores_base = pd.Series(scores_base)

    cal_epoch = [10, 50, 200]
    for e in cal_epoch:
        scores_1 = pd.Series([item[0][e] for item in scores])
        scores_2 = pd.Series([item[1][e] for item in scores])
        scores_3 = pd.Series([item[2][e] for item in scores])
        pearson, spearman = utils.cal_corr(scores_1, scores_2, scores_3, scores_base)
        logging.info(f'pearson:\n{pearson}\nspearman:\n{spearman}')
        plot_corr(scores_1, scores_2, title=f'{e} epoch: 1st-2nd rank correlation graph')
        plot_corr(scores_1, scores_3, title=f'{e} epoch: 1st-2nd rank correlation graph')
        plot_corr(scores_2, scores_3, title=f'{e} epoch: 2nd-3rd rank correlation graph')
        plot_corr(scores_base, scores_1, title=f'{e} epoch: base-1st rank correlation graph')
        plot_corr(scores_base, scores_2, title=f'{e} epoch: base-2nd rank correlation graph')
        plot_corr(scores_base, scores_3, title=f'{e} epoch: base-3rd rank correlation graph')
        plot_1strank_base_acc(scores_1, scores_2, scores_3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bench', type=str, help='which nas-bench to use')
    parser.add_argument('-d', '--dataset', type=str, help='which dataset to test')
    parser.add_argument('-s', '--sampler', type=str, help='proxy data selected method')
    parser.add_argument('-n', '--net', default='resnet18', type=str, help='pretrained model to cal entropy')
    parser.add_argument('-t', '--ratio', type=float, help='proxy_data_size / whole_data_size')
    parser.add_argument('-a', '--average_times', type=int, default=3, help='average times on proxy data')
    parser.add_argument('-r', '--root', type=str, help='path of dataset')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    main(args.bench, args.dataset, args.sampler, args.net, args.ratio, args.average_times, args.root, args.batch, args.learning_rate, args.momentum, args.weight_decay, args.epoch, args.resume)
