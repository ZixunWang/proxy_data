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
from torch.utils.data.sampler import SubsetRandomSampler

import utils
from log_utils import Logger
from analysis import plot_corr, plot_1strank_base_acc
from build_proxy_data import *
from nats_bench import create
from nas_bench.utils import get_dataset, get_split, get_net_201, get_rank_201, NUM_BENCH_201

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def train(net, train_loader, criterion, optimizer):
    scaler = torch.cuda.amp.GradScaler()
    net.train()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    for data, target in train_loader:
        n = data.size(0)
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out, logits = net(data)
            loss = criterion(logits, target)
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        prec1, prec5 = utils.accuracy(logits, target, (1, 5))
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
    return top1.avg


def infer(net, test_loader):
    scaler = torch.cuda.amp.GradScaler()
    net.eval()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            n = data.size(0)
            data = data.cuda()
            target = target.cuda()

            with torch.cuda.amp.autocast():
                out, logits = net(data)

            prec1, prec5 = utils.accuracy(logits, target, (1, 5))
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
    return top1.avg

# def load_state_from_file_check_repeat(logger, indices):
#     state = logger.load_file('state')
#     scores = []
#     for i, cell in enumerate(indices):
#         if cell in state['scores']:
#             scores.append(state['scores'][cell])
#             continue


def main(cfg_file):
    cfg = utils.cfg_from_file(cfg_file)
    title = f'{cfg.dataset}-{cfg.ratio}{cfg.sampler}-{cfg.train["epoch"]}epoch'
    logger = Logger(cfg_file=cfg_file, log_dir=f'./log/{title}')
    logger.log(f'config: {cfg}')

    # train_data, test_data, xshape, class_num = get_dataset(cfg.dataset, cfg.root)
    if os.path.exists('/mnt/sda1/hehaowei/ImageNet16'):
        cfg.root = '/mnt/sda1/hehaowei/ImageNet16'
        train_data, test_data, xshape, class_num = get_dataset(cfg.dataset, cfg.root)
    elif os.path.exists('/mnt/lustre/hehaowei/ImageNet16'):
        cfg.root = '/mnt/lustre/hehaowei/ImageNet16'
        train_data, test_data, xshape, class_num = get_dataset(cfg.dataset, cfg.root)
    else:
        raise NotImplementedError

    train_indices = np.arange(len(train_data))
    if logger.data_sample_file.exists():
        indices = logger.load_file('data sample')
    else:
        if cfg.sampler == 'random':
            indices = random_sampler(train_indices, cfg.ratio, sampler=False)
            # if cfg.ratio == 0.125:
            #     indices = np.load('./result/ImageNet16-120_125random_sample.npy')
            # elif cfg.ratio == 0.25:  # fix
            #     indices = np.load('./result/ImageNet16-120_25random_sample.npy')
            # elif cfg.ratio == 0.5:
            #     indices = np.load('./result/ImageNet16-120_50random_sample.npy')
            # elif cfg.ratio == 1:
            #     indices = np.load('./result/ImageNet16-120_100random_sample.npy')
        elif cfg.sampler == 'low entropy':
            indices = low_entropy_sampler(cfg.dataset, cfg.net_name, train_indices, ratio=cfg.ratio, sampler=False)
        elif cfg.sampler == 'high entropy':
            indices = high_entropy_sampler(cfg.dataset, cfg.net_name, train_indices, ratio=cfg.ratio, sampler=False)
        elif cfg.sampler == 'tail entropy':
            indices = tail_entropy_sampler(cfg.dataset, cfg.net_name, train_indices, ratio=cfg.ratio, sampler=False)
        elif cfg.sampler == 'mid entropy':
            indices = mid_entropy_sampler(cfg.dataset, cfg.net_name, train_indices, ratio=cfg.ratio, sampler=False)
        elif cfg.sampler == 'influence':
            indices = influence_sampler(cfg.dataset, cfg.net_name, train_indices, ratio=cfg.ratio, sampler=False)
        elif cfg.sampler == 'pseudo influence':
            indices = pseudo_influence_sampler(cfg.dataset, cfg.net_name, train_indices, ratio=cfg.ratio, sampler=False)
        elif cfg.sampler == 'low L2':
            indices = low_L2_sampler(cfg.dataset, cfg.net_name, train_indices, ratio=cfg.ratio, sampler=False)
        elif cfg.sampler == 'high L2':
            indices = high_L2_sampler(cfg.dataset, cfg.net_name, train_indices, ratio=cfg.ratio, sampler=False)
        elif cfg.sampler == 'tail L2':
            indices = tail_L2_sampler(cfg.dataset, cfg.net_name, train_indices, ratio=cfg.ratio, sampler=False)
        elif cfg.sampler == 'dynamic random':
            indices = random_sampler(train_indices, cfg.ratio, sampler=False)
        elif cfg.sampler == 'dynamic low entropy':
            indices = dynamic_low_entropy_sampler(train_indices, cfg.ratio, sampler=False, load_epoch=10)
        elif cfg.sampler == 'dynamic tail entropy':
            indices = dynamic_tail_entropy_sampler(cfg.dataset, train_indices, cfg.ratio, sampler=False, load_epoch=10)
        else:
            raise NotImplementedError  # todo: other selected method
        # logger.save_file('data sample', indices)
    sampler = SubsetRandomSampler(indices)

    logger.log(f'gpu count: {torch.cuda.device_count()}')
    dataloader = DataLoader(
        train_data,
        sampler=sampler,
        batch_size=cfg.train['batch_size'],
        num_workers=2,
        pin_memory=True
    )
    testloader = DataLoader(
        test_data,
        batch_size=cfg.train['batch_size'],
        num_workers=2,
        pin_memory=True,
    )
    bench_api = create(None, 'tss', fast_mode=True, verbose=False)

    if logger.state_file.exists():
        state = logger.load_file('state')
        logger.log(f'resume state from {logger.state_file}')
    else:
        state = {'scores': {}}

    scores = []
    scores_total = np.load(cfg.scores_total)

    # if logger.model_sample_file.exists():
    #    indices = logger.load_file('model sample')
    # else:
    #    indices = np.random.choice(NUM_BENCH_201, cfg.model_num, replace=False)
    #    logger.save_file('model sample', indices)
    indices = np.load('./result/tss_100_sample_model.npy')
    logger.log('fix model indices')
    scores_base = scores_total[indices]



    epoch = cfg.train['epoch']
    for i, cell in enumerate(indices):
        state = logger.load_file('state') if logger.state_file.exists() else {'scores': {}}
        scores = state['scores']
        if (cell in scores) or i<args.start_model_index or i>=args.end_model_index:
            continue
        start = time.time()
        cell_score = []
        for t in range(cfg.train['average_times']):
            net = get_net_201(cfg.dataset, 'tss', cell, bench_api)
            net = net.cuda()
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = torch.optim.SGD(
                net.parameters(),
                cfg.train['learning_rate'],
                momentum=cfg.train['momentum'],
                weight_decay=cfg.train['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epoch,
                eta_min=cfg.train['eta_min']
            )



            logger.log(f'{i} iter, cell - {cell}, {t}th calc:')
            cur_score = []
            _i_old = -1
            for e in range(1, epoch + 1):
                lr = scheduler.get_last_lr()[0]
                logger.log(f'epoch {e}, learning rate {lr}')
                if cfg.sampler == 'dynamic random':
                    sampler = random_sampler(train_indices, cfg.ratio)
                    dataloader = DataLoader(
                        train_data,
                        sampler=sampler,
                        batch_size=cfg.train['batch_size'],
                        num_workers=2,
                        pin_memory=True
                    )
                if cfg.sampler in ['dynamic low entropy', 'dynamic tail entropy']:
                    _i = 0
                    for each in cfg.milestone_epoch:
                        if e >= each:
                            _i += 1
                    if _i_old != _i:
                        to_load_epoch = cfg.load_epoch[_i]
                        print('using dynamic low entropy sample at epoch %2d, loading from %3d' % (e, to_load_epoch))
                        if cfg.sampler=='dynamic low entropy': sampler = dynamic_low_entropy_sampler(train_indices, cfg.ratio, sampler=True, load_epoch=to_load_epoch)
                        elif cfg.sampler=='dynamic tail entropy': sampler = dynamic_tail_entropy_sampler(cfg.dataset, train_indices, cfg.ratio, sampler=False, load_epoch=to_load_epoch)
                        dataloader = DataLoader(
                            train_data,
                            sampler=sampler,
                            batch_size=cfg.train['batch_size'],
                            num_workers=2,
                            pin_memory=True
                        )
                        _i_old = _i
                train_acc = train(net, dataloader, criterion, optimizer)
                logger.log(f'top1 train acc: {train_acc}')
                test_acc = infer(net, testloader)
                cur_score.append(test_acc)
                logger.log(f'top1 test acc: {test_acc}')
                scheduler.step()
            cell_score.append(cur_score)

        logger.log(f'cell time consuming: {time.time() - start} s')
        state = logger.load_file('state') if logger.state_file.exists() else {'scores': {}}
        scores = state['scores']
        scores[cell] = cell_score
        state['scores'][cell] = cell_score
        logger.save_file('state', state)

    scores_base = pd.Series(scores_base)

    scores = [pd.Series([item[i][-1] for item in scores]) for i in range(cfg.train['average_times'])]
    if cfg.train['average_times'] != 1:
        plot_1strank_base_acc(*scores, title=title + 'self', dir_=logger.log_dir)

    scores.insert(0, scores_base)
    pearson, spearman = utils.cal_corr(*scores)
    logger.log(f'\npearson:\n{pearson}\nspearman:\n{spearman}')
    plot_1strank_base_acc(*scores, title=title + 'base', dir_=logger.log_dir)

    top50_indices = np.argsort(scores_base)[len(scores_base) // 2:]
    top50_scores = [score[top50_indices] for score in scores]
    pearson, spearman = utils.cal_corr(*top50_scores)
    logger.log(f'\ntop50 person:\n{pearson}\ntop50 spearman:\n{spearman}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file', type=str, help='proxy data constrution configs')
    parser.add_argument('--start_model_index', type=int, default=None, help='only train start_index to end_index models')
    parser.add_argument('--end_model_index', type=int, default=None, help='only train start_index to end_index models')
    args = parser.parse_args()
    main(args.cfg_file)

'''
CUDA_VISIBLE_DEVICES=0 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamiclowentropy_50epoch_100model.py --start_model_index 0 --end_model_index 25 &
CUDA_VISIBLE_DEVICES=1 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamiclowentropy_50epoch_100model.py --start_model_index 25 --end_model_index 50 &
CUDA_VISIBLE_DEVICES=2 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamiclowentropy_50epoch_100model.py --start_model_index 50 --end_model_index 75 &
CUDA_VISIBLE_DEVICES=3 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamiclowentropy_50epoch_100model.py --start_model_index 75 --end_model_index 100 &

CUDA_VISIBLE_DEVICES=0 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamicrandom_50epoch_100model.py --start_model_index 0 --end_model_index 25 &
CUDA_VISIBLE_DEVICES=1 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamicrandom_50epoch_100model.py --start_model_index 25 --end_model_index 50 &
CUDA_VISIBLE_DEVICES=2 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamicrandom_50epoch_100model.py --start_model_index 50 --end_model_index 75 &
CUDA_VISIBLE_DEVICES=3 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamicrandom_50epoch_100model.py --start_model_index 75 --end_model_index 100 &

CUDA_VISIBLE_DEVICES=0 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamictailentropy_50epoch_100model.py --start_model_index 0 --end_model_index 25 &
CUDA_VISIBLE_DEVICES=1 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamictailentropy_50epoch_100model.py --start_model_index 25 --end_model_index 50 &
CUDA_VISIBLE_DEVICES=2 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamictailentropy_50epoch_100model.py --start_model_index 50 --end_model_index 75 &
CUDA_VISIBLE_DEVICES=3 python from_cfg_test_partial_dynamic_entropy.py --cfg_file configs/ImageNet16-120_25dynamictailentropy_50epoch_100model.py --start_model_index 75 --end_model_index 100 &


'''