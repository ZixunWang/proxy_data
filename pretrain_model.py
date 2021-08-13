import os
import sys
import argparse
import logging
import shutil

from easydict import EasyDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import utils
from model.resnet import resnet18, resnet34
from model.vgg import vgg11_bn, vgg19_bn
from nas_bench.utils import get_dataset


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')

name2model = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'vgg11_bn': vgg11_bn,
    'vgg19_bn': vgg19_bn,
}

CONFIG = {  # fixed
    'epoch': 200,
    'batch': 256,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'eta_min': 0.0,
}
CONFIG = EasyDict(CONFIG)


def train(net, train_loader, criterion, optimizer):
    net.train()
    loss_avg = utils.AverageMeter()
    for data, target in train_loader:
        n = data.size(0)
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.item(), n)
    return loss_avg.avg


def infer(net, test_loader, criterion):
    net.eval()
    loss_avg = utils.AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            n = data.size(0)
            data = data.cuda()
            target = target.cuda()
            logits = net(data)
            loss = criterion(logits, target)
            loss_avg.update(loss.item(), n)
    return loss_avg.avg


def main(model_name, dataset_name):
    logging.info(
        'Pretrain {} on {} ...'.format(model_name, dataset_name)
    )
    train_data, test_data, xshape, class_num = get_dataset(dataset_name)
    model = name2model[model_name](dataset_name, num_classes=class_num)
    model = nn.DataParallel(model.cuda())
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        CONFIG.lr,
        momentum=CONFIG.momentum,
        weight_decay=CONFIG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG.epoch,
        eta_min=CONFIG.eta_min,
    )
    trainloader = DataLoader(
        train_data,
        batch_size=CONFIG.batch,
        num_workers=2,
        pin_memory=True
    )
    testloader = DataLoader(
        test_data,
        batch_size=CONFIG.batch,
        num_workers=2,
        pin_memory=True,
    )
    min_loss = float('infinity')
    start_epoch = 1

    resume_ckpt = f'./checkpoints/pretrain_{model_name}_{dataset_name}.pth'
    if os.path.exists(resume_ckpt):
        checkpoint = torch.load(resume_ckpt)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        min_loss = checkpoint['min_loss']
        logging.info('Resume from {}: epoch={}'.format(resume_ckpt, start_epoch))
        
    for e in range(start_epoch, CONFIG.epoch+1):
        lr = scheduler.get_last_lr()[0]
        logging.info('Epoch {}, lr {}'.format(e, lr))
        train_loss = train(model, trainloader, criterion, optimizer)
        test_loss = infer(model, testloader, criterion)
        logging.info('train loss {}, test loss {}'.format(train_loss, test_loss))
        scheduler.step()

        ckpt = {
            'epoch': e,
            'state_dict': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_loss': min(min_loss, test_loss),
        }
        torch.save(ckpt, resume_ckpt)

        if test_loss < min_loss:
            min_loss = test_loss
            best_ckpt = './model/state_dicts/{}/{}.pt'.format(dataset_name, model_name)
            shutil.copyfile(resume_ckpt, best_ckpt)
            logging.info('save best model to {}'.format(best_ckpt))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretrain Model')
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    args = parser.parse_args()
    main(args.model, args.dataset)
