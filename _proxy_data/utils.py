import os
import sys
from importlib import import_module

from easydict import EasyDict
import pandas as pd
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable


class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  res = []
  for k in topk:
    correct_k = correct[:k].contiguous().view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def cal_corr(*scores):
    n = len(scores)
    assert n > 1, 'at least 2 variables to calc correlation'
    scores = [pd.Series(x) for x in scores]
    pearson_matrix = np.ones((n, n))
    spearman_matrix = np.ones((n,n))
    for i in range(n):
        for j in range(i+1, n):
            p = scores[i].corr(scores[j], method='pearson')
            s = scores[i].corr(scores[j], method='spearman')
            pearson_matrix[i, j] = pearson_matrix[j, i] = p
            spearman_matrix[i, j] = spearman_matrix[j, i] = s
    return pearson_matrix, spearman_matrix


def cfg_from_file(cfg_file):
    if cfg_file.endswith('.py'):
        module_name = os.path.basename(cfg_file)[:-3]
        if '.' in module_name:
            raise ValueError()
        config_dir = os.path.dirname(cfg_file)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
    else:
        raise NotImplementedError()
    return EasyDict(cfg_dict)
