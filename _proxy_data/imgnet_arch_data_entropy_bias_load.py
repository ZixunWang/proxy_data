import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

from tqdm import tqdm

import time

results = torch.load('imgnet_arch_data_entropy_bias/entropy_of_models.pth')
print(results.keys()) #dict_keys(['resnet18', 'resnet50', 'vgg19'])

results_keys = list(results.keys())

results_list = []
for each_key in results.keys():
    results_list.append(results[each_key]['entropy'])

for i in range(len(results_list)):
    for j in range(i+1, len(results_list)):
        tmp_i = pd.Series(results_list[i])
        tmp_j = pd.Series(results_list[j])
        print(tmp_i.corr(tmp_j, method='spearman'), results_keys[i], results_keys[j])


# print(a, b, c) # 0.7877916754214503 0.8316686187516427 0.8133897021687078
