import os

import pandas as pd
import numpy as np
from nats_bench import create

from build_proxy_data import random_sampler
from nas_bench import utils
from nas_bench.utils import NUM_BENCH_201
from utils import AverageMeter


def cifar10_random_indices_prep():
    split_info = utils.get_split('cifar10')
    indices = random_sampler(split_info.train, 0.2, sampler=False)
    np.save('./result/cifar10_random_splited_indices.npy', indices)
    print(set(indices.tolist()) - set(split_info.train))
    print('done')


def scores_prep(dataset):
    scores = utils.get_score_201(dataset)
    np.save(f'./result/201_{dataset}_tss_test-accuracy_200epoch.npy', np.array(scores))
    print('done')


if __name__ == '__main__':
    eval_stability()
