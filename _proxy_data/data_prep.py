import os

# import fire
import pandas as pd
import numpy as np
from nats_bench import create

from build_proxy_data import random_sampler
from nas_bench import utils
from nas_bench.utils import NUM_BENCH_201
from utils import AverageMeter


def scores_prep(dataset):
    scores = utils.get_score_201(dataset)
    # print(scores)
    np.save(f'./result/201_{dataset}_tss_test-accuracy_200epoch.npy', np.array(scores))
    print('done')


if __name__ == '__main__':
    # fire.Fire()
    scores_prep("ImageNet16-120")
