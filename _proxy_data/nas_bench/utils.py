import os
import sys
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset

from nats_bench import create
from .tools.XAutoDL.xautodl.models import get_cell_based_tiny_net
from .tools.XAutoDL.xautodl.config_utils.config_utils import load_config
from .tools.XAutoDL.xautodl.datasets.DownsampledImageNet import ImageNet16

NUM_BENCH_101 = None
NUM_BENCH_201 = 15625
Dataset2Class = {
    'cifar10': 10,
    'cifar100': 100,
    'ImageNet16-120': 120,
}


class SimpleDataset(Dataset):
    def __init__(self, data, transform=None):
        super(SimpleDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


class CUTOUT(object):
    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return "{name}(length={length})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_dataset(name, root=None, cutout=-1, eval_on_train=False):
    if root is None:
        root = (
            os.environ['DATASET']
            if 'DATASET' in os.environ
            else os.path.join(os.environ['HOME'], 'dataset')
        )
        root = os.path.join(root, 'cifar.python', name if name != 'ImageNet16-120' else 'ImageNet16')
    if name == "cifar10":
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == "cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith('ImageNet16'):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == "cifar10" or name == "cifar100":
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 32, 32)
    elif name.startswith('ImageNet16'):
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 16, 16)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == "cifar10":
        train_data = dset.CIFAR10(
            root, train=True, transform=train_transform if eval_on_train is False else test_transform, download=True
        )
        test_data = dset.CIFAR10(
            root, train=False, transform=test_transform, download=True
        )
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == "cifar100":
        train_data = dset.CIFAR100(
            root, train=True, transform=train_transform if eval_on_train is False else test_transform, download=True
        )
        test_data = dset.CIFAR100(
            root, train=False, transform=test_transform, download=True
        )
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'ImageNet16-120':
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num


def get_split(name):
    cfg_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs')
    if name == 'cifar10':
        split_info = load_config(os.path.join(cfg_root, 'cifar-split.txt'), None, None)
    else:
        raise NotImplementedError()
    return split_info


def get_score_201(dataset, search_space='tss', epoch=None, metric='test-accuracy', api=None):
    if api is None:
        api = create(None, search_space, fast_mode=True, verbose=False)
    score = []
    print(f'getting {metric} on {dataset} from nats bench with total epoch=200')
    for i in tqdm(range(NUM_BENCH_201)):
        score.append(api.get_more_info(i, dataset, iepoch=epoch, is_random=False, hp=200)[metric])
    return score


def get_rank_201(dataset, search_space='tss', epoch=None, metric='test-accuracy', store=None, api=None):
    score = get_score_201(dataset, search_space, epoch, metric, api)
    rank =  np.argsort(score)
    if store is not None:
        np.save(os.path.join(store, 'nas201_score.npy'), score)
        np.save(os.path.join(store, 'nas201_rank.npy'), rank)
    return rank


def get_net_201(dataset, search_space, index, api=None):
    if api is None:
        api = create(None, search_space, fast_mode=True, verbose=False)
    config = api.get_net_config(index, dataset)
    network = get_cell_based_tiny_net(config)
    return network


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bench', type=str, help='which bench to use')
    parser.add_argument('-d', '--dataset', type=str, help='which dataset to use')
    parser.add_argument('-s', '--search_space', type=str, help='search space of architecture')
    parser.add_argument('-e', '--epoch', type=int, help='')
    parser.add_argument('-m', '--metric', type=str, help='sort criteria')
    args = parser.parse_args()
    if args.bench == '201':
        get_rank_201(args.dataset, args.search_space, args.epoch, args.metric)
