import os
import sys
from tqdm import tqdm

import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

from nas_bench.utils import get_dataset
from model.resnet import resnet18, resnet34
from model.vgg import vgg11_bn, vgg19_bn

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')

config = {
    #'cifar10': {
    #    'resnet18': resnet18,
    #    'resnet34': resnet34,
    #    'vgg11': vgg11_bn,
    #    'vgg19': vgg19_bn,
    #},
    'ImageNet16-120': {
        'resnet18': resnet18,
    }
}


def random_sampler(indices, ratio=0.2, sampler=True):
    sample = np.random.choice(indices, int(len(indices)*ratio), replace=False)
    if not sampler:
        return sample
    sampler = SubsetRandomSampler(sample)
    return sampler


def load_prepared_result(dataset, net, result):
    # actually is log10-scale-entropy
    record_file = './result/sample_results.pth'
    results = torch.load(record_file)
    return results[dataset][net][result]


def low_entropy_sampler(dataset, net, indices, ratio=0.2, sampler=True):
    entropy = load_prepared_result(dataset, net, 'entropy')[indices]
    num_sample = int(len(indices) * ratio)
    sample = np.argsort(entropy)[:num_sample]
    if not sampler:
        return sample
    sampler = SubsetRandomSampler(sample)
    return sampler


def mid_entropy_sampler(dataset, net, indices, ratio=0.2, sampler=True):
    entropy = load_prepared_result(dataset, net, 'entropy')[indices]
    num_sample = int(len(indices) * ratio)
    sample = np.argsort(entropy)[len(entropy)//2 - num_sample//2:len(entropy)//2 + num_sample//2]
    if not sampler:
        return sample
    sampler = SubsetRandomSampler(sample)
    return sampler


def high_entropy_sampler(dataset, net, indices, ratio=0.2, sampler=True):
    entropy = load_prepared_result(dataset, net, 'entropy')[indices]
    num_sample = int(len(indices) * ratio)
    sample = np.argsort(entropy)[::-1][:num_sample]
    if not sampler:
        return sample
    sampler = SubsetRandomSampler(sample)
    return sampler


def tail_entropy_sampler(dataset, net, indices, ratio=0.2, sampling_type=1, sampler=True):
    '''refer: https://github.com/nabk89/NAS-with-Proxy-data/blob/main/sampler.py'''
    entropy = load_prepared_result(dataset, net, 'entropy')[indices]
    min_entropy, max_entropy = np.min(entropy), np.max(entropy)
    if dataset == 'cifar10': bin_width = 0.5
    elif dataset == 'cifar100': bin_width = 0.25
    elif dataset.startswith('ImageNet'): bin_width = 0.25

    low_bin = np.round(min_entropy)
    while min_entropy < low_bin:
        low_bin -= bin_width
    high_bin = np.round(max_entropy)
    while max_entropy > high_bin:
        high_bin += bin_width
    bins = np.arange(low_bin, high_bin+bin_width, bin_width)

    def get_bin_idx(ent):
        for i in range(len(bins)-1):
            if (bins[i] <= ent) and (ent <= bins[i+1]):
                return i
        return None

    index_histogram = []
    for i in range(len(bins)-1):
        index_histogram.append([])

    for index, e in enumerate(entropy):
        bin_idx = get_bin_idx(e)
        if bin_idx is None:
            raise ValueError("[Error] histogram bin settings is wrong ... histogram bins: [%f ~ %f], current: %f"%(low_bin, high_bin, e))
        index_histogram[bin_idx].append(index)

    histo = np.array([len(h) for h in index_histogram])
    if sampling_type == 1:
        inv_histo = (max(histo) - histo + 1) * (histo != 0)
        inv_histo_prob = inv_histo / np.sum(inv_histo)
    elif sampling_type == 2:
        inv_histo_prob = np.array([1/(len(bins)-1) for _ in index_histogram])
    elif sampling_type == 3:
        inv_histo_prob = np.array([(1/len(l) if len(l) != 0 else 0) for l in index_histogram])
    else:
        raise ValueError("Error in sampling type for histogram-based sampling")

    if dataset == 'imagenet':
        num_proxy_data = int(int(np.floor(ratio * len(entropy))) / 1000) * 1000
    else:
        num_proxy_data = int(np.floor(ratio * len(entropy)))

    indices = []
    total_indices = []
    total_prob = []
    for index_bin, prob in zip(index_histogram, inv_histo_prob):
        if len(index_histogram) == 0:
            continue
        total_indices += index_bin
        temp = np.array([prob for _ in range(len(index_bin))])
        temp = temp/len(index_bin)
        total_prob += temp.tolist()
    total_prob = total_prob / np.sum(total_prob)
    indices = np.random.choice(total_indices, size=num_proxy_data, replace=False, p=total_prob)
    
    if not sampler:
        return indices
    sampler = SubsetRandomSampler(indices)
    return sampler


def influence_sampler(dataset, net, indices, ratio=0.2, sampler=True):
    influence = load_prepared_result(dataset, net, 'influence')[indices]
    num_sample = int(len(indices) * ratio)
    sample = np.argsort(influence)[::-1][:num_sample]
    if not sampler:
        return sample
    sampler = SubsetRandomSampler(sample)
    return sampler


def pseudo_influence_sampler(dataset, net, indices, ratio=0.2, sampler=True):
    pinfluence = load_prepared_result(dataset, net, 'pseudo influence')[indices]
    num_sample = int(len(indices) * ratio)
    sample = np.argsort(pinfluence)[::-1][:num_sample]
    if not sampler:
        return sample
    sampler = SubsetRandomSampler(sample)
    return sampler


def skeleton_sampler(dataset, net, indices, ratio=0.2, sampler=True):
    skeleton = load_prepared_result(dataset, net, 'skeleton')
    num_sample = int(len(indices) * ratio)
    sample = []
    num_class = len(skeleton.keys())

    center = {}
    for key in skeleton.keys():
        center[key] = np.mean(np.vstack([x[0] for x in skeleton[key]]))
    
    offset = 100
    for key in skeleton.keys():
        dis = [(x[0] - center[key], x[1]) for x in skeleton[key]]
        dis.sort(reverse=True)
        sample.extend([x[1] for x in dis[offset: num_sample//num_class+offset]])
    if not sampler:
        return sample
    sampler = SubsetRandomSampler(sample)
    return sampler


def forgetting_sampler(dataset, net, indices, ratio=0.2, sampler=True):
    forgetting = load_prepared_result(dataset, net, 'forgetting')
    num_sample = int(len(indices) * ratio)
    sample = np.argsort(forgetting)[::-1][:num_sample]
    if not sampler:
        return sample
    sampler = SubsetRandomSampler(sample)
    return sampler


def _cal_entropy(dataset, net, batch=32):

    def _entropy(p):
        '''log2 scale entropy'''
        return np.log2(-np.sum(p * np.log2(p), axis=1))

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch, num_workers=2)
    if torch.cuda.is_available():
        device = 'cuda:0'
        net = nn.DataParallel(net.to(device))
    else:
        device = 'cpu'
    net.eval()
    ret = []
    with torch.no_grad():
        for data, _ in dataloader:
            pred = net(data.to(device)).detach().cpu()
            entropy = _entropy(F.softmax(pred, dim=1).numpy())
            ret.append(entropy)

    return np.hstack(ret)


def _cal_influence(trainset, testset, net, dataset):
    # assert torch.cuda.is_available(), 'need cuda to compute influence function, while not'
    from ptif.calc_influence_function import calc_s_test
    # import pytorch_influence_functions as ptif
    # net = net.cuda()
    trainloader = DataLoader(trainset, shuffle=True, batch_size=16, num_workers=2)
    testloader = DataLoader(testset, shuffle=False, batch_size=1)
    save_path = f'./result/s_test/{net}_{dataset}_stest.npy'
    if not os.path.exists(save_path):
        s_test, _ = calc_s_test(net, trainloader, testloader)
        np.save(save_path, s_test)


def _cal_pseudo_influence(trainset, testset, net, dataset):
    def _calc_grad(data, target, net, criterion):
        net.zero_grad()
        pred = net(data)
        loss = criterion(pred, target)
        params = [p for p in net.parameters() if p.requires_grad]
        grad = torch.autograd.grad(loss, params, create_graph=True)
        grad = torch.cat([torch.flatten(x) for x in grad], 0).detach().cpu()
        return grad

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        device = 'cuda:0'
        net = nn.DataParallel(net.to(device))
    else:
        device = 'cpu'
    net.eval()

    test_grad = 0
    logging.info('calculating test grad')
    for data, target in tqdm(testset):
        data = torch.Tensor(data).unsqueeze(0).to(device)
        target = torch.Tensor([target]).long().to(device)
        grad = _calc_grad(data, target, net, criterion)
        test_grad += grad.numpy()
    test_grad /= len(testset)

    logging.info('calclating train grad')
    pseudo_influence = []
    for data, target in tqdm(trainset):
        data = torch.Tensor(data).unsqueeze(0).to(device)
        target = torch.Tensor([target]).long().to(device)
        grad = _calc_grad(data, target, net, criterion).numpy()
        pseudo_influence.append(test_grad.dot(grad) / len(grad))
    return np.array(pseudo_influence, dtype=np.float64)


def _cal_skeleton(trainset, net):
    from sklearn.manifold import TSNE
    feas = []

    def _hook(module, fea_in, fea_out):
        feas.append(fea_out)
        return fea_out
    
    skeleton = {}
    net.avgpool.register_forward_hook(hook=_hook)
    if torch.cuda.is_available():
        device = 'cuda:0'
        net = nn.DataParallel(net.to(device))
    else:
        device = 'cpu'
    net.eval()

    res = []
    for idx, (data, target) in tqdm(enumerate(trainset)):
        data = torch.Tensor(data).unsqueeze(0).to(device)
        _ = net(data)
        if target not in skeleton:
            skeleton[target] = []
        code = feas[-1].detach().cpu().view(1, -1).squeeze().numpy()
        res.append((code, idx, target))
        feas.pop()

    embedded = TSNE(n_components=2).fit_transform(np.vstack([x[0] for x in res]))

    for _, idx, target in res:
        skeleton[target].append((embedded[idx], idx))

    return skeleton


def _cal_forgetting(trainset, net):
    class DatasetWarp(Dataset):
        def __init__(self, data):
            super(DatasetWarp, self).__init__()
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return idx, self.data[idx]
    
    if torch.cuda.is_available():
        device = 'cuda:0'
        net = nn.DataParallel(net.to(device))
    else:
        device = 'cpu'
    forgetting = [0] * len(trainset)
    prev_acc = [0] * len(trainset)
    trainset = DatasetWarp(trainset)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=256)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), 0.1, 0.9, 5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for e in range(200):
        for idx, data_pack in trainloader:
            data, target = data_pack
            data = data.to(device)
            target = target.to(device)

            net.eval()
            pred = net(data)
            pred = pred.argmax(dim=1)
            for i in range(len(idx)):
                acc = int(pred[i] == target[i])
                if prev_acc[idx[i]] > acc:
                    forgetting[idx[i]] += 1
                prev_acc[idx[i]] = acc

            net.train()
            optimizer.zero_grad()
            logits = net(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return forgetting

def get_sample_results():
    logging.info('start to get sample results ...')
    record_file = './result/sample_results.pth'
    if os.path.exists(record_file):
        results = torch.load(record_file)
    else:
        results = dict()
    for dataset in config:
        if dataset not in results:
            results[dataset] = dict()
        nets = config[dataset]
        trainset, testset, _, class_num = get_dataset(dataset, eval_on_train=True)  # cancel aug on train
        for net_name in nets:
            logging.info(f'==[dataset: {dataset}, net: {net_name}]==')
            if net_name not in results[dataset]:
                results[dataset][net_name] = dict()
            net = nets[net_name](dataset, pretrained=True, num_classes=class_num)

            if 'entropy' not in results[dataset][net_name]:
                logging.info(f'[dataset: {dataset}, net: {net_name}]: getting entropy...')
                entropy = _cal_entropy(trainset, net)
                results[dataset][net_name]['entropy'] = entropy
                logging.info(f'[dataset: {dataset}, net: {net_name}]: done')
            else:
                logging.info(f'[dataset: {dataset}, net: {net_name}]: entropy has already calculated')

            '''if 'influence' not in results[dataset][net_name] and dataset[0]=='I':
                logging.info(f'[dataset: {dataset}, net: {net_name}]: getting influence...')
                influence = _cal_influence(trainset, testset, net, dataset)
                results[dataset][net_name]['influence'] = influence
                logging.info(f'[dataset: {dataset}, net: {net_name}]: done')
            else:
                logging.info(f'[dataset: {dataset}, net: {net_name}]: influence has already calculated')
            '''
            if 'pseudo influence' not in results[dataset][net_name]:
                logging.info(f'[dataset: {dataset}, net: {net_name}]: getting pseudo influence...')
                pseudo_influence = _cal_pseudo_influence(trainset, testset, net, dataset)
                results[dataset][net_name]['pseudo influence'] = pseudo_influence
                logging.info(f'[dataset: {dataset}, net: {net_name}]: done')
            else:
                logging.info(f'[dataset: {dataset}, net: {net_name}]: pseudo influence has already calculated')

            '''if dataset[0]=='I':
                logging.info(f'[dataset: {dataset}, net: {net_name}]: getting skeleton...')
                skeleton = _cal_skeleton(trainset, net)
                results[dataset][net_name]['skeleton'] = skeleton
                logging.info(f'[dataset: {dataset}, net: {net_name}]: done')
            else:
                logging.info(f'[dataset: {dataset}, net: {net_name}]: skeleton has already calculated')
            '''

            if 'forgetting' not in results[dataset][net_name]:
                logging.info(f'[dataset: {dataset}, net: {net_name}]: getting forgetting events...')
                forgetting = _cal_forgetting(trainset, net)
                results[dataset][net_name]['forgetting'] = forgetting
                logging.info(f'[dataset: {dataset}, net: {net_name}: done')
            else:
                logging.info(f'[dataset: {dataset}, net: {net_name}]: forgetting events has already calculated')

    torch.save(results, record_file)
    logging.info(f'sample results have been stored to {os.path.abspath(record_file)} !')


if __name__ == '__main__':
    get_sample_results()
