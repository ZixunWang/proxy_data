import os

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from build_proxy_data import load_prepared_result, random_sampler, low_entropy_sampler, high_entropy_sampler, tail_entropy_sampler
from nas_bench.utils import get_score_201, NUM_BENCH_201
from nats_bench import create


sns.set(palette='muted')


def plot_cifar10_entropy_hist_by_models():
    results = torch.load('./result/sample_results.pth')['cifar10']
    # === single ===
    for model in results:
        print(f'plotting histogram of {model} on cifar10')
        entropy = results[model]['entropy']
        plt.hist(entropy, bins=40, facecolor="red", edgecolor="black", alpha=0.7)
        plt.xlabel(f'entropy')
        plt.xlim((-5, 1))
        plt.ylim((0, 15000))
        plt.xticks(np.arange(-5, 1, 0.5))
        plt.yticks(np.arange(0, 15000, 1000))
        plt.title(f'cifar10_{model}_log-entropy_hist')
        plt.savefig(f'./result/plot/cifar10_{model}_log-entropy_hist.png')
        plt.close()
    
    # === merged ===
    print('plotting merged histograms to 3d space')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ylabels = list(results.keys())
    for i, model in enumerate(results):
        entropy = results[model]['entropy']
        hist, bins = np.histogram(entropy, bins=40)
        ax.bar(bins[:-1], hist, width=0.1, zs=i, zdir='y', alpha=0.7)
    ax.set_yticks([i for i in range(len(ylabel))])
    ax.set_yticklabels(ylabels)
    plt.savefig(f'./result/plot/cifar10_models_log-entropy_hist.png')
    plt.close()


def plot_cifar10_entropy_hist_by_samplers(model='resnet18', ratio=0.5):
    train_indices = np.arange(50000)  # cifar10 train
    random_indices = random_sampler(train_indices, ratio=ratio, sampler=False)
    entropy = load_prepared_result('cifar10', model, 'entropy')
    low_indices = low_entropy_sampler('cifar10', model, train_indices, ratio=ratio, sampler=False)
    high_indices = high_entropy_sampler('cifar10', model, train_indices, ratio=ratio, sampler=False)
    tail_indices = tail_entropy_sampler('cifar10', model, train_indices, ratio=ratio, sampler=False)
    indices_list = [random_indices, low_indices, high_indices, tail_indices]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ylabels = ['random', 'low', 'high', 'tail', 'total']
    for i, indices in enumerate(indices_list):
        hist, bins = np.histogram(entropy[indices], bins=20)
        ax.bar(bins[:-1], hist, width=0.1, zs=i, zdir='y', alpha=0.7)
    hist, bins = np.histogram(entropy, bins=40)
    ax.bar(bins[:-1], hist, width=0.1, zs=len(indices_list), zdir='y', alpha=0.7)

    ax.set_yticks([i for i in range(len(ylabels))])
    ax.set_yticklabels(ylabels)
    plt.savefig(f'./result/plot/cifar10_{model}_different_entropy_samplers_hist.png')
    plt.close()


def plot_ImageNet_skeleton(model='resnet18'):
    skeleton = load_prepared_result('ImageNet16-120', model, 'skeleton')
    cmap = plt.get_cmap('Paired')


def plot_corr(A, B, title='correlation graph'):
    assert len(A) == len(B)
    n = len(A)
    indices_A = np.argsort(A)
    indices_B = np.argsort(B)
    rank_B = [0] * n
    for i in range(n):
        rank_B[indices_B[i]] = i
    plt.plot(np.arange(n), np.arange(n))
    plt.scatter(np.arange(n), [rank_B[x] for x in indices_A], s=1, c='red', alpha=0.7)
    plt.title(title)
    plt.savefig(f'./result/plot/{title}.png')
    plt.close()


def eval_bench_12_200():
    api = create(None, 'tss', fast_mode=True, verbose=False)
    base12 = [api.get_more_info(i, 'cifar10', is_random=False, hp=12)['test-accuracy'] for i in range(NUM_BENCH_201)]
    base200 = [api.get_more_info(i, 'cifar10', is_random=False, hp=200)['test-accuracy'] for i in range(NUM_BENCH_201)]
    spearman = pd.Series(base200).corr(pd.Series(base12), method='spearman')
    pearson = pd.Series(base200).corr(pd.Series(base12), method='pearson')
    print(f'spearman: {spearman}, pearson: {pearson}')
    plot_corr(base200, base12, 'base200-base12')


def eval_bench_self(hp):
    api = create(None, 'tss', fast_mode=True, verbose=False)
    a = [api.get_more_info(i, 'cifar10', is_random=True, hp=hp)['test-accuracy'] for i in range(NUM_BENCH_201)]
    b = [api.get_more_info(i, 'cifar10', is_random=True, hp=hp)['test-accuracy'] for i in range(NUM_BENCH_201)]
    spearman = pd.Series(a).corr(pd.Series(b), method='spearman')
    pearson = pd.Series(a).corr(pd.Series(b), method='pearson')
    print(f'spearman: {spearman}, pearson: {pearson}')
    plot_corr(a, b, f'{hp}-self')


def plot_1strank_base_acc(*scores, title, dir_):
    scores_base = scores[0]
    indices_base = np.argsort(scores_base)
    n = len(indices_base)
    for i in range(1, len(scores)):
        plt.plot(np.arange(n), scores[i][indices_base])
    plt.title('title')
    plt.plot(np.arange(n), scores_base[indices_base])
    plt.savefig(os.path.join(str(dir_), f'{title}.png'))
    plt.close()


def main():
    state = torch.load('./result/cifar10_random_avg_val_dict.pth')
    scores_total = np.load('./result/201_cifar10_test-accuracy_11epoch.npy')
    model = np.load('./result/last_1000_model.npy')
    scores_base = scores_total[model]
    scores_1 = [state['scores'][x][0] for x in model]
    scores_2 = [state['scores'][x][1] for x in model]
    scores_3 = [state['scores'][x][2] for x in model]
    plot_corr(scores_base, scores_1)


if __name__ == '__main__':
    eval_bench_12_200()
