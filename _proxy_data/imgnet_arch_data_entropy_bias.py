import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

from tqdm import tqdm

import time

# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet_v2 = models.mobilenet_v2(pretrained=True)
# mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
# mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)

def _cal_entropy(dataset, net, batch=32):

    def _entropy(p):
        '''log2 scale entropy'''
        return np.log2(-np.sum(p * np.log2(p), axis=1))

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch, num_workers=4, prefetch_factor=2)
    if torch.cuda.is_available():
        device = 'cuda:0'
        net = nn.DataParallel(net.to(device))
    else:
        device = 'cpu'
    net.eval()
    ret = []
    with torch.no_grad():
        dataloader_bar = tqdm(dataloader)
        for data, _ in dataloader_bar:
            iter_start = time.time()
            data = data.to(device)
            iter_datatime = time.time()-iter_start
            pred = net(data)
            entropy = _entropy(F.softmax(pred, dim=1).detach().cpu().numpy())
            dataloader_bar.set_description("iter time %s %s" % (iter_datatime, time.time()-iter_start))
            ret.append(entropy)

    return np.hstack(ret)



# model_name = 'resnet18'
# model_name = 'resnet50'
# model_name = 'vgg19'
model_name = 'resnext50_32x4d'

data_path = '/mnt/sda1/hehaowei/ImageNet'
batch_size = 128

os.makedirs('imgnet_arch_data_entropy_bias', exist_ok=True)

model = getattr(models, model_name)(pretrained=True)

traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

if os.path.exists('imgnet_arch_data_entropy_bias/entropy_of_models.pth'):
    results = torch.load('imgnet_arch_data_entropy_bias/entropy_of_models.pth')
else:
    results = dict()

data_entropy = _cal_entropy(train_dataset, model, batch=batch_size)
try:
    results = torch.load('imgnet_arch_data_entropy_bias/entropy_of_models.pth')
except:
    pass
results[model_name] = dict()
results[model_name]['entropy'] = data_entropy
torch.save(results, 'imgnet_arch_data_entropy_bias/entropy_of_models.pth')

