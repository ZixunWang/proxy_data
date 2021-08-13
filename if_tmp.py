import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nas_bench.utils import get_dataset
import ptif
from model.resnet import resnet18

trainset, testset, _, _ = get_dataset('cifar10', eval_on_train=True)
test_loader = DataLoader(testset, shuffle=False)
train_loader = DataLoader(trainset, shuffle=True, batch_size=1)
model = resnet18(pretrained=True)

s_test, _ = ptif.calc_s_test(model, test_loader, train_loader)

print(len(s_test))
print(s_test[0].size())
