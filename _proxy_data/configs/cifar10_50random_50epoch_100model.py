bench = '201'

dataset = 'cifar10'
root = None

sampler = 'random'
ratio = 0.5
net_name = None  # pretrained model to cal entropy

scores_total = f'./result/201_{dataset}_tss_test-accuracy_200epoch.npy'
model_num = 100

train = {
    'batch_size': 256,
    'average_times': 1,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'eta_min': 0.0,
    'epoch': 50,
}


