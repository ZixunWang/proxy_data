bench = '201'

dataset = 'ImageNet16-120'
root = None

sampler = 'dynamic tail entropy'
ratio = 0.5
net_name = None  # pretrained model to cal entropy

milestone_epoch = [10, 20] # at which epoch you want to change entropy loader
load_epoch = [50, 120, 190] # change to which entropy record

scores_total = f'./result/201_{dataset}_tss_test-accuracy_200epoch.npy'
model_num = 100

train = {
    'batch_size': 256,
    'average_times': 3,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'eta_min': 0.0,
    'epoch': 25,
}


