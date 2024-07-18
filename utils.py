# This code is based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import time
from networks import ConvNet, CNN_CIFAR10, CNN_FEMNIST, CharLSTM, CNN_CelebA
import random
from torch.utils.data import DataLoader, random_split
import torchvision
import json
from PIL import Image
from torchvision import transforms
import os

def lda_partition(trainset, n_clients, alpha, dataset_dir):
    if dataset_dir:
        cache_path = os.path.join(dataset_dir, f"split_dir_{alpha}.pt")
        if os.path.exists(cache_path):
            net_dataidx_map = torch.load(cache_path)
            return net_dataidx_map

    # using dirichlet distribution to sampling niid data for each clients
    labels = []
    if isinstance(trainset, list):
        labels = trainset
    else:
        for _, label in trainset:
            labels.append(int(label))
    SRC_CLASSES = np.unique(labels)
    min_sample = len(SRC_CLASSES) * 1 # min. 2 samples per class
    min_size = 0 # track minimal samples per user
    max_samples_per_user = len(labels) / n_clients

    data_by_label = []
    for i in SRC_CLASSES:
        data_by_label.append(np.where(labels == i)[0])

    ###### Determine Sampling #######
    while min_size < min_sample:
        print("Try to find valid data separation")
        idx_batch = [{} for _ in range(n_clients)]
        samples_per_user = [0 for _ in range(n_clients)]

        for l in SRC_CLASSES:
            # get indices for all that label
            idx_l = data_by_label[l]
            np.random.shuffle(idx_l)

            proportions=np.random.dirichlet(np.repeat(alpha, n_clients))
            # re-balance proportions
            proportions=np.array([p * (n_per_user < max_samples_per_user) for p, n_per_user in zip(proportions, samples_per_user)])
            proportions=proportions / proportions.sum()
            proportions=(np.cumsum(proportions) * len(idx_l)).astype(int)[:-1]
            # participate data of that label
            for u, new_idx in enumerate(np.split(idx_l, proportions)):
                # add new idex to the user
                idx_batch[u][l] = new_idx.tolist()
                samples_per_user[u] += len(idx_batch[u][l])
        min_size=min(samples_per_user)

    net_dataidx_map = [[] for _ in range(n_clients)]
    for i in range(n_clients):
        for l in SRC_CLASSES:
            net_dataidx_map[i] += idx_batch[i][l]
        # ct = [0 for _ in range(200)]
        # for k in net_dataidx_map[i]:
        #     ct[labels[k]] += 1
        # print(ct)

    # torch.save(net_dataidx_map, cache_path)
    return net_dataidx_map

def get_lengths(X):

    lengths = X.shape[1] - (X == 0).sum(1)
    return lengths.sort(0, descending=True)

def annealing(base, r, max_r, method):
    if method=='None' or method is None:
        return base
    elif 'step' in method:
        step = int(method.split('_')[1])
        return base * (0.9 ** (r // step))
    elif 'cos' in method:
        return base * 0.5 * (1.0 + np.cos(np.pi * r / max_r))
    elif 'power' in method:
        return base * (1.0 - (r / max_r)**5)
    assert 2<1


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def discounted_mean(trace, factor=0.9):
    weight = factor ** np.flip(np.arange(len(trace)), axis=0)
    return np.inner(trace, weight) / weight.sum()

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def select_user_idxs(num_users, num_selected_users):
    '''selects num_clients clients weighted by number of samples from possible_clients
    Args:
        num_clients: number of clients to select; default 20
            note that within function, num_clients is set to
            min(num_clients, len(possible_clients))
    Return:
        list of selected clients objects
    '''
    if (num_users <= num_selected_users or num_selected_users==-1):
        return list(range(num_users))

    return np.random.choice(list(range(num_users)), num_selected_users, replace=False)

def select_users(users, num_selected_users):
    '''selects num_clients clients weighted by number of samples from possible_clients
    Args:
        num_clients: number of clients to select; default 20
            note that within function, num_clients is set to
            min(num_clients, len(possible_clients))
    Return:
        list of selected clients objects
    '''
    if (len(users) <= num_selected_users or num_selected_users==-1):
        return users

    return np.random.choice(users, num_selected_users, replace=False)

def setup_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_loader_celeba(task, task_test):
    image_root = "/home/cc/leaf/data/celeba/data/raw/img_align_celeba/"
    ToTensor = transforms.ToTensor()
    X, Y = task['X'], task['Y']
    len_test = len(task_test['Y'])//2
    X_val, Y_val = task_test['X'][:len_test], task_test['Y'][:len_test]
    X_test, Y_test = task_test['X'][len_test:], task_test['Y'][len_test:]
    data = {
        'train': (X, Y),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test),
    }
    def loader(*args):
        output = []
        for arg in args:
            Xarg, Yarg = data[arg]
            X = []
            for p in Xarg:
                img = Image.open(image_root + p)
                X.append(ToTensor(img))
            X = torch.stack(X, 0)
            output.append(X.cuda(non_blocking=True))
            output.append(Yarg.cuda(non_blocking=True))
        return output
    return loader

def get_loader_femnist(task, task_test):
    X, Y = task['X'], task['Y']
    len_test = len(task_test['Y'])//2
    X_val, Y_val = task_test['X'][:len_test], task_test['Y'][:len_test]
    X_test, Y_test = task_test['X'][len_test:], task_test['Y'][len_test:]
    data = {
        'train': (X, Y),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test),
    }
    def loader(*args):
        output = []
        for arg in args:
            Xarg, Yarg = data[arg]
            output.append(Xarg.cuda(non_blocking=True))
            output.append(Yarg.cuda(non_blocking=True))
        return output
    return loader

def get_loader_cifar10(inds, trainset=None, testset=None, use_mp = True):
    data = {}
    trainset_ind, valset_ind, testset_ind = inds
    if trainset:
        data['train'] = torch.utils.data.DataLoader(trainset,
                                                sampler=torch.utils.data.SubsetRandomSampler(trainset_ind), #without replacement
                                                batch_size=len(trainset_ind),)
        data['val'] = torch.utils.data.DataLoader(trainset,
                                                sampler=torch.utils.data.SubsetRandomSampler(valset_ind), #without replacement
                                                batch_size=len(valset_ind),)
    if testset:
        data['test'] = torch.utils.data.DataLoader(testset,
                                                sampler=torch.utils.data.SubsetRandomSampler(testset_ind), #without replacement, meaning no duplication
                                                batch_size=len(testset_ind),)

    def loader(*args):
        output = []
        for arg in args:
            Xarg, Yarg = next(iter(data[arg]))
            output.append(Xarg.cuda(non_blocking=True))
            output.append(Yarg.cuda(non_blocking=True))
        return output
    if use_mp:
        return data
    return loader

def train_multi_epoch(epochs, loader, net, criterion, device, bs=-1, optimizer=None):
    net.train()
    L_tot = 0.
    if callable(loader) and isinstance(loader('train')[0], list):
        X, Y = loader('train')
        X = X.to(device)
        Y = Y.to(device)
        N = len(Y)
        if bs==-1: bs=N
        for ep in range(epochs):
            randperm = torch.randperm(N)
            X, Y = X[randperm], Y[randperm]
            for i in range(0, N, bs):
                Xbatch, Ybatch =X[i:i+bs], Y[i:i+bs]
                if isinstance(net, CharLSTM):
                    lengths, sortperm = get_lengths(Xbatch)
                    o = net(Xbatch[sortperm], lengths.cpu())
                    Ybatch = Ybatch[sortperm]
                else:
                    o = net(Xbatch)
                L = criterion(o, Ybatch)
                L_tot += float(L) * len(Ybatch)
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
    else:
        trainset = loader
        N = len(trainset)
        if bs==-1: bs=N
        loader = torch.utils.data.DataLoader(trainset, batch_size=int(bs), shuffle=True, num_workers=4)
        for ep in range(epochs):
            for batch in loader:
                if 'pixel_values' in batch.keys():
                    Xbatch = batch['pixel_values'].to(device)
                    Ybatch = batch['label'].to(device)
                else:
                    Xbatch, Ybatch = batch
                    N += len(Ybatch)
                o = net(Xbatch.to(device))
                L = criterion(o, Ybatch.to(device))
                L_tot += float(L) * len(Ybatch)
                optimizer.zero_grad()
                L.backward()
                optimizer.step()

    return L_tot/(N*epochs), N

def conduct_one_epoch(mode, loader, net, criterion, device, bs=128, optimizer=None):
    if mode=='train':
        net.train()
    else:
        net.eval()

    if (callable(loader) and isinstance(loader(mode)[0], list)) or isinstance(loader, list):
        if isinstance(loader, list):
            X, Y = loader[0], loader[1]
        else:
            X, Y = loader(mode)
        L_tot = 0.
        acc = 0.
        N = len(Y)
        randperm = torch.randperm(N)
        X, Y = X[randperm].to(device), Y[randperm].to(device)
        if bs==-1: bs=N
        for i in range(0, N, bs):
            Xbatch, Ybatch =X[i:i+bs], Y[i:i+bs]
            if isinstance(net, CharLSTM):
                lengths, sortperm = get_lengths(Xbatch)
                o = net(Xbatch[sortperm], lengths.cpu())
                Ybatch = Ybatch[sortperm]
            else:
                o = net(Xbatch)
            L = criterion(o, Ybatch)
            if mode!='train':
                acc += (Ybatch == o.argmax(1)).sum().float()
            L_tot += float(L) * len(Ybatch)
            if mode=='train':
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
    else:
        dataset = loader
        N = len(dataset)
        if bs==-1: bs=N
        loader = torch.utils.data.DataLoader(dataset, batch_size=int(bs), shuffle=True, num_workers=4)
        L_tot = 0.
        acc = 0.
        for batch in loader:
            if 'pixel_values' in batch.keys():
                Xbatch = batch['pixel_values'].to(device)
                Ybatch = batch['label'].to(device)
            else:
                Xbatch, Ybatch = batch
                N += len(Ybatch)
            o = net(Xbatch.to(device))
            L = criterion(o, Ybatch.to(device))
            L_tot += float(L) * len(Ybatch)
            if mode!='train':
                acc += (Ybatch.to(device) == o.argmax(1)).sum().float()
            if mode=='train':
                optimizer.zero_grad()
                L.backward()
                optimizer.step()

    return L_tot/N, acc/N, N

def create_network(config, pretrained=False):
    if config['backbone']=='resnet18':
        T = torchvision.models.resnet18(pretrained=pretrained)
        T.fc = nn.Linear(512, config['num_classes'])
        T.feature_dim = 512

    elif config['backbone']=='resnet34':
        T = torchvision.models.resnet34(pretrained=pretrained)
        T.fc = nn.Linear(512, config['num_classes'])
        T.feature_dim = 512

    elif config['backbone']=='resnet50':
        T = torchvision.models.resnet50(pretrained=pretrained)
        T.fc = nn.Linear(2048, config['num_classes'])
        T.feature_dim = 2048

    elif config['backbone']=='alexnet':
        T = AlexNet_BN(num_classes=config['num_classes'])

    elif config['backbone']=='efficientnet':
        T = torchvision.models.efficientnet_b0(pretrained=True)
        T.classifier = nn.Sequential(OrderedDict({
            f'dropout': nn.Dropout(p=0.2, inplace=True),
            f'fc': nn.Linear(1280, class_num, bias=True)
            }))
        T.feature_dim = 1280

    elif config['backbone']=='ConvNet':
        T = ConvNet(config['num_classes'], config['net_width'], config['net_depth'], config['net_act'], config['net_norm'], config['net_pooling'], config['img_size'], 3)


    elif config['backbone']=='CNN_CIFAR10':
        T = CNN_CIFAR10()

    elif config['backbone']=='CNN_FEMNIST':
        T = CNN_FEMNIST()

    elif config['backbone']=='CNN_CelebA':
        T = CNN_CelebA()

    elif config['backbone']=='CharLSTM':
        T = CharLSTM()

    return T
