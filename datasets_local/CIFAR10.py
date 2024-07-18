import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as tfs
from PIL import Image
from torchvision.datasets import USPS, MNIST, SVHN
from torch.utils.data import ConcatDataset, random_split

class BaseDataset(data.Dataset):
    def __init__(self, dataset_root_dir, mode, domain=None, img_size=None):
        self.root_dir = dataset_root_dir
        self.mode = mode

    def collate_fn(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs).cuda()
        labels = torch.stack(labels).cuda()
        return imgs, labels
        
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index):     
        img, label = self.data[index]
        return img, label  

    def prepare_aux_lists(self):
        self.classes = list(range(10))
        self.num_classes = len(self.classes)
        self.class2idx = {x:i for i, x in enumerate(self.classes)}
        self.idx2class2 = {i:x for i, x in enumerate(self.classes)}
        self.cls2imgidx = [[] for _ in range(self.num_classes)]
        for i, d in enumerate(self.data):
            self.cls2imgidx[d[1]].append(i)

    def get_imgs_from_cls(self, cls, img_num):
        assert img_num>0
        idx_shuffle = np.random.permutation(self.cls2imgidx[cls])[:img_num]
        if img_num==1:
            return self.data[idx_shuffle[0]][0].unsqueeze(0)
        else:
            return torch.stack([self.data[idx][0] for idx in idx_shuffle], 0)

class CIFAR10(BaseDataset):
    def __init__(self, dataset_root_dir, mode):
        BaseDataset.__init__(self, dataset_root_dir, mode)
     
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.RandomCrop(32, 4),
            tfs.RandomHorizontalFlip(),
            tfs.ColorJitter(0.1, 0.1, 0.1, 0.1),
            # tfs.RandomErasing(),
            # tfs.RandomRotation(15),
            tfs.Normalize(mean=self.mean, std=self.std)
        ])

        transform_test= tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize(mean=self.mean, std=self.std)
        ])

        if self.mode=='train':
            self.data = torchvision.datasets.CIFAR10(self.root_dir, train=True, download=True, transform=transform)
        elif self.mode=='test':
            self.data = torchvision.datasets.CIFAR10(self.root_dir, train=False, download=True, transform=transform_test)
        
        self.img_size = 32
        self.img_channel = 3
        # self.prepare_aux_lists()

# code from https://github.com/zhuangdizhu/FedGen/blob/main/data/Mnist/generate_niid_dirichlet.py
def devide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS, min_sample, alpha=0.5, sampling_ratio=0.5):
    min_sample = 10#len(SRC_CLASSES) * min_sample
    min_size = 0 # track minimal samples per user
    ###### Determine Sampling #######
    while min_size < min_sample:
        print("Try to find valid data separation")
        idx_batch=[{} for _ in range(NUM_USERS)]
        samples_per_user = [0 for _ in range(NUM_USERS)]
        max_samples_per_user = sampling_ratio * n_sample / NUM_USERS
        for l in SRC_CLASSES:
            # get indices for all that label
            idx_l = [i for i in range(len(data[l]))]
            np.random.shuffle(idx_l)
            if sampling_ratio < 1:
                samples_for_l = int( min(max_samples_per_user, int(sampling_ratio * len(data[l]))) )
                idx_l = idx_l[:samples_for_l]
                print(l, len(data[l]), len(idx_l))
            # dirichlet sampling from this label
            proportions=np.random.dirichlet(np.repeat(alpha, NUM_USERS))
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

    ###### CREATE USER DATA SPLIT #######
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    Labels=[set() for _ in range(NUM_USERS)]
    print("processing users...")
    for u, user_idx_batch in enumerate(idx_batch):
        for l, indices in user_idx_batch.items():
            if len(indices) == 0: continue
            X[u] += data[l][indices].tolist()
            y[u] += (l * np.ones(len(indices))).tolist()
            Labels[u].add(l)

    return X, y, Labels, idx_batch, samples_per_user

def divide_test_data(NUM_USERS, SRC_CLASSES, test_data, Labels, unknown_test):
    # Create TEST data for each user.
    test_X = [[] for _ in range(NUM_USERS)]
    test_y = [[] for _ in range(NUM_USERS)]
    idx = {l: 0 for l in SRC_CLASSES}
    for user in trange(NUM_USERS):
        if unknown_test: # use all available labels
            user_sampled_labels = SRC_CLASSES
        else:
            user_sampled_labels =  list(Labels[user])
        for l in user_sampled_labels:
            num_samples = int(len(test_data[l]) / NUM_USERS )
            assert num_samples + idx[l] <= len(test_data[l])
            test_X[user] += test_data[l][idx[l]:idx[l] + num_samples].tolist()
            test_y[user] += (l * np.ones(num_samples)).tolist()
            assert len(test_X[user]) == len(test_y[user]), f"{len(test_X[user])} == {len(test_y[user])}"
            idx[l] += num_samples
    return test_X, test_y

def get_datasets(trainset, testset, num_clients):
    num_samples_train_base = int(len(trainset)//num_clients)
    num_samples_test = len(testset)//num_clients    
    train_perm, test_perm = torch.randperm(len(trainset)), torch.randperm(len(testset))
    datas = []
    for i in range(num_clients):

        train_base_idxs = train_perm[i*num_samples_train_base:(i+1)*num_samples_train_base]
        train_idxs = train_base_idxs[:len(train_base_idxs)*0.9]
        val_idxs = train_base_idxs[len(train_base_idxs)*0.9:]
        
        test_idxs = test_perm[i*num_samples_test:(i+1)*num_samples_test]
        
        data = {}
        data['train'] = train_idxs
        data['val'] = val_idxs
        data['test'] = test_idxs    
        datas.append(data)

    return datas

