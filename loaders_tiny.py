import random
from torch.utils.data import Dataset, DataLoader
from torchvision import models,utils,datasets,transforms
import numpy as np
import sys
import os
from PIL import Image
import torchvision.transforms as tfs
from utils import *
from datasets import load_dataset

def get_loader_tiny(data, transforms):
    def loader(*args):
        output = []
        for arg in args:
            Xarg, Yarg = data[arg]
            Y = torch.Tensor(Yarg).long()
            X = []
            for img in Xarg:
                X.append(transforms[arg](img.convert('RGB')))
            X = torch.stack(X, 0)
            output.append(X.cuda(non_blocking=True))
            output.append(Y.cuda(non_blocking=True))
        return output
    return loader

def split_data(raw_train, raw_val, niid):
    client_num = 50
    if niid==0:
        shuffled_indices = np.random.permutation(len(raw_train))
        split_indices = np.array_split(shuffled_indices, client_num)

        shuffled_indices_val = np.random.permutation(len(raw_val))
        split_indices_val = np.array_split(shuffled_indices_val, client_num)
        trainsets, valsets, testsets = [], [], []
        for i in range(client_num):
            len_train = int(len(split_indices[i]) * 0.9)
            imgs, gts = [], []
            for j in split_indices[i][:len_train]:
                j = int(j)
                imgs.append(raw_train[j]['image'])
                gts.append(raw_train[j]['label'])
            trainsets.append([imgs, gts])

            imgs, gts = [], []
            for j in split_indices[i][len_train:]:
                j = int(j)
                imgs.append(raw_train[j]['image'])
                gts.append(raw_train[j]['label'])
            valsets.append([imgs, gts])

            testset = []
            imgs, gts = [], []
            for j in split_indices_val[i]:
                j = int(j)
                imgs.append(raw_val[j]['image'])
                gts.append(raw_val[j]['label'])
            testsets.append([imgs, gts])

        return trainsets, valsets, testsets

    else:
        split_map = dict()

        # container
        client_indices_list = [[] for _ in range(client_num)]

        # iterate through all classes
        for c in range(200):
            # get corresponding class indices
            target_class_indices = np.where(np.array(raw_train.targets) == c)[0]

            # shuffle class indices
            np.random.shuffle(target_class_indices)

            # get label retrieval probability per each client based on a Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(niid, client_num))
            proportions = np.array([p * (len(idx) < len(raw_train) / client_num) for p, idx in zip(proportions, client_indices_list)])

            # normalize
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(target_class_indices)).astype(int)[:-1]

            # split class indices by proportions
            idx_split = np.array_split(target_class_indices, proportions)
            client_indices_list = [j + idx.tolist() for j, idx in zip(client_indices_list, idx_split)]

        # shuffle finally and create a hashmap
        for j in range(client_num):
            np.random.seed(args.global_seed); np.random.shuffle(client_indices_list[j])
            if len(client_indices_list[j]) > 10:
                split_map[j] = client_indices_list[j]
        return split_map


def init_loaders_tiny(
        config
    ):
        loaders = []
        root = '/home/cc/datasets/tiny-imagenet-200'
        img_size = 64
        transforms = {}
        transforms['val'] = tfs.Compose([tfs.ToTensor(),
                                        tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
        transforms['test'] = transforms['val']
        transforms['train'] = tfs.Compose([
                tfs.transforms.RandomRotation(20),
                tfs.transforms.RandomHorizontalFlip(0.5),
                tfs.transforms.ToTensor(),
                tfs.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        num_clients = 50
        loaders = []
        tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
        tiny_imagenet_val = load_dataset('Maysee/tiny-imagenet', split='valid')
        trainsets, valsets, testsets = split_data(tiny_imagenet_train, tiny_imagenet_val, config['niid'])
        for i in range(num_clients):
            data = {
                'train': trainsets[i],
                'val': valsets[i],
                'test': testsets[i]
            }
            loader = get_loader_tiny(data, transforms)
            loaders.append(loader)
        return loaders