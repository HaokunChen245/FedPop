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


def split_data(raw_train, raw_val, niid, seed=1):
    client_num = 100

    if niid==0:
        shuffled_indices = np.random.permutation(len(raw_train))
        split_indices = np.array_split(shuffled_indices, client_num)

        shuffled_indices_val = np.random.permutation(len(raw_val))
        split_indices_val = np.array_split(shuffled_indices_val, client_num)
        trainsets, valsets, testsets = [], [], []
        for i in range(client_num):
            len_train = int(len(split_indices[i]) * 0.9)
            trainsets.append(torch.utils.data.Subset(raw_train, split_indices[i][:len_train].tolist()))
            valsets.append(torch.utils.data.Subset(raw_train, split_indices[i][len_train:].tolist()))
            testsets.append(torch.utils.data.Subset(raw_val, split_indices_val[i].tolist()))

        return trainsets, valsets, testsets

    else:
        split_map = dict()

        # container
        client_indices_list = [[] for _ in range(client_num)]
        raw_all = torch.utils.data.ConcatDataset([raw_train, raw_val])
        targets = [d['label'] for d in raw_all]
        # iterate through all classes
        for c in range(len(np.unique(targets))):
            # get corresponding class indices
            target_class_indices = np.where(np.array(targets) == c)[0]

            # shuffle class indices
            np.random.shuffle(target_class_indices)

            # get label retrieval probability per each client based on a Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(niid, client_num))
            proportions = np.array([p * (len(idx) < (len(raw_train) + len(raw_val)) / client_num) for p, idx in zip(proportions, client_indices_list)])

            # normalize
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(target_class_indices)).astype(int)[:-1]

            # split class indices by proportions
            idx_split = np.array_split(target_class_indices, proportions)
            client_indices_list = [j + idx.tolist() for j, idx in zip(client_indices_list, idx_split)]

        # shuffle finally and create a hashmap
        for j in range(client_num):
            np.random.seed(seed); np.random.shuffle(client_indices_list[j])
            if len(client_indices_list[j]) > 10:
                split_map[j] = client_indices_list[j]

        trainsets, valsets, testsets = [], [], []
        for i in range(client_num):
            len_train = int(len(split_map[i]) * 0.8)
            len_val = int(len(split_map[i]) * 0.1)
            imgs, gts = [], []
            trainsets.append(torch.utils.data.Subset(raw_all, split_map[i][:len_train].tolist()))
            valsets.append(torch.utils.data.Subset(raw_all, split_map[i][len_train:(len_train+len_val)].tolist()))
            testsets.append(torch.utils.data.Subset(raw_all, split_map[i][(len_train+len_val):].tolist()))

        return trainsets, valsets, testsets

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

def get_loader(data, transforms):
    def loader(*args):
        return data[args[0]]
    return loader

def init_loaders_full(
        config
    ):
        loaders = []
        root = '/home/cc/datasets/imagenet-1k'
        img_size = 224
        num_clients = 100
        loaders = []
        imagenet_train = load_dataset('imagenet-1k', split='train').with_format("torch")
        imagenet_train.set_transform(transforms_full_train)
        imagenet_val = load_dataset('imagenet-1k', split='validation').with_format("torch")
        imagenet_val.set_transform(transforms_full_val)
        print(type(imagenet_val))

        trainsets, valsets, testsets = split_data(imagenet_train, imagenet_val, config['niid'])
        for i in range(num_clients):
            data = {
                'train': trainsets[i],
                'val': valsets[i],
                'test': testsets[i]
            }
            loaders.append(data)
        return loaders

def transforms_full_val(examples):
    transforms = tfs.Compose([
        tfs.Resize(224),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
    examples["pixel_values"] = [transforms(image.convert("RGB")) for image in examples["image"]]
    del examples["image"]
    return examples

def transforms_full_train(examples):
    transforms = tfs.Compose([
        tfs.RandomRotation(10),
        tfs.RandomHorizontalFlip(0.5),
        tfs.Resize(256),
        tfs.RandomCrop(224),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    examples["pixel_values"] = [transforms(image.convert("RGB")) for image in examples["image"]]
    del examples["image"]
    return examples

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
                tfs.RandomRotation(20),
                tfs.RandomHorizontalFlip(0.5),
                tfs.ToTensor(),
                tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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