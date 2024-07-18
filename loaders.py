import copy
import math
import os
import random
import time
from datetime import datetime
from math import ceil, log
from glob import glob

import numpy as np
from torch import optim

from datasets_local import CIFAR10
from utils import *
import wandb


def init_loaders(
        config
    ):
    loaders = []

    if "CIFAR10" in config['dataset']:
        # init trainset for each client
        dataset_dir = f"/home/cc/datasets/{config['dataset'].replace('_scale', '')}"
        trainset = CIFAR10(dataset_dir, mode="train")
        train_ind = list(range(len(trainset)))
        random.shuffle(train_ind)
        testset = CIFAR10(dataset_dir, mode="test")
        test_ind = list(range(len(testset)))
        random.shuffle(test_ind)

        if config['niid']==-1:
            # centralized training
            len_valset = int(len(train_ind) * 0.2)
            trainset_ind = train_ind[len_valset:]
            valset_ind = train_ind[:len_valset]
            testset_ind = test_ind
            inds = (trainset_ind, valset_ind, testset_ind)
            loaders.append(
                get_loader_cifar10(inds, trainset=trainset, testset=testset)
            )

        elif config['niid']>0:
            dataidx_map = lda_partition(trainset, config['num_users'], config['niid'], dataset_dir)
            cur_len_test = len(testset) // config['num_users']
            prev_ind_test = 0

            for idx in range(config['num_users']):
                train_ind = dataidx_map[idx]
                random.shuffle(train_ind)
                # use 20% of training set as validation set
                len_valset = int(len(train_ind) * 0.2)

                trainset_ind = train_ind[len_valset:]
                valset_ind = train_ind[:len_valset]

                testset_ind = test_ind[prev_ind_test : prev_ind_test + cur_len_test]
                prev_ind_test += cur_len_test

                inds = (trainset_ind, valset_ind, testset_ind)
                loaders.append(
                    get_loader_cifar10(inds, trainset=trainset, testset=testset)
                )

        else:
            cur_len = len(trainset) // config['num_users']
            prev_ind = 0
            cur_len_test = len(testset) // config['num_users']
            prev_ind_test = 0

            for idx in range(config['num_users']):
                # random sample
                len_valset = int(
                    cur_len * 0.2
                )  # use 20% of training set as validation set
                trainset_ind = train_ind[prev_ind : prev_ind + (cur_len - len_valset)]
                valset_ind = train_ind[
                    prev_ind + (cur_len - len_valset) : prev_ind + cur_len
                ]
                prev_ind += cur_len

                testset_ind = test_ind[prev_ind_test : prev_ind_test + cur_len_test]
                prev_ind_test += cur_len_test

                inds = (trainset_ind, valset_ind, testset_ind)
                loaders.append(
                    get_loader_cifar10(inds, trainset=trainset, testset=testset)
                )

    elif config['dataset'] == "FEMNIST":

        if config['niid']==-1:
            # Centralized training
            X_train_all = []
            Y_train_all = []
            X_test_all = []
            Y_test_all = []
            DATA = '/home/cc/leaf/data/femnist/data/train'

            for i, fname in enumerate(glob(os.path.join(DATA, '*.json'))):
                if config['debug']>0 and i>config['debug']:
                    break

                print('\rCaching task', i+1, end='')
                with open(fname, 'r') as f_json:
                    data = json.load(f_json)
                fname = fname.replace('train', 'test')
                with open(fname, 'r') as f_json:
                    data_test = json.load(f_json)

                for n in data['users']:
                    X = torch.Tensor(data['user_data'][n]['x'])
                    X_train_all.append(X)
                    Y = torch.LongTensor(data['user_data'][n]['y'])
                    Y_train_all.append(Y)

                    X_test = torch.Tensor(data_test['user_data'][n]['x'])
                    X_test_all.append(X_test)
                    Y_test = torch.LongTensor(data_test['user_data'][n]['y'])
                    Y_test_all.append(Y_test)

            X_train_all = torch.cat(X_train_all, 0)
            Y_train_all = torch.cat(Y_train_all, 0)
            X_test_all = torch.cat(X_test_all, 0)
            Y_test_all = torch.cat(Y_test_all, 0)

            randperm = torch.randperm(len(X_train_all))
            randperm_test = torch.randperm(len(X_test_all))
            print('train size:', len(X_train_all), 'test size:', len(X_test_all))

            task = {}
            task['X'] = X_train_all[randperm[:]]
            task['Y'] = Y_train_all[randperm[:]]

            task_test = {}
            task_test['X'] = X_test_all[randperm_test[:]]
            task_test['Y'] = Y_test_all[randperm_test[:]]
            loader = get_loader_femnist(task, task_test)
            if not loader is None:
                loaders.append(loader)


        elif config['niid']:
            DATA = '/home/cc/leaf/data/femnist/data/train_niid'
            for i, fname in enumerate(glob(os.path.join(DATA, '*.json'))):
                if config['debug']>0 and i>config['debug']:
                    break

                print('\rCaching task', i+1, end='')
                with open(fname, 'r') as f_json:
                    data = json.load(f_json)
                fname = fname.replace('train', 'test')
                with open(fname, 'r') as f_json:
                    data_test = json.load(f_json)

                for n in data['users']:
                    task = {}
                    task['X'] = torch.Tensor(data['user_data'][n]['x'])
                    task['Y'] = torch.LongTensor(data['user_data'][n]['y'])
                    task_test = {}
                    task_test['X'] = torch.Tensor(data_test['user_data'][n]['x'])
                    task_test['Y'] = torch.LongTensor(data_test['user_data'][n]['y'])
                    loader = get_loader_femnist(task, task_test)
                    if not loader is None:
                        loaders.append(loader)
        else:
            DATA = '/home/cc/leaf/data/femnist/data/train'
            for i, fname in enumerate(glob(os.path.join(DATA, '*.json'))):
                if config['debug']>0 and i>config['debug']:
                    break

                print('\rCaching task', i+1, end='')
                with open(fname, 'r') as f_json:
                    data = json.load(f_json)
                fname = fname.replace('train', 'test')
                with open(fname, 'r') as f_json:
                    data_test = json.load(f_json)
                num_users_per_json = 98

                for n in data['users']:
                    X = torch.Tensor(data['user_data'][n]['x'])
                    Y = torch.LongTensor(data['user_data'][n]['y'])
                    X_test = torch.Tensor(data_test['user_data'][n]['x'])
                    Y_test = torch.LongTensor(data_test['user_data'][n]['y'])
                    randperm = torch.randperm(len(Y))
                    randperm_test = torch.randperm(len(Y_test))
                    len_user = len(Y)//num_users_per_json
                    len_user_test = len(Y_test)//num_users_per_json

                    for i in range(num_users_per_json):
                        task = {}
                        task['X'] = X[randperm[i*len_user:min((i+1)*len_user,len(Y))]]
                        task['Y'] = Y[randperm[i*len_user:min((i+1)*len_user,len(Y))]]

                        task_test = {}
                        task_test['X'] = X_test[randperm_test[i*len_user_test:min((i+1)*len_user_test,len(Y_test))]]
                        task_test['Y'] = Y_test[randperm_test[i*len_user_test:min((i+1)*len_user_test,len(Y_test))]]
                        loader = get_loader_femnist(task, task_test)
                        if not loader is None:
                            loaders.append(loader)

    # Task is too easy and the training is too random, ignore CelebA
    # elif config['dataset'] == "CelebA":
    #     if config['niid']:
    #         fname = '/home/cc/leaf/data/celeba/data/train_niid/all_data_niid_0_keep_5_train_8.json'
    #         with open(fname, 'r') as f_json:
    #             data = json.load(f_json)
    #         fname_test = '/home/cc/leaf/data/celeba/data/test_niid/all_data_niid_0_keep_5_test_8.json'
    #         with open(fname_test, 'r') as f_json_test:
    #             data_test = json.load(f_json_test)

    #         for uid, n in enumerate(data['users']):
    #             if config['debug']>0 and uid>config['debug']:
    #                 break
    #             task = {}
    #             task['X'] = data['user_data'][n]['x']
    #             task['Y'] = torch.LongTensor([int(i) for i in data['user_data'][n]['y']])
    #             task_test = {}
    #             task_test['X'] = data_test['user_data'][n]['x']
    #             if len(data_test['user_data'][n]['x'])<=5:
    #                 continue
    #             task_test['Y'] = torch.LongTensor([int(i) for i in data_test['user_data'][n]['y']])
    #             loader = get_loader_celeba(task, task_test)
    #             if not loader is None:
    #                 loaders.append(loader)
    #     else:
    #         pass

    elif config['dataset'] == "shakespeare":
        from collections import defaultdict
        import string
        portion = 0.1 #small sized dataset

        CHARMAP = defaultdict(lambda: 1)
        CHARMAP.update({char: i+2 for i, char in enumerate(string.printable)})
        def process_chars(s):
            if len(s[0])>1:
                o = torch.zeros(len(s), 80).long()
                for i in range(len(s)):
                    for j in range(len(s[i])):
                        o[i][j] = CHARMAP[s[i][j]]
            else:
                o = torch.empty(len(s)).long()
                for i in range(len(s)):
                    o[i] = CHARMAP[s[i]]
            return o

        if config['niid']==-1:
            #Centralized training

            fname = '/home/cc/leaf/data/shakespeare/data/train/all_data_iid_01_0_keep_0_train_9.json'
            with open(fname, 'r') as f_json:
                data = json.load(f_json)
            fname_test = '/home/cc/leaf/data/shakespeare/data/test/all_data_iid_01_0_keep_0_test_9.json'
            with open(fname_test, 'r') as f_json_test:
                data_test = json.load(f_json_test)

            task = {'X': [], 'Y': []}
            task_test = {'X': [], 'Y': []}
            for uid, n in enumerate(data['users']):
                print(f"\rProcessing user {uid}/{len(data['users'])}", end='')
                length_train = int(portion * len(data['user_data'][n]['x']))
                ind_train = torch.randperm(len(data['user_data'][n]['x']))[:length_train]
                length_test = int(portion * len(data_test['user_data'][n]['x']))
                ind_test = torch.randperm(len(data_test['user_data'][n]['x']))[:length_test]

                data['user_data'][n]['x'] = [data['user_data'][n]['x'][i] for i in ind_train]
                task['X'].append(process_chars(data['user_data'][n]['x']))
                data['user_data'][n]['y'] = [data['user_data'][n]['y'][i] for i in ind_train]
                task['Y'].append(process_chars(data['user_data'][n]['y']))

                data_test['user_data'][n]['x'] = [data_test['user_data'][n]['x'][i] for i in ind_test]
                task_test['X'].append(process_chars(data_test['user_data'][n]['x']))
                data_test['user_data'][n]['y'] = [data_test['user_data'][n]['y'][i] for i in ind_test]
                task_test['Y'].append(process_chars(data_test['user_data'][n]['y']))

            X = torch.cat(task['X'], 0)
            Y = torch.cat(task['Y'], 0)
            X_test = torch.cat(task_test['X'], 0)
            Y_test = torch.cat(task_test['Y'], 0)

            randperm = torch.randperm(len(Y))
            randperm_test = torch.randperm(len(Y_test))
            print('trainset size:', len(Y), 'testset size:', len(Y_test))

            task = {}
            task['X'] = X[randperm[:]]
            task['Y'] = Y[randperm[:]]

            task_test = {}
            task_test['X'] = X_test[randperm_test[:]]
            task_test['Y'] = Y_test[randperm_test[:]]

            loader = get_loader_femnist(task, task_test)
            if not loader is None:
                loaders.append(loader)

        elif config['niid']:
            fname = '/home/cc/leaf/data/shakespeare/data/train_niid/all_data_niid_0_keep_0_train_9.json'
            with open(fname, 'r') as f_json:
                data = json.load(f_json)
            fname_test = '/home/cc/leaf/data/shakespeare/data/test_niid/all_data_niid_0_keep_0_test_9.json'
            with open(fname_test, 'r') as f_json_test:
                data_test = json.load(f_json_test)

            s = 0
            for uid, n in enumerate(data['users']):
                if config['debug']>0 and uid>config['debug']:
                    break
                print(f"\rProcessing user {uid}/{len(data['users'])}", end='')
                length_train = int(portion * len(data['user_data'][n]['x']))
                ind_train = torch.randperm(len(data['user_data'][n]['x']))[:length_train]
                length_test = int(portion * len(data_test['user_data'][n]['x']))
                ind_test = torch.randperm(len(data_test['user_data'][n]['x']))[:length_test]

                task = {'X': [], 'Y': []}
                data['user_data'][n]['x'] = [data['user_data'][n]['x'][i] for i in ind_train]
                task['X'] = process_chars(data['user_data'][n]['x'])
                data['user_data'][n]['y'] = [data['user_data'][n]['y'][i] for i in ind_train]
                task['Y'] = process_chars(data['user_data'][n]['y'])

                task_test = {'X': [], 'Y': []}
                data_test['user_data'][n]['x'] = [data_test['user_data'][n]['x'][i] for i in ind_test]
                if len(data_test['user_data'][n]['x'])<=10:
                    continue
                task_test['X'] = process_chars(data_test['user_data'][n]['x'])
                data_test['user_data'][n]['y'] = [data_test['user_data'][n]['y'][i] for i in ind_test]
                task_test['Y'] = process_chars(data_test['user_data'][n]['y'])
                s+=len(task['Y'])

                loader = get_loader_femnist(task, task_test)
                if not loader is None:
                    loaders.append(loader)
            print(s)

        else:
            fname = '/home/cc/leaf/data/shakespeare/data/train/all_data_iid_01_0_keep_0_train_9.json'
            with open(fname, 'r') as f_json:
                data = json.load(f_json)
            fname_test = '/home/cc/leaf/data/shakespeare/data/test/all_data_iid_01_0_keep_0_test_9.json'
            with open(fname_test, 'r') as f_json_test:
                data_test = json.load(f_json_test)

            task = {'X': [], 'Y': []}
            task_test = {'X': [], 'Y': []}
            for uid, n in enumerate(data['users']):
                print(f"\rProcessing user {uid}/{len(data['users'])}", end='')
                length_train = int(portion * len(data['user_data'][n]['x']))
                ind_train = torch.randperm(len(data['user_data'][n]['x']))[:length_train]
                length_test = int(portion * len(data_test['user_data'][n]['x']))
                ind_test = torch.randperm(len(data_test['user_data'][n]['x']))[:length_test]

                data['user_data'][n]['x'] = [data['user_data'][n]['x'][i] for i in ind_train]
                task['X'].append(process_chars(data['user_data'][n]['x']))
                data['user_data'][n]['y'] = [data['user_data'][n]['y'][i] for i in ind_train]
                task['Y'].append(process_chars(data['user_data'][n]['y']))

                data_test['user_data'][n]['x'] = [data_test['user_data'][n]['x'][i] for i in ind_test]
                task_test['X'].append(process_chars(data_test['user_data'][n]['x']))
                data_test['user_data'][n]['y'] = [data_test['user_data'][n]['y'][i] for i in ind_test]
                task_test['Y'].append(process_chars(data_test['user_data'][n]['y']))

            X = torch.cat(task['X'], 0)
            Y = torch.cat(task['Y'], 0)
            X_test = torch.cat(task_test['X'], 0)
            Y_test = torch.cat(task_test['Y'], 0)
            num_users = 660

            randperm = torch.randperm(len(Y))
            randperm_test = torch.randperm(len(Y_test))
            len_user = len(Y)//num_users
            print(len_user)
            len_user_test = len(Y_test)//num_users

            for i in range(num_users):
                task = {}
                task['X'] = X[randperm[i*len_user:min((i+1)*len_user,len(Y))]]
                task['Y'] = Y[randperm[i*len_user:min((i+1)*len_user,len(Y))]]

                task_test = {}
                task_test['X'] = X_test[randperm_test[i*len_user_test:min((i+1)*len_user_test,len(Y_test))]]
                task_test['Y'] = Y_test[randperm_test[i*len_user_test:min((i+1)*len_user_test,len(Y_test))]]

                loader = get_loader_femnist(task, task_test)
                if not loader is None:
                    loaders.append(loader)

    config['num_users'] = len(loaders)
    print()
    print('Found users:', config['num_users'])
    return loaders
