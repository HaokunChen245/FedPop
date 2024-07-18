from pathos.multiprocessing import ProcessingPool as Pool
import torch.multiprocessing as mp
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

from utils import *
import wandb

def conduct_one_opt_client(user_idx, user, net,
                            L_vals, L_trains, acc_vals,
                            num_data_train_list, num_data_val_list
                            ):

    # print(f"PID {os.getpid()} start training, parent PID {os.getppid()}.")
    start_time = time.time()
    if hasattr(net, "dropout"):
        net.dropout.p = user['param']['dropout']
    device = torch.device(f"cuda:{user_idx//2}")
    ce_criterion = nn.CrossEntropyLoss()
    net.to(device)
    # prox_criterion = get_prox(server["net"], ce_criterion, mu = user["param"]["mu"])
    optimizer = optim.SGD(
        net.parameters(),
        lr=user["param"]["lr"],
        momentum=user["param"]["momentum"],
        weight_decay=user["param"]["weight_decay"],
    )
    L_train, len_train = train_multi_epoch(
        user["param"]["epochs"],
        user['loader']['train'],
        net,
        ce_criterion,
        device,
        user["param"]["bs"],
        optimizer,
    )
    L_val, acc_val, len_val = conduct_one_epoch(
        "val", user['loader']['val'], net, ce_criterion, device,
    )

    L_vals[user_idx] = float(L_val)
    L_trains[user_idx] = float(L_train)
    acc_vals[user_idx] = float(acc_val)
    num_data_train_list[user_idx] = len_train
    num_data_val_list[user_idx] = len_val
    end_time = time.time()

    # print(f"PID {os.getpid()} terminates, PPID {os.getppid()}, time: {end_time - start_time} sec.")

def communication_round_mp(server, selected_users, j=0, r=0):
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()

    nc = len(selected_users)
    L_vals = torch.tensor([0.0 for _ in range(nc)]).share_memory_()
    L_trains = torch.tensor([0.0 for _ in range(nc)]).share_memory_()
    acc_vals = torch.tensor([0.0 for _ in range(nc)]).share_memory_()
    num_data_train_list = manager.list([0 for _ in range(nc)])
    num_data_val_list = manager.list([0 for _ in range(nc)])

    # cnets = manager.list([]) # for multi-process
    cnets = []
    for i in range(len(selected_users)):
        cnets.append(copy.deepcopy(server["net"]).to(f'cuda:{i//2}'))
    processes = []
    for idx_user, user in enumerate(selected_users):
        p = mp.Process(target=conduct_one_opt_client,
                        args=( idx_user, user, cnets[idx_user],
                                L_vals, L_trains, acc_vals,
                                num_data_train_list, num_data_val_list
                                ))

        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    user_inds = range(len(selected_users))
    total_num_train = sum(num_data_train_list)
    for idx, ind in enumerate(user_inds):
        num_train = num_data_train_list[ind]
        user_net = cnets[ind].to("cuda:0")
        if idx > 0:
            for agg, param in zip(
                aggregate.parameters(), user_net.parameters()
            ):
                agg.data += num_train * param.data
        else:
            for param in user_net.parameters():
                param.data *= num_train
            aggregate = user_net
    server["opt"].zero_grad()
    for agg, param in zip(aggregate.parameters(), server["net"].parameters()):
        param.grad = param.data - agg / total_num_train
    server["opt"].step()
    server["sched"].step()

    return (
        server,
        np.array(L_trains),
        np.array(L_vals),
        np.array(acc_vals),
        np.array(num_data_val_list),
    )

def get_prox(model, criterion=nn.CrossEntropyLoss(), mu=0.0):

    if not mu:
        return criterion

    mu *= 0.5
    model0 = [param.data.clone() for param in model.parameters()]

    def objective(*args, **kwargs):

        prox = sum((param-param0).pow(2).sum()
                   for param, param0 in zip(model.parameters(), model0))
        return criterion(*args, **kwargs) + mu * prox

    return objective

def sampling_user_lr():
    return 10.0 ** np.random.uniform(low=-4.0, high=0.0)
def sampling_user_momentum():
    return np.random.uniform(low=0.0, high=1.0)
def sampling_user_weight_decay():
    return 10.0 ** np.random.uniform(low=-5.0, high=-1.0)
def sampling_user_mu():
    return 10.0 ** np.random.uniform(low=-5.0, high=0.0)
def sampling_user_dropout():
    return np.random.uniform(low=0.0, high=0.5)
def sampling_server_lr():
    return 10.0 ** np.random.uniform(low=-1.0, high=1.0)
def sampling_server_gamma():
    return 1.0 - 10.0 ** np.random.uniform(low=-4.0, high=-2.0)

class TunerBase:
    def __init__(self, config, loaders):
        self.partial_avg = False
        self.apply_server_hyps = True
        self.loaders = loaders

        self.use_mp = config["use_mp"]
        self.algorithm = config["algorithm"]
        self.dataset = config["dataset"]
        self.device = config["device"]
        self.total_resources = config["total_resources"]
        self.num_users = len(self.loaders)
        self.num_active_users = config["num_active_users"]
        self.net_type = config["net"]
        self.eps_annealing = config['eps_annealing']
        self.apply_acc_for_selection = config['apply_acc_for_selection']
        self.max_resources = config['max_resources']
        self.niid = config['niid']
        self.update_base_hyp = config['update_base_hyp']
        self.run_name = (
            config["algorithm"]
            + "_"
            + config["dataset"]
            + "_"
            + time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        )
        self.folder_dir = (
            f"/home/cc/fedhp/logs/{config['algorithm']}/{self.run_name}"
        )

        self.setting_dict = self.get_setting_dict(config)
        self.SERVER = {
            "lr": sampling_server_lr, #lambda: 10 ** np.random.uniform(low=-1.0, high=1.0),
            "momentum": [0.0, 0.9],
            "step": 1,
            "gamma": sampling_server_gamma,# lambda: 1.0 - 10.0 ** np.random.uniform(low=-4.0, high=-2.0),
        }

        if config["dataset"] == 'imagenet':
            self.USER = {
                "lr": sampling_user_lr, #lambda: 10.0 ** np.random.uniform(low=-4.0, high=0.0),
                "momentum": sampling_user_momentum, #lambda: np.random.uniform(low=0.0, high=1.0),
                "weight_decay": sampling_user_weight_decay, #lambda: 10.0 ** np.random.uniform(low=-5.0, high=-1.0),
                "epochs": 1,
                "bs": [2**i for i in range(4, 8)],
                'mu': lambda: 10.0 ** np.random.uniform(low=-5.0, high=0.0), # used for FedProx weight
                'dropout': lambda: np.random.uniform(low=0.0, high=0.5),
            }
        else:
            self.USER = {
                "lr": sampling_user_lr, #lambda: 10.0 ** np.random.uniform(low=-4.0, high=0.0),
                "momentum": sampling_user_momentum, #lambda: np.random.uniform(low=0.0, high=1.0),
                "weight_decay": sampling_user_weight_decay, #lambda: 10.0 ** np.random.uniform(low=-5.0, high=-1.0),
                "epochs": [i for i in range(1, 6)] if config["dataset"] not in ['shakespeare', 'PACS', 'DomainNet', 'rxrx1', 'OfficeHome', 'OfficeCaltech', 'imagenet'] else 1,
                "bs": [2**i for i in range(3, 8)] if config["dataset"] not in ['PACS', 'DomainNet', 'rxrx1', 'OfficeHome', 'OfficeCaltech'] else [2**i for i in range(3, 7)],
                'mu': sampling_user_mu, #lambda: 10.0 ** np.random.uniform(low=-5.0, high=0.0), # used for FedProx weight
                'dropout': sampling_user_dropout, #lambda: np.random.uniform(low=0.0, high=0.5),
            }

    def init_one_server(self, net=None):
        server = {}
        server["param"] = {}
        for k, v in self.SERVER.items():
            if isinstance(v, list):
                server["param"][k] = np.random.choice(v)
            elif callable(v):
                server["param"][k] = v()
            else:
                server["param"][k] = v
        if net is not None:
            server["net"] = copy.deepcopy(net)
        else:
            server["net"] = create_network(self.net_type).to(self.device)
        server["opt"] = optim.SGD(
            server["net"].parameters(),
            lr=server["param"]["lr"],
            momentum=server["param"]["momentum"],
        )
        server["sched"] = optim.lr_scheduler.StepLR(
            server["opt"], server["param"]["step"], gamma=server["param"]["gamma"]
        )
        server["loaders"] = copy.deepcopy(self.loaders)

        self.validate_one_server(server['param'])
        return server

    def init_one_server_default(self, net=None):
        server = {}
        server["param"] = {
            "lr": 1.0,
            "momentum": 0.9,
            "step": 1,
            "gamma": 0.999,
        }
        if net is not None:
            server["net"] = copy.deepcopy(net)
        else:
            server["net"] = create_network(self.net_type).to(self.device)
        server["opt"] = optim.SGD(
            server["net"].parameters(),
            lr=server["param"]["lr"],
            momentum=server["param"]["momentum"],
        )
        server["sched"] = optim.lr_scheduler.StepLR(
            server["opt"], server["param"]["step"], gamma=server["param"]["gamma"]
        )
        server["loaders"] = copy.deepcopy(self.loaders)
        self.validate_one_server(server['param'])
        return server

    def init_one_user_default(self):
        other_user = {}
        other_user['param'] = {
            "lr": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.001,
            "epochs": 5,
            "bs": 64,
            'mu': 0.01,
            'dropout': 0.25,
        }
        return other_user

    def validate_one_server(self, server):
        if "param" in server.keys():
            server = server["param"]
        server["lr"] = 10 ** np.clip(np.log10(server["lr"]), -1.0, 1.0)
        server["gamma"] = 1.0 - 10 ** np.clip(
            np.log10(1.0 - server["gamma"]), -4.0, -2.0
        )
        server["step"] = 1

    def update_user_base_hyps(self, hyp_setting):
        user_total = {}
        for i in range(self.local_hyps_per_setting):
            user = hyp_setting[f'local_{i}']['param']

            for k in user.keys():
                if k in ['bs', 'epochs']:
                    if k not in user_total.keys():
                        user_total[k] = []
                    user_total[k].append(user[k])
                else:
                    if k not in user_total.keys():
                        user_total[k] = 0.
                    user_total[k] += user[k]

        for k in user.keys():
            if k in ['bs', 'epochs']:
                v = max(set(user_total[k]), key = user_total[k].count)
            else:
                v = user_total[k]/self.local_hyps_per_setting
            hyp_setting[f'local_base']['param'][k] = v

        return hyp_setting[f'local_base']['param']

    def validate_one_user(self, user, base_user=None):
        if "param" in user.keys():
            user = user["param"]
        if base_user and "param" in base_user.keys():
            base_user = base_user["param"]

        eps = self.perturb_eps

        if base_user:
            log_lr_min = np.log10(base_user["lr"]) - 4*eps
            log_lr_max = np.log10(base_user["lr"]) + 4*eps
            user["lr"] = 10 ** np.clip(np.log10(user["lr"]), log_lr_min, log_lr_max)

            momentum_min = base_user["momentum"] - eps
            momentum_max = base_user["momentum"] + eps
            user["momentum"] = np.clip(user["momentum"], momentum_min, momentum_max)

            log_weight_decay_min = np.log10(base_user["weight_decay"]) - 4*eps
            log_weight_decay_max = np.log10(base_user["weight_decay"]) + 4*eps
            user["weight_decay"] = 10 ** np.clip(np.log10(user["weight_decay"]), log_weight_decay_min, log_weight_decay_max)

            if self.dataset not in ['shakespeare', 'PACS', 'DomainNet', 'rxrx1', 'OfficeHome', 'imagenet']:
                epochs_min = base_user["epochs"] - 1
                epochs_max = base_user["epochs"] + 1
                user["epochs"] = np.clip(user["epochs"], epochs_min, epochs_max)

            bs_min = base_user["bs"] / 2
            bs_max = base_user["bs"] * 2
            user["bs"] = np.clip(user["bs"], bs_min, bs_max)

            log_mu_min = np.log10(base_user["mu"]) - 5*eps
            log_mu_max = np.log10(base_user["mu"]) + 5*eps
            user['mu'] = 10 ** np.clip(np.log10(user["mu"]), log_mu_min, log_mu_max)

            dropout_min = base_user['dropout'] - 0.5*eps
            dropout_max = base_user['dropout'] + 0.5*eps
            user['dropout'] = np.clip(user['dropout'], dropout_min, dropout_max)

        user["lr"] = 10 ** np.clip(np.log10(user["lr"]), -4.0, 0.0)
        user["momentum"] = np.clip(user["momentum"], 0, 1.0)
        user["weight_decay"] = 10 ** np.clip(np.log10(user["weight_decay"]), -5.0, -1.0)
        user["epochs"] = np.clip(user["epochs"], 1, 6)
        user["bs"] = int(2 ** np.clip(np.log2(user["bs"]), 3, 8))
        user['mu'] = 10 ** np.clip(np.log10(user["mu"]), -5.0, 0.0)
        user['dropout'] = np.clip(user['dropout'], 0, 0.5)

        return user

    def perturb_one_server(self, init_server, eps=0.1, resample_probability=0.0, apply_fix_perturb=False):
        other_server = copy.deepcopy(init_server)

        if random.random() > resample_probability:
            log_lr = np.log10(other_server["lr"])
            pert = np.random.uniform(2 * -eps, 2 * eps) if not apply_fix_perturb else np.random.choice([2 * -eps, 2 * eps])
            other_server["lr"] = 10 ** (log_lr + pert)

            log_gamma = np.log10(1.0 - other_server["gamma"])
            pert = np.random.uniform(2 * -eps, 2 * eps) if not apply_fix_perturb else np.random.choice([2 * -eps, 2 * eps])
            other_server["gamma"] = 1.0 - 10 ** (log_gamma + pert)

            if random.random() > 0.5:
                other_server["momentum"] = 0.9 - other_server["momentum"]
            else:
                pass
        else:
            other_server = {}
            for k, v in self.SERVER.items():
                if isinstance(v, list):
                    other_server[k] = np.random.choice(v)
                elif callable(v):
                    other_server[k] = v()
                else:
                    other_server[k] = v

        self.validate_one_server(other_server)
        return other_server

    def perturb_one_user(self, init_user, base_user, eps=0.1, resample_probability=0.0, apply_fix_perturb=False):

        p = random.random()

        if (not base_user and not init_user) or (not base_user and p<resample_probability):
            # sampling from random init.
            other_user = {}
            for k, v in self.USER.items():
                if isinstance(v, list):
                    other_user[k] = np.random.choice(v)
                elif callable(v):
                    other_user[k] = v()
                else:
                    other_user[k] = v
            return other_user

        if p > resample_probability:
            # perturb based on the init_user
            other_user = copy.deepcopy(init_user)
        else:
            # perturb based on the base_user
            other_user = copy.deepcopy(base_user)

        log_lr = np.log10(other_user["lr"])
        pert = np.random.uniform(4 * -eps, 4 * eps) if not apply_fix_perturb else np.random.choice([4 * -eps, 4 * eps])
        other_user["lr"] = 10 ** (log_lr + pert)

        log_mu = np.log10(other_user["mu"])
        pert = np.random.uniform(5 * -eps, 5 * eps) if not apply_fix_perturb else np.random.choice([5 * -eps, 5 * eps])
        other_user['mu'] = 10 **  (log_mu + pert)

        pert = np.random.uniform(0.5 * -eps, 0.5 * eps) if not apply_fix_perturb else np.random.choice([0.5 * -eps, 0.5 * eps])
        other_user["dropout"] = other_user["dropout"] + pert

        pert = np.random.uniform(-eps, eps) if not apply_fix_perturb else np.random.choice([-eps, eps])
        other_user["momentum"] = other_user["momentum"] + pert

        log_wd = np.log10(other_user["weight_decay"])
        pert = np.random.uniform(4 * -eps, 4 * eps) if not apply_fix_perturb else np.random.choice([4 * -eps, 4 * eps])
        other_user["weight_decay"] = 10 ** (log_wd + pert)

        if self.dataset not in ['shakespeare', 'PACS', 'DomainNet', 'rxrx1', 'OfficeHome', 'OfficeCaltech', 'imagenet']:
            p = random.random()
            if p <= 1/3:
                pass
            elif p >= 2/3:
                other_user["epochs"] = max(self.USER['epochs'][0], other_user["epochs"]-1)
            else:
                other_user["epochs"] = min(self.USER['epochs'][-1], other_user["epochs"]+1)

        p = random.random()
        if p <= 1/3:
            pass
        elif p >= 2/3:
            other_user["bs"] = max(self.USER['bs'][0], other_user["bs"]/2)
        else:
            other_user["bs"] = min(self.USER['bs'][-1], other_user["bs"]*2)

        other_user = self.validate_one_user(other_user, base_user)
        return other_user

    def init_one_user(self, init_user=None, eps=0.1):
        other_user = {}
        if init_user:  # use init_user to init a new user with parameter perturbation
            other_user['param'] = self.perturb_one_user(init_user['param'], eps=0.1, resample_probability=0.0)
        else:
            other_user["param"] = self.perturb_one_user(None, eps=0.1, resample_probability=1.0)

        return other_user

    def get_setting_dict(self, config):
        out = {}
        for k in config["hps_list"]:
            v = config[k]
            if not v and v != 0:
                continue
            if "/" in str(k) or "/" in str(v):
                continue  # filter out path
            out[k] = v
        return out

    def test(self, dg_net, loaders, setting_idx = -1):
        # test global model
        test_accs_global = []
        test_accs_refine = []
        num_test_list = []
        for idx, loader in enumerate(loaders):
            # compute global accuracy
            net = copy.deepcopy(dg_net)
            if self.use_mp:
                loader = loader['test']
            _, test_acc_glob, len_test = conduct_one_epoch(
                "test", loader, net, self.ce_criterion, self.device
            )
            test_accs_global.append(test_acc_glob * len_test)
            num_test_list.append(len_test)

            if setting_idx!=-1:
                user = self.func_get_one_client_hyp(setting_idx)
            else:
                user = self.func_get_one_client_hyp()
            # compute refine accuracy
            optimizer = optim.SGD(
                net.parameters(),
                lr=user["param"]["lr"],
                momentum=user["param"]["momentum"],
                weight_decay=user["param"]["weight_decay"],
            )
            _ = train_multi_epoch(
                user["param"]["epochs"],
                loader,
                net,
                self.ce_criterion,
                self.device,
                user["param"]["bs"],
                optimizer,
            )
            _, test_acc_refine, len_test = conduct_one_epoch(
                "test", loader, net, self.ce_criterion, self.device
            )
            del net

            test_accs_refine.append(test_acc_refine * len_test)
            print(
                f"Evaluated client {idx}/{self.num_users}  global: {round(float(test_acc_glob * 100),3)}  refine: {round(float(test_acc_refine * 100),3)}"
            )
        wandb.log(
            {
                "Average_test_acc_global": sum(test_accs_global) / sum(num_test_list),
                "Average_test_acc_refine": sum(test_accs_refine) / sum(num_test_list),
            },
            step=0,
        )
        print(
            "Average_test_acc: (global)",
            round(float(sum(test_accs_global) * 100 / sum(num_test_list)), 3),
            "(refine)",
            round(float(sum(test_accs_refine) * 100/ sum(num_test_list)), 3),
        )
        print(self.setting_dict)
        return float(sum(test_accs_global) / sum(num_test_list)), float(sum(test_accs_refine) / sum(num_test_list))

    def communication_round_noavg(self, hyp_setting, selected_users, j=0, r=0):
        L_vals = []
        L_trains = []
        acc_vals = []
        user_net_param_list = []
        num_data_train_list = []
        num_data_val_list = []
        for idx_user, user in enumerate(selected_users):
            net_prox = copy.deepcopy(hyp_setting['nets'][idx_user])
            prox_criterion = get_prox(net_prox, self.ce_criterion, mu = user["param"]["mu"])
            if hasattr(hyp_setting['nets'][idx_user], "dropout"):
                hyp_setting['nets'][idx_user].dropout.p = user['param']['dropout']

            optimizer = optim.SGD(
                hyp_setting['nets'][idx_user].parameters(),
                lr=user["param"]["lr"],
                momentum=user["param"]["momentum"],
                weight_decay=user["param"]["weight_decay"],
            )
            loader = user["loader"]
            L_train, len_train = train_multi_epoch(
                user["param"]["epochs"],
                loader,
                hyp_setting['nets'][idx_user],
                prox_criterion,
                self.device,
                user["param"]["bs"],
                optimizer,
            )
            L_val, acc_val, len_val = conduct_one_epoch(
                "val", loader, hyp_setting['nets'][idx_user], self.ce_criterion, self.device
            )
            L_vals.append(float(L_val))
            L_trains.append(float(L_train))
            acc_vals.append(float(acc_val))
            num_data_val_list.append(len_val)

        return (
            np.array(L_trains),
            np.array(L_vals),
            np.array(acc_vals),
            np.array(num_data_val_list),
        )

    def communication_round(self, server, selected_users, j=0, r=0):
        L_vals = []
        L_trains = []
        acc_vals = []
        user_net_param_list = []
        num_data_train_list = []
        num_data_val_list = []
        for idx_user, user in enumerate(selected_users):
            net = copy.deepcopy(server["net"]).to(self.device)
            prox_criterion = get_prox(server["net"], self.ce_criterion, mu = user["param"]["mu"])
            if hasattr(net, "dropout"):
                net.dropout.p = user['param']['dropout']

            optimizer = optim.SGD(
                net.parameters(),
                lr=user["param"]["lr"],
                momentum=user["param"]["momentum"],
                weight_decay=user["param"]["weight_decay"],
            )
            loader = user["loader"]
            L_train, len_train = train_multi_epoch(
                user["param"]["epochs"],
                loader,
                net,
                prox_criterion,
                self.device,
                user["param"]["bs"],
                optimizer,
            )
            L_val, acc_val, len_val = conduct_one_epoch(
                "val", loader, net, self.ce_criterion, self.device
            )
            user_net_param_list.append(net)
            L_vals.append(float(L_val))
            L_trains.append(float(L_train))
            acc_vals.append(float(acc_val))
            num_data_train_list.append(len_train)
            num_data_val_list.append(len_val)

        if self.partial_avg:  # whether to use only part of the client models
            num_data_train_list = np.array(num_data_train_list)
            top_inds = argsort(L_vals)[: int(self.quantile_top * self.num_active_users)]
            total_num_train = sum(num_data_train_list[top_inds])
            user_inds = top_inds
        else:
            user_inds = range(len(selected_users))
            total_num_train = sum(num_data_train_list)

        if self.apply_server_hyps:
            for idx, ind in enumerate(user_inds):
                user_net = user_net_param_list[ind]
                num_train = num_data_train_list[ind]
                if idx > 0:
                    for agg, param in zip(
                        aggregate.parameters(), user_net.parameters()
                    ):
                        agg.data += num_train * param.data
                else:
                    for param in user_net.parameters():
                        param.data *= num_train
                    aggregate = user_net
            server["opt"].zero_grad()
            for agg, param in zip(aggregate.parameters(), server["net"].parameters()):
                param.grad = param.data - agg / total_num_train
            server["opt"].step()
            server["sched"].step()
        else:
            for key in server["net"].state_dict().keys():
                if "num_batches_tracked" in key:
                    server["net"].state_dict()[key].data.copy_(net.state_dict()[key])
                else:
                    temp = torch.zeros_like(server["net"].state_dict()[key])
                    for ind in user_inds:
                        user_net = user_net_param_list[ind]
                        num_train = num_data_train_list[ind]
                        temp += num_train / total_num_train * user_net.state_dict()[key]
                    server["net"].state_dict()[key].data.copy_(temp)

        return (
            server,
            np.array(L_trains),
            np.array(L_vals),
            np.array(acc_vals),
            np.array(num_data_val_list),
        )
