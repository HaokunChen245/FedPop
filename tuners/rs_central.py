import yaml
from utils import *
import copy
import torch.nn as nn
import torch
import random
import numpy as np
from torch import optim
import time
from datetime import datetime
from tuners.base import TunerBase
from itertools import product
import os
from math import ceil, log, isnan

class RS_Central(TunerBase):
    def __init__(self, config, loaders):
        TunerBase.__init__(self, config, loaders)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.num_hyp_setting = config['num_hyp_setting']
        self.config_net = config['net']
        self.perturb_eps = config['perturb_eps']
        self.apply_fix_perturb = config['apply_fix_perturb']
        self.apply_server_hyps = True
        self.partial_avg = False
        self.apply_only_active_users = False
        self.init_hyp_settings()
        self.total_resources = config['total_resources'] #number of overall communication rounds

    def func_get_one_client_hyp(self):
        for j, hyp_setting in enumerate(self.hyp_settings):
            return hyp_setting[f'local_base']

    def init_hyp_settings(self):
        self.hyp_settings = []
        for _ in range(self.num_hyp_setting):
            hyp_setting = {} 
            hyp_setting['server'] = self.init_one_server()
            hyp_setting['acc_list'] = []
            hyp_setting['local_base'] = {
                'param': self.perturb_one_user(
                    init_user=None,
                    base_user=None
            )}
            self.hyp_settings.append(hyp_setting)

    def train(self):
        time_start = datetime.now()
        cur_comm = 0
        for j, hyp_setting in enumerate(self.hyp_settings):
            for r in range(self.max_resources):
                cur_comm += 1    
                if len(hyp_setting['acc_list'])>=5 and sum(hyp_setting['acc_list'][:5])/5 - hyp_setting['acc_list'][0]<0.01 and hyp_setting['acc_list'][0]<0.2:   
                    continue              

                net = hyp_setting['server']["net"].to(self.device)
                if hasattr(net, "dropout"):
                    net.dropout.p = hyp_setting[f'local_base']['param']['dropout']
                optimizer = optim.SGD(
                    net.parameters(),
                    lr=hyp_setting[f'local_base']['param']["lr"],
                    momentum=hyp_setting[f'local_base']['param']["momentum"],
                    weight_decay=hyp_setting[f'local_base']['param']["weight_decay"],
                )
                loader = hyp_setting['server']['loaders'][0]
                L_train, len_train = train_multi_epoch(
                    hyp_setting[f'local_base']['param']["epochs"],
                    loader,
                    net,
                    self.ce_criterion,
                    self.device,
                    hyp_setting[f'local_base']['param']["bs"],
                    optimizer,
                )
                with torch.no_grad():
                    L_val, acc_val, len_val = conduct_one_epoch(
                        "val", loader, net, self.ce_criterion, self.device, bs=128
                    )

                hyp_setting['acc_list'].append(acc_val)
                
                # use the most recent L_val to compute score
                time_diff = datetime.now() - time_start
                time_diff_str = ':'.join(str(time_diff).split(':')[:2] + [str(round(float(str(time_diff).split(':')[2]),2))])
                print(f"E:{self.num_hyp_setting}|R:{r+1}/{self.max_resources}({cur_comm}/{self.total_resources})| Hyp_{j}: acc_val: {round(float(acc_val), 4)} ({time_diff_str})")

    def test(self):     
        acc_globs, acc_refines = [], []   
        for j, hyp_setting in enumerate(self.hyp_settings):
            acc_glob, acc_refine = super().test(hyp_setting['server']['net'], hyp_setting['server']['loaders'])
            acc_globs.append(acc_glob)
            acc_refines.append(acc_refine)
        i = argsort(acc_globs)[-1]

        t = copy.deepcopy(self.hyp_settings[i])
        del self.hyp_settings
        self.hyp_settings = [t]
        acc_glob, acc_refine = super().test(self.hyp_settings[0]['server']['net'], self.hyp_settings[0]['server']['loaders'])
        return acc_glob, acc_refine
