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

class SHA_Central(TunerBase):
    def __init__(self, config, loaders):
        TunerBase.__init__(self, config, loaders)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.num_hyp_setting = config['num_hyp_setting']
        self.elim_rate, self.elim_sched = self.get_schedule(config['max_resources'], config['total_resources'])
        self.config_net = config['net']
        self.apply_server_hyps = True
        self.partial_avg = False
        self.apply_only_active_users = False
        self.init_hyp_settings()
        self.total_resources = config['total_resources'] #number of overall communication rounds

    def func_get_one_client_hyp(self):
        for j, hyp_setting in enumerate(self.hyp_settings):
            if not hyp_setting['active']: continue
            return hyp_setting[f'local_base']

    def init_hyp_settings(self):
        self.hyp_settings = []
        for _ in range(self.num_hyp_setting):
            hyp_setting = {} 
            hyp_setting['server'] = self.init_one_server()
            hyp_setting['local_base'] = {
                'param': self.perturb_one_user(
                    init_user=None,
                    base_user=None
                )
            }
            hyp_setting['active'] = True
            hyp_setting['acc_list'] = []
            self.hyp_settings.append(hyp_setting)

    def train(self):
        time_start = datetime.now()
        start, num_active_settings = 0, self.num_hyp_setting
        cur_comm = 0
        for i, stop in enumerate(self.elim_sched):
            scores = [float('inf') for _ in range(self.num_hyp_setting)]
            for j, hyp_setting in enumerate(self.hyp_settings):
                if not hyp_setting['active']: continue
                for r in range(start, stop):                    
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
                    # use the most recent L_val to compute score
                    scores[j] = float(L_val)
                    hyp_setting['acc_list'].append(acc_val)
 
                    time_diff = datetime.now() - time_start
                    time_diff_str = ':'.join(str(time_diff).split(':')[:2] + [str(round(float(str(time_diff).split(':')[2]),2))])
                    print(f"E:{num_active_settings}/{self.num_hyp_setting}|R:{r+1}/{stop}({cur_comm}/{self.total_resources})| Hyp_{j}: acc_val: {round(float(acc_val), 4)} ({time_diff_str})")
                    
            print('eliminating....')
            for i in range(len(scores)):
                if isnan(scores[i]):
                    scores[i] = float('inf')
            scores_inds = argsort(scores) # low -> high            
            num_active_settings //= self.elim_rate
            if num_active_settings>0:
                for i in scores_inds[int(num_active_settings):]:
                    self.hyp_settings[i]['active'] = False
                    self.hyp_settings[i]['users'] = None
            start = stop

    def test(self):        
        for j, hyp_setting in enumerate(self.hyp_settings):
            if not hyp_setting['active']: continue
            acc_glob, acc_refine = super().test(hyp_setting['server']['net'], hyp_setting['server']['loaders'])
        return acc_glob, acc_refine
        
    def get_schedule(
                    self,   
                    max_resources, 
                    total_resources, 
                    elim_rate=3, 
                    num_elim=3, 
                    ):
        '''returns rate and schedule for use by 'successive_elimination'
        Args:
            max_resources: most resources (steps) assigned to single arm
            total_resources: overall resource limit
            elim_rate: multiplicative elimination rate
            num_elim: number of elimination rounds; if 0 runs random search
        Returns:
            elimination rate as an int, elimination schedule as a list, evaluation schedule as a list
        '''

        assert max_resources <= total_resources, "max_resources cannot be greater than total_resources"
        assert elim_rate > 1, "elim_rate must be greater than 1"
        if self.num_hyp_setting==9:
            num_elim = 2

        # e.g., total_resources=4000, max_resources=800 meaning we can conduct 4000 communication rounds in total
        # and maximum communication rounds for one hyperparameter setting is 800
        # for 27 configurations, we have elim_sched=[89, 177, 268, 800], 
        # 89*27 + (177-89)*9 + (268-177)*3 + (800-268) = 4000
 
        diff = total_resources - max_resources
        geos = (elim_rate**(num_elim+1) - 1) / (elim_rate-1)
        u = int(diff / (geos-num_elim-1))
        resources = 0
        v = lambda i: 1 + ceil((diff+(num_elim-geos+elim_rate**i)*u) / (elim_rate**i-1))
        for opt in product(*(range(u, v(i)) for i in reversed(range(1, num_elim+1)))):
            used = max_resources + sum((elim_rate**i-1)*r 
                                    for i, r in zip(reversed(range(1, num_elim+1)), opt))
            if resources <= used <= total_resources:
                best, resources = opt, used
        assert not 0 in best, "invalid: use more resources or fewer eliminations, or increase rate"
        elim_sched = list(np.cumsum(best)) + [max_resources]

        print(elim_sched)      
        return elim_rate, elim_sched
