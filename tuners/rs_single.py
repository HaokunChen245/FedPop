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
from tuners.base import communication_round_mp

class RS_Single(TunerBase):
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
        self.default = False
        if 'default' in config['algorithm']:
            self.default = True
        self.init_hyp_settings()
        self.total_resources = config['total_resources'] #number of overall communication rounds

    def func_get_one_client_hyp(self):
        for j, hyp_setting in enumerate(self.hyp_settings):
            return hyp_setting[f'local_base']

    def init_hyp_settings(self):
        self.hyp_settings = []
        for _ in range(self.num_hyp_setting):
            hyp_setting = {} 
            if not self.default:
                hyp_setting['server'] = self.init_one_server()
                hyp_setting['local_base'] = {
                    'param': self.perturb_one_user(
                        init_user=None,
                        base_user=None
                )}
            else:
                hyp_setting['server'] = self.init_one_server_default()
                hyp_setting['local_base'] = self.init_one_user_default()
            hyp_setting['acc_list'] = []
            hyp_setting['active'] = True
            self.hyp_settings.append(hyp_setting)

    def train(self):
        time_start = datetime.now()
        cur_comm = 0
        max_trial = 50
        for r in range(self.max_resources):
            for j, hyp_setting in enumerate(self.hyp_settings):
                cur_comm += 1    
                users = [{} for _ in range(self.num_active_users)]
                selected_user_idxs = select_user_idxs(self.num_users, self.num_active_users)
                for idx, user_idx in enumerate(selected_user_idxs):
                    users[idx]['loader'] = hyp_setting['server']['loaders'][user_idx]
                    users[idx]['param'] = hyp_setting[f'local_base']['param']

                if self.use_mp:
                    hyp_setting['server'], L_trains, L_vals, acc_vals, num_data_val_list = \
                        communication_round_mp(hyp_setting['server'], users, j, r)
                else:
                    hyp_setting['server'], L_trains, L_vals, acc_vals, num_data_val_list = \
                        self.communication_round(hyp_setting['server'], users, j, r)

                if isnan(sum(L_vals)):
                    print('found nan')
                # if len(hyp_setting['acc_list'])>=max_trial and hyp_setting['acc_list'][-1] - sum(hyp_setting['acc_list'][-max_trial:])/max_trial<0.01:
                    hyp_setting['server'] = self.init_one_server()
                    hyp_setting['local_base'] = {
                        'param': self.perturb_one_user(
                            init_user=None,
                            base_user=None
                    )}
                    hyp_setting['acc_list'] = []

                # use the most recent L_val to compute score
                acc_val_batch = np.inner(acc_vals, num_data_val_list) / num_data_val_list.sum()
                hyp_setting['acc_list'].append(float(acc_val_batch))
                time_diff = datetime.now() - time_start
                time_diff_str = ':'.join(str(time_diff).split(':')[:2] + [str(round(float(str(time_diff).split(':')[2]),2))])
                print(f"E:{self.num_hyp_setting}|R:{r+1}/{self.max_resources}({cur_comm}/{self.total_resources})| Hyp_{j}: acc_val: {round(acc_val_batch, 4)} ({time_diff_str})")

    def test(self):     
        acc_globs, acc_refines = [], []   
        for j, hyp_setting in enumerate(self.hyp_settings):
            if not hyp_setting['active']:
                acc_globs.append(0)
                acc_refines.append(0)
                continue
            acc_glob, acc_refine = super().test(hyp_setting['server']['net'], hyp_setting['server']['loaders'])
            acc_globs.append(acc_glob)
            acc_refines.append(acc_refine)
        i = argsort(acc_globs)[-1]

        t = copy.deepcopy(self.hyp_settings[i])
        del self.hyp_settings
        self.hyp_settings = [t]
        acc_glob, acc_refine = super().test(t['server']['net'], t['server']['loaders'])
        return acc_glob, acc_refine
