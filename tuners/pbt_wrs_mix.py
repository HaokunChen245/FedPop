import copy
import json
import os
import random
import time
from datetime import datetime
from itertools import product
from math import ceil, log, isnan
from tuners.base import communication_round_mp

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import optim

import wandb
from tuners.pbt import PBT
from tuners.sha import SHA
from tuners.base import TunerBase
from utils import *

import numpy as np

class PBT_wRS_Mix(TunerBase):
    def __init__(self, config, loaders):
        PBT.__init__(self, config, loaders)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.num_hyp_setting = config["num_hyp_setting"]
        self.config_net = config["net"]
        self.max_rounds_local_perturb = config['max_rounds_local_perturb']
        self.use_batch = config['use_batch']
        self.init_hyp_settings()
        self.update_base_hyp = config['update_base_hyp']

        self.total_resources = config[
            "total_resources"
        ]  # number of overall communication rounds
        self.local_hyps_per_setting = self.num_active_users

        currentDateAndTime = datetime.now()
        self.log_time = currentDateAndTime.strftime("%y%m%d%H%M%S")
        
    def func_get_one_client_hyp(self, setting_idx):
        lidx = self.hyp_settings[setting_idx]["idx_selected"]
        return self.hyp_settings[setting_idx][f'local_{lidx}']

    def init_hyp_settings(self):
        self.hyp_settings = []
        for _ in range(self.num_hyp_setting):
            hyp_setting = {} 
            hyp_setting['server'] = self.init_one_server()
            hyp_setting['num_rounds'] = 0
            hyp_setting["idx_selected"] = 0
            hyp_setting['local_base'] = {'param': self.perturb_one_user(
                    init_user=None,
                    base_user=None
                )}
            hyp_setting['score_lists'] = [[] for _ in range(self.local_hyps_per_setting)]
            for i in range(self.local_hyps_per_setting):
                hyp_setting[f'local_{i}'] = {
                    'param': self.perturb_one_user(
                        init_user=hyp_setting['local_base']['param'], 
                        base_user=None,
                        eps=self.perturb_eps,
                        resample_probability=0.0,
                        apply_fix_perturb=self.apply_fix_perturb
                    )
                } 
            self.hyp_settings.append(hyp_setting)

    # TODO: add perturbation also for server hyperparameters
    def train(self):
        time_start = datetime.now()
        cur_comm = 0
        
        for r in range(self.max_resources):
            scores_setting = [float('inf') for _ in range(self.num_hyp_setting)]
            for j, hyp_setting in enumerate(self.hyp_settings):                                            
                hyp_setting['num_rounds'] += 1
                cur_comm += 1
                if hyp_setting['num_rounds']>self.max_rounds_local_perturb  and self.max_rounds_local_perturb>0:
                    if self.update_base_hyp:
                        self.hyp_settings[j][f'local_base']['param'] = self.update_user_base_hyps(self.hyp_settings[j])

                users = [{} for _ in range(self.num_active_users)]
                selected_user_idxs = select_user_idxs(self.num_users, self.num_active_users)
                for idx, user_idx in enumerate(selected_user_idxs):
                    users[idx]['loader'] = hyp_setting['server']['loaders'][user_idx]
                    if self.use_batch==False and (hyp_setting['num_rounds']>self.max_rounds_local_perturb or self.max_rounds_local_perturb<0):
                        users[idx]['param'] = hyp_setting[f'local_{hyp_setting["idx_selected"]}']['param']
                    else:
                        users[idx]['param'] = hyp_setting[f'local_{idx}']['param']

                if self.use_mp:
                    (
                        hyp_setting["server"],
                        L_trains,
                        L_vals,
                        acc_vals,
                        num_data_val_list,
                    ) = communication_round_mp(
                        hyp_setting["server"], users, j, r
                    )
                else:
                    (
                        hyp_setting["server"],
                        L_trains,
                        L_vals,
                        acc_vals,
                        num_data_val_list,
                    ) = self.communication_round(
                        hyp_setting["server"], users, j, r
                    )
                torch.save(hyp_setting["server"]["net"].state_dict(), f"models/imagenet_pbt_hyp_{j}_{self.niid}.pth")
                
                if self.apply_acc_for_selection:
                    for k, acc_val in enumerate(acc_vals):
                        hyp_setting['score_lists'][k].append(1. - acc_val)
                else:
                    for k, L_val in enumerate(L_vals):
                        hyp_setting['score_lists'][k].append(L_val)


                # use the most recent L_val to compute score
                acc_val_batch = (
                    np.inner(acc_vals, num_data_val_list) / num_data_val_list.sum()
                )
                scores_setting[j] = np.inner(L_vals, num_data_val_list) / num_data_val_list.sum()
                wandb.log(
                    {
                        f"Hyp_{j}/avg_L_train": float(
                            sum(L_trains) / len(L_trains)
                        ),
                        f"Hyp_{j}/avg_L_val": float(sum(L_vals) / len(L_vals)),
                    },
                    step=r,
                )
                time_diff = datetime.now() - time_start
                time_diff_str = ":".join(
                    str(time_diff).split(":")[:2]
                    + [str(round(float(str(time_diff).split(":")[2]), 2))]
                )
                print(
                    f"E:{self.num_hyp_setting}|R:{r+1}/{self.max_resources}({cur_comm}/{self.total_resources})| Hyp_{j}: acc_val: {round(acc_val_batch, 4)} ({time_diff_str})"
                )

                # avoid the drawbacks of nan at any local clients:
                if isnan(sum(L_vals)):
                    print(f"found nan, reinit the hyps of hyp_setting {j}")
                    active_idxs = [idx for idx, hyp_setting in enumerate(self.hyp_settings)]
                    tgt_idx = random.choice(active_idxs)
                    self.perturb_hyp_setting_pair(j, tgt_idx, r)
                    continue

                    # if not working, change this!!!!
                    # print(f"found nan, reinit the hyps of hyp_setting {j}")
                    # temp = copy.deepcopy(scores_setting)
                    # if scores_setting[argsort(temp)[0]]<10 ** 9:
                    #     tgt_idx = argsort(temp)[0]
                    # else:
                    #     active_idxs = [idx for idx, hyp_setting in enumerate(self.hyp_settings) if hyp_setting["active"] and idx!=j]
                    #     tgt_idx = random.choice(active_idxs)
                    # self.perturb_hyp_setting_pair(j, tgt_idx, r)
                    # continue


                # conduct exploit and exploration for bottom q% of clients
                if hyp_setting['num_rounds'] % self.freq_exploit_explore == 0 and hyp_setting['num_rounds']>0 and hyp_setting['num_rounds']<=self.max_rounds_local_perturb and self.max_rounds_local_perturb>0:                        
                    scores = []
                    print("Perturbing client hyps...")
                    for l in hyp_setting['score_lists']:
                        score = discounted_mean(l, factor=0.9)
                        if isnan(score):
                            score = 10**9
                        scores.append(score)                        
                    hyp_setting['score_lists'] = [[] for _ in range(self.local_hyps_per_setting)]

                    argsort_inds = argsort(scores)
                    hyp_setting['idx_selected'] = argsort_inds[0]
                    top_inds = argsort_inds[: ceil(self.quantile_top * len(argsort_inds))]
                    bottom_inds = argsort_inds[
                        -ceil(self.quantile_bottom * len(argsort_inds)) :
                    ]

                    bottom_new_inds = random.choices(top_inds, k=len(bottom_inds))

                    for idx, idx_to_copy in zip(bottom_inds, bottom_new_inds):
                        hyp_setting[f'local_{idx}']['param'] = self.perturb_one_user(
                            init_user=hyp_setting[f'local_{idx_to_copy}']["param"],
                            base_user=hyp_setting['local_base']["param"], 
                            eps = annealing(self.perturb_eps, hyp_setting['num_rounds'], self.max_rounds_local_perturb, self.eps_annealing),
                            resample_probability=self.resample_probability,
                            apply_fix_perturb=self.apply_fix_perturb                                    
                        )
                        

                if self.freq_exploit_explore_server>0 and r % self.freq_exploit_explore_server == 0 and r>0:
                    print("Perturbing server hyps...")
                    for i in range(len(scores_setting)):
                        if isnan(scores_setting[i]):
                            scores_setting[i] = 10**9

                    scores_inds = argsort(copy.deepcopy(scores_setting))[:self.num_hyp_setting]  # length: 27 -> 9 ->
                    top_inds = scores_inds[: ceil(self.quantile_top * self.num_hyp_setting)]
                    bottom_inds = scores_inds[
                        -ceil(self.quantile_bottom * self.num_hyp_setting) :
                    ]
                    bottom_new_inds = random.choices(top_inds, k=len(bottom_inds))

                    for idx, idx_to_copy in zip(bottom_inds, bottom_new_inds):
                        self.perturb_hyp_setting_pair(idx, idx_to_copy, r)

    def perturb_hyp_setting_pair(self, idx, idx_to_copy, r):
        self.hyp_settings[idx] = copy.deepcopy(self.hyp_settings[idx_to_copy])
        hyp_setting = self.hyp_settings[idx]
        hyp_setting['num_rounds'] = 0
        
        # init the new user hyperparameter base by perturb or resample
        hyp_setting['local_base']['param'] = self.perturb_one_user(
            init_user=hyp_setting['local_base']["param"],
            base_user=None,
            eps = self.perturb_eps * 2,
            resample_probability=self.resample_probability,
            apply_fix_perturb=self.apply_fix_perturb
        )

        for i in range(self.local_hyps_per_setting):
            hyp_setting[f'local_{i}'] = {
                'param': self.perturb_one_user(
                    init_user=hyp_setting['local_base']['param'], 
                    base_user=None,
                    eps=self.perturb_eps,
                    resample_probability=0.0,
                    apply_fix_perturb=self.apply_fix_perturb
                )
            } 

        # perturb the server hyperparameters
        hyp_setting["server"]["param"] = self.perturb_one_server(
            hyp_setting["server"]["param"],
            eps = annealing(self.perturb_eps, r, self.max_resources, self.eps_annealing),
            resample_probability=self.resample_probability,
            apply_fix_perturb=self.apply_fix_perturb
        )
        server = hyp_setting["server"]

        # applying the new server hyperparameters
        del server["opt"]
        server["opt"] = optim.SGD(
            server["net"].parameters(),
            lr = 0.1
        )
        server["opt"].load_state_dict(self.hyp_settings[idx_to_copy]['server']["opt"].state_dict())
        for g in server["opt"].param_groups:
            g['lr'] = server["param"]["lr"]
            g['momentum'] = server["param"]["momentum"]
        
        del server["sched"]
        server["sched"] = optim.lr_scheduler.StepLR(
            server["opt"],
            server["param"]["step"],
            gamma=server["param"]["gamma"],
        )

    def test(self):     
        acc_globs, acc_refines = [], []   
        for j, hyp_setting in enumerate(self.hyp_settings):
            acc_glob, acc_refine = super().test(hyp_setting['server']['net'], hyp_setting['server']['loaders'], j)
            acc_globs.append(acc_glob)
            acc_refines.append(acc_refine)
        i = argsort(acc_globs)[-1]
        return acc_globs[i], acc_refines[i]
