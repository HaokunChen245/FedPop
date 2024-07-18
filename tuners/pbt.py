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
import os
from datetime import datetime
import json
from math import ceil, log

class PBT(TunerBase):
    def __init__(self, config, loaders):
        TunerBase.__init__(self, config, loaders)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.quantile_top = config['quantile_top']
        self.quantile_bottom = config['quantile_bottom']
        self.freq_exploit_explore = config['freq_exploit_explore']
        self.perturb_eps = config['perturb_eps']
        self.resample_probability = config["resample_prob"]
        self.discount_factor = config['discount_factor']
        self.freq_exploit_explore_server = config['freq_exploit_explore_server']
        self.apply_fix_perturb = config['apply_fix_perturb']
        
        self.hyp_setting = {} 
        self.hyp_setting['server'] = self.init_one_server()
        self.hyp_setting['local_base'] = []
        self.local_hyps_per_setting = self.num_active_users
        for i in range(self.local_hyps_per_setting):
            self.hyp_setting[f'local_{i}'] = []
        for i in range(self.num_users):
            if len(self.hyp_setting['local_base'])>0:
                self.hyp_setting['local_base'].append(
                    {'param': self.perturb_one_user(
                        init_user=self.hyp_setting['local_base'][0]['param'],
                        base_user=None,
                        eps=self.perturb_eps,
                        resample_probability=0.0,
                    )}
                )
            else:
                self.hyp_setting['local_base'].append(
                    {'param': self.perturb_one_user(
                        init_user=None,
                        base_user=None
                    )}
                )

            for i in range(self.local_hyps_per_setting):
                self.hyp_setting[f'local_{i}'].append(
                    {'param': self.perturb_one_user(
                        init_user=self.hyp_setting['local_base'][-1]['param'], 
                        base_user=None,
                        eps=self.perturb_eps,
                        resample_probability=0.0,
                    )}
                )
            break

        self.apply_only_active_users = False

        currentDateAndTime = datetime.now()
        self.log_time = currentDateAndTime.strftime("%y%m%d%H%M%S")

    def train(self):
        time_start = datetime.now()
        selected_local_hyp_idxs = np.random.choice(list(range(self.local_hyps_per_setting)), self.num_active_users, replace=False) 
        score_lists = [[] for _ in range(self.num_active_users)]

        for r in range(1, self.total_resources+1):   
            #randomly sampling the users 
            users = [{} for _ in range(self.num_active_users)]
            selected_user_idxs = select_user_idxs(self.num_users, self.num_active_users)
            if self.apply_only_active_users:
                for i in range(len(selected_users)):
                    selected_users[i]['param'] = self.hyps_active_users[i]
            else:
                idx = 0
                for user_idx, local_hyp_idx in zip(selected_user_idxs, selected_local_hyp_idxs):
                    users[idx]['loader'] = self.hyp_setting['server']['loaders'][user_idx]
                    users[idx]['param'] = self.hyp_setting[f'local_{local_hyp_idx}'][0]['param']
                    idx += 1

            self.hyp_setting['server'], L_trains, L_vals, acc_vals, num_data_val_list = self.communication_round(self.hyp_setting['server'], users, j=0, r=r)
            for j, L_val in enumerate(L_vals):
                score_lists[j].append(L_val)
                
            acc_val_batch = np.inner(acc_vals, num_data_val_list) / num_data_val_list.sum()
            time_diff = datetime.now() - time_start
            time_diff_str = ':'.join(str(time_diff).split(':')[:2] + [str(round(float(str(time_diff).split(':')[2]),2))])
            print(f"R:{r}/{self.total_resources} acc_val: {round(acc_val_batch, 4)} ({time_diff_str})")

            # conduct exploit and exploration for bottom q% of clients     
            if r%self.freq_exploit_explore==0:
                scores = []
                for l in score_lists:
                    scores.append(discounted_mean(l, self.discount_factor))
                score_lists = [[] for _ in range(self.num_active_users)]

                top_inds = argsort(scores)[:ceil(self.quantile_top*self.num_active_users)] 
                bottom_inds = argsort(scores)[-ceil(self.quantile_bottom*self.num_active_users):] 
                bottom_new_inds = random.choices(top_inds, k=len(bottom_inds))
                for idx, idx_to_copy in zip(selected_local_hyp_idxs[bottom_inds], selected_local_hyp_idxs[bottom_new_inds]):
                    # perturb the idx
                    for i in range(len(self.hyp_setting['local_base'])):
                        self.hyp_setting[f'local_{idx}'][i]['param'] = self.perturb_one_user(
                            init_user=self.hyp_setting[f'local_{idx_to_copy}'][i]["param"],
                            base_user=self.hyp_setting['local_base'][i]["param"], 
                            eps=self.perturb_eps,
                            resample_probability=self.resample_probability,
                        )
                        self.validate_one_user(self.hyp_setting[f'local_{idx}'][i]['param'])
                        
                selected_local_hyp_idxs = np.random.choice(list(range(self.local_hyps_per_setting)), self.num_active_users, replace=False) 

            if self.apply_only_active_users:
                for i in range(len(selected_users)):
                    self.hyps_active_users[i] = selected_users[i]['param']
  
            # if r%self.freq_exploit_explore==0:
            #     with open(f"./log_{self.log_time}.txt", "a") as f:
            #         f.write(json.dumps({"round": r, "hyps": self.hyps_active_users, "avg_L_train": float(sum(L_trains)/len(L_trains)), "avg_L_val": float(sum(L_vals)/len(L_vals)) }) + '\n')
            # else:
            #     with open(f"./log_{self.log_time}.txt", "a") as f:
            #         f.write(json.dumps({"round": r, "avg_L_train": float(sum(L_trains)/len(L_trains)), "avg_L_val": float(sum(L_vals)/len(L_vals))}) + '\n')

    def test(self):
        acc_glob = super().test(self.hyp_setting['server'], self.hyp_setting['server']['loaders'])
        # torch.save(
        #     self.hyp_setting['server']['net'],
        #     os.path.join(self.folder_dir, "server_model.pt"),
        # )
        return acc_glob 

    # def explore(self, cur_param, mutations, resample_probability = 0.0):
    #     new_param = copy.deepcopy(cur_param)
    #     eps = 0.1
    #     for key, distribution in mutations.items():
    #         if isinstance(distribution, list):            
    #             if (
    #                 random.random() < resample_probability
    #                 or cur_param[key] not in distribution #current config not in the mutation list
    #             ):
    #                 new_param[key] = random.choice(distribution) # select from the list
    #             elif random.random() > 0.5:
    #                 new_param[key] = distribution[
    #                     max(0, distribution.index(cur_param[key]) - 1) # 0.5 prob of selecting the previous option
    #                 ]
    #             else:
    #                 new_param[key] = distribution[
    #                     min(len(distribution) - 1, distribution.index(cur_param[key]) + 1) # 0.5 prob of selecting the next option
    #                 ]
    #         else:
    #             if random.random() < resample_probability:
    #                 new_param[key] = distribution()
    #             elif random.random() > 0.5:
    #                 new_param[key] = cur_param[key] * (1 + eps)  # 0.5 prob of perturbing with 1.2
    #             else:
    #                 new_param[key] = cur_param[key] * (1 - eps)  # 0.5 prob of perturbing with 0.8
    #             if isinstance(cur_param[key], int):
    #                 new_param[key] = int(new_param[key])
    #     return new_param
