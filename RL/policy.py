import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from typing import Dict
import torch.nn.functional as F
from Model.feature_model import Feature
from Model.actor import HiFren_planner_RL
from torch.nn.parallel import DistributedDataParallel as DDP
from Model.critic import OptionCritic, ActionCritic
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
from Model.util import regularize, de_regularize, X_MEAN, X_STD, Y_MEAN, Y_STD
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, final_lr_ratio=0.2):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(warmup_steps)
        elif step >= total_steps:
            return final_lr_ratio
        else:
            progress = (step - warmup_steps) / float(total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return final_lr_ratio + (1.0 - final_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)

class OC_policy():
    def __init__(self, 
            args,
            noise_lib_X,
            noise_lib_Y,
            checkpoint_path:str = None,
            bc_checkpoint_path:str = None,
            gpu_id:int = 0,
    ) -> None:
        # define models
        self.planner = HiFren_planner_RL(
            args=args,
            gpu_id=gpu_id,
            noise_lib_X=noise_lib_X,
            noise_lib_Y=noise_lib_Y,
        ).to(gpu_id)
        self.critic1 = ActionCritic(
            args=args,
            hidden_dim=args.critic1_hidden_dim,
            layer_num=args.critic1_layer_num,
        ).to(gpu_id)
        torch.manual_seed(args.random_seed + 1) # make sure the critic1 and critic2 have different weights
        self.critic2 = ActionCritic(
            args=args,
            hidden_dim=args.critic2_hidden_dim,
            layer_num=args.critic2_layer_num,
        ).to(gpu_id)

        
        # define old models
        self.planner_old = deepcopy(self.planner).to(gpu_id)
        self.critic1_old = deepcopy(self.critic1).to(gpu_id)
        self.critic2_old = deepcopy(self.critic2).to(gpu_id)
        
        # load checkpoint
        if bc_checkpoint_path is not None and checkpoint_path is None:
            pth = torch.load(bc_checkpoint_path)
            self.planner.load_state_dict(pth)
        if checkpoint_path is not None:
            pth = torch.load(checkpoint_path)
            self.planner.load_state_dict(pth['planner'])
            self.critic1.load_state_dict(pth['critic1'])
            self.critic2.load_state_dict(pth['critic2'])
            self.planner_old.load_state_dict(pth['planner_old'])
            self.critic1_old.load_state_dict(pth['critic1_old'])
            self.critic2_old.load_state_dict(pth['critic2_old'])

        # DDP
        self.planner = DDP(self.planner, device_ids=[gpu_id], find_unused_parameters=True)
        self.critic1 = DDP(self.critic1, device_ids=[gpu_id])
        self.critic2 = DDP(self.critic2, device_ids=[gpu_id])
        self.planner_old = DDP(self.planner_old, device_ids=[gpu_id], find_unused_parameters=True)
        self.critic1_old = DDP(self.critic1_old, device_ids=[gpu_id])
        self.critic2_old = DDP(self.critic2_old, device_ids=[gpu_id])

        # define optimizers
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.planner.parameters()),
            lr=args.lr_actor,
            weight_decay=args.weight_decay,
        )
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps=10, total_steps=args.total_epochs, final_lr_ratio=0.1)
        for _ in range(160):
            self.scheduler.step()
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=args.lr_critic, weight_decay=args.weight_decay)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=args.lr_critic, weight_decay=args.weight_decay)
        if checkpoint_path is not None:
            self.optimizer.load_state_dict(pth['optimizer'])
            self.critic1_optimizer.load_state_dict(pth['critic1_optimizer'])
            self.critic2_optimizer.load_state_dict(pth['critic2_optimizer'])

        self.planner_old.eval()
        self.critic1_old.eval()
        self.critic2_old.eval()

        self.alpha = args.alpha
        self.tau = args.tau
        self.gamma = args.gamma
        self.sync_freq = args.sync_freq
        self.sync_counter = 0
        self.gpu_id = gpu_id
        self._last_loss = 0.0
        self._last_bc_loss = 0.0

    def train(self) -> None:
        self.planner.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.planner.eval()
        self.critic1.eval()
        self.critic2.eval()

    
    def sync_weight(self) -> None:
        for old_param, param in zip(self.planner_old.parameters(), self.planner.parameters()):
            old_param.data.copy_(old_param.data * (1.0 - self.tau) + param.data * self.tau)
        for old_param, param in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            old_param.data.copy_(old_param.data * (1.0 - self.tau) + param.data * self.tau)
        for old_param, param in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            old_param.data.copy_(old_param.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, batch, next_batch, actor_train = True) -> Dict[str, float]:
        action_x_0 = batch['action_x'].to(self.gpu_id)
        action_y_0 = batch['action_y'].to(self.gpu_id)
        action_x = regularize(action_x_0, X_MEAN, X_STD)
        action_y = regularize(action_y_0, Y_MEAN, Y_STD)
        reward = batch['reward'].to(self.gpu_id)
        option = batch['option'].to(self.gpu_id)
        done = batch['done'].to(self.gpu_id)
        action = torch.cat([action_x, action_y], dim=-1)

        # critic update
        with torch.no_grad():
            norm_X, norm_Y, option_raw, state = self.planner_old(batch)
            next_norm_X, next_norm_Y, next_option_raw, next_state = self.planner_old(next_batch)
            next_option = next_option_raw.argmax(dim=-1, keepdim=True)
            next_action = torch.cat([next_norm_X, next_norm_Y], dim=-1)
            next_q1 = self.critic1_old(next_state, next_option, next_action)
            next_q2 = self.critic2_old(next_state, next_option, next_action)
            next_q = torch.min(next_q1, next_q2)
            target_q = reward.unsqueeze(-1) + self.gamma * next_q * (1 - done.unsqueeze(-1))
            target_q = target_q.clamp(-10, 10)
        q1 = self.critic1(state, option, action)
        q2 = self.critic2(state, option, action)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # actor update
        if (self.sync_counter+1) % self.sync_freq == 0:
            if actor_train:
                norm_X, norm_Y, option_raw, state = self.planner(batch)
                predict_action = torch.cat([norm_X, norm_Y], dim=-1)
                predict_option = torch.argmax(option_raw, dim=-1, keepdim=True)
                q = self.critic1(state, predict_option, predict_action)
                _lambda = self.alpha / q.abs().mean().detach()
                original_x = de_regularize(norm_X, X_MEAN, X_STD)
                original_y = de_regularize(norm_Y, Y_MEAN, Y_STD)
                bc_loss = F.mse_loss(original_x, action_x_0) + F.mse_loss(original_y, action_y_0) +  F.cross_entropy(option_raw, option)

                # loss = bc_loss
                loss = -_lambda * q.mean() + bc_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self._last_loss = loss.item()
                self._last_bc_loss = bc_loss.item()
                self.sync_counter = 0
            self.sync_weight()

        self.sync_counter += 1

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': self._last_loss,
            'bc_loss': self._last_bc_loss,
        }
    def validate(self, batch, next_batch) -> Dict[str, float]:
        action_x_0 = batch['action_x'].to(self.gpu_id)
        action_y_0 = batch['action_y'].to(self.gpu_id)
        action_x = regularize(action_x_0, X_MEAN, X_STD)
        action_y = regularize(action_y_0, Y_MEAN, Y_STD)
        option = batch['option'].to(self.gpu_id)
        heading = batch['heading'].to(self.gpu_id)
        norm_X, norm_Y, option_raw, state = self.planner(batch)
        original_x = de_regularize(norm_X, X_MEAN, X_STD)
        original_y = de_regularize(norm_Y, Y_MEAN, Y_STD)
        bc_loss = F.mse_loss(original_x, action_x_0) + F.mse_loss(original_y, action_y_0)  + F.cross_entropy(option_raw, option)
        return {
            'bc_loss': bc_loss.item(),
        }
        
    def save_checkpoint(self, path:str) -> None:
        torch.save({
            'planner': self.planner.module.state_dict(),
            'critic1': self.critic1.module.state_dict(),
            'critic2': self.critic2.module.state_dict(),
            'planner_old': self.planner_old.module.state_dict(),
            'critic1_old': self.critic1_old.module.state_dict(),
            'critic2_old': self.critic2_old.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'scedule': self.scheduler.state_dict(),
        }, path)
