import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple

import tqdm
from Model.feature_model import Feature
from Model.actor import Actor,ActorOption
from Model.critic import OptionCritic, ActionCritic
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
from torch.utils.tensorboard import SummaryWriter
from RL.policy import OC_policy

class RL_Trainer:
    def __init__(
            self, 
            args, 
            gpu_id,
            train_data: DataLoader,
            valid_data: DataLoader,
            noise_lib_X, 
            noise_lib_Y, 
            checkpoint_path: str = None,
            bc_checkpoint_path: str = None,
            save_path: str = None,
            begin_epoch: int = 0,
        ):
        self.args = args
        self.gpu_id = gpu_id
        self.save_path = save_path
        self.epoch_global = begin_epoch
        os.makedirs(save_path, exist_ok=True)
        
        self.policy = OC_policy(
            args=args,
            noise_lib_X=noise_lib_X,
            noise_lib_Y=noise_lib_Y,
            checkpoint_path=checkpoint_path,
            bc_checkpoint_path=bc_checkpoint_path,
            gpu_id=gpu_id,
        )

        self.train_data = train_data
        self.valid_data = valid_data
        if gpu_id == 0:
            self.writer = SummaryWriter(f'/root/tf-logs/RL_{args.experiment_name}')
        else:
            self.writer = None

    def train(self, total_epoch, val_every, save_every, critic_epochs):
        for epoch in range(total_epoch):
            self.policy.train()
            self.train_data.sampler.set_epoch(epoch)
            average_actor_loss = 0
            average_bc_loss = 0
            average_critic1_loss = 0
            average_critic2_loss = 0
            if self.gpu_id == 0:
                train_bar = tqdm.tqdm(self.train_data, desc='Training')
            else:
                train_bar = self.train_data
            
            train_actor_begin = (epoch >= critic_epochs)
            for batch, next_batch in train_bar:
                result = self.policy.learn(batch, next_batch, train_actor_begin)
                average_actor_loss += result['actor_loss']
                average_bc_loss += result['bc_loss']
                average_critic1_loss += result['critic1_loss']
                average_critic2_loss += result['critic2_loss']
                if self.gpu_id == 0:
                    train_bar.set_postfix(
                        {
                            'actor_loss': result['actor_loss'],
                            'bc_loss': result['bc_loss'],
                            'critic1_loss': result['critic1_loss'],
                            'critic2_loss': result['critic2_loss'],
                        }
                    )
            average_actor_loss /= len(self.train_data)
            average_bc_loss /= len(self.train_data)
            average_critic1_loss /= len(self.train_data)
            average_critic2_loss /= len(self.train_data)

            average_actor_loss_tensor = torch.tensor(average_actor_loss).to(self.gpu_id)
            average_bc_loss_tensor = torch.tensor(average_bc_loss).to(self.gpu_id)
            average_critic1_loss_tensor = torch.tensor(average_critic1_loss).to(self.gpu_id)
            average_critic2_loss_tensor = torch.tensor(average_critic2_loss).to(self.gpu_id)
            dist.all_reduce(average_actor_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(average_bc_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(average_critic1_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(average_critic2_loss_tensor, op=dist.ReduceOp.SUM)
            average_actor_loss = average_actor_loss_tensor.item() / dist.get_world_size()
            average_bc_loss = average_bc_loss_tensor.item() / dist.get_world_size()
            average_critic1_loss = average_critic1_loss_tensor.item() / dist.get_world_size()
            average_critic2_loss = average_critic2_loss_tensor.item() / dist.get_world_size()

            if self.gpu_id == 0:
                self.writer.add_scalars(
                    'loss',
                    {
                        'actor_loss': average_actor_loss,
                        'bc_loss': average_bc_loss,
                        'critic1_loss': average_critic1_loss,
                        'critic2_loss': average_critic2_loss,
                    },
                    self.epoch_global
                )

            if self.gpu_id == 0:
                print(
                    f'Epoch {epoch + 1}/{total_epoch}, '
                    f'Actor Loss: {average_actor_loss:.4f}, '
                    f'BC Loss: {average_bc_loss:.4f}, '
                    f'Critic1 Loss: {average_critic1_loss:.4f}, '
                    f'Critic2 Loss: {average_critic2_loss:.4f}'
                )

            if (epoch + 1) % val_every == 0:
                self.validate(epoch)
            if (epoch + 1) % save_every == 0 and self.gpu_id == 0:
                self.save_model(self.epoch_global)
            self.epoch_global += 1

    def validate(self, epoch):
        self.policy.eval()
        average_bc_loss = 0
        for batch, next_batch in self.valid_data:
            result = self.policy.validate(batch, next_batch)
            loss = result['bc_loss']
            average_bc_loss += loss
        average_bc_loss /= len(self.valid_data)
        average_bc_loss_tensor = torch.tensor(average_bc_loss).to(self.gpu_id)
        dist.all_reduce(average_bc_loss_tensor, op=dist.ReduceOp.SUM)
        average_bc_loss = average_bc_loss_tensor.item() / dist.get_world_size()
        if self.gpu_id == 0:
            self.writer.add_scalar('val/bc_loss', average_bc_loss, self.epoch_global)
            print(f'Validation BC Loss: {average_bc_loss:.4f}')
            
    def save_model(self, epoch):
        self.policy.save_checkpoint(f'{self.save_path}/RL_{epoch}.pth')