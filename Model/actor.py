import abc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from Model.feature_model import MLP, Feature, MLP_RELU
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import one_hot
from Model.DDPM import DDPM
from Model.util import regularize, de_regularize, X_MIN, X_MAX, Y_MIN, Y_MAX

class HiFren_planner_TRAIN(nn.Module):
    def __init__(
            self, args, gpu_id,
            noise_lib_X: np.ndarray,
            noise_lib_Y: np.ndarray,
        ):
        super(HiFren_planner_TRAIN, self).__init__()
        self.args = args
        self.Feature_model = Feature(args, gpu_id)
        self.option_model = ActorOption(
            args=args,
        )
        self.actor_model = Actor(
            args=args,
            noise_lib_X=noise_lib_X,
            noise_lib_Y=noise_lib_Y,
        )

    def forward(self, data):
        feature = self.Feature_model(data)
        option = self.option_model(feature, data['last_option'])
        option = option.argmax(dim=-1)
        norm_X, norm_Y = self.actor_model(feature, option)
        return norm_X, norm_Y, option
    
class HiFren_planner_RL(nn.Module):
    def __init__(
            self, args, gpu_id,
            noise_lib_X: np.ndarray,
            noise_lib_Y: np.ndarray,
        ):
        super(HiFren_planner_RL, self).__init__()
        self.args = args
        self.Feature_model = Feature(args, gpu_id)
        self.option_model = ActorOption(
            args=args,
        )
        self.actor_model = Actor(
            args=args,
            noise_lib_X=noise_lib_X,
            noise_lib_Y=noise_lib_Y,
        )

    def forward(self, data):
        feature = self.Feature_model(data)
        option_raw = self.option_model(feature, data['last_option'])
        option = option_raw.argmax(dim=-1)
        norm_X, norm_Y = self.actor_model(feature, option)
        return norm_X, norm_Y, option_raw, feature
    
class Actor(nn.Module):
    def __init__(self,
                args,
                noise_lib_X,
                noise_lib_Y,
            ):
        super(Actor, self).__init__()
        self.actoraction = ActorAction(
            args=args,
            noise_lib_X=noise_lib_X,
            noise_lib_Y=noise_lib_Y
        )
        self.diffusion_model = DiffusionModel(
            args=args
        )

        self.args = args
        
    def forward(self, state, option):
        '''
        返回的是归一化的动作
        '''
        noise_X, noise_Y = self.actoraction(state, option)
        generate_X, generate_Y = self.diffusion_model(state, option, noise_X, noise_Y)
        return generate_X, generate_Y


class ActorAction(nn.Module):
    def __init__(self, args, noise_lib_X, noise_lib_Y):
        super(ActorAction, self).__init__()
        self.args = args
        self.noise_lib_X = torch.tensor(noise_lib_X).float() #(lib_len, action_dim)
        self.noise_lib_Y = torch.tensor(noise_lib_Y).float() #(option_num, lib_len, action_dim)
        self.num_classes = self.noise_lib_Y.shape[0]
        self.lib_len_Y = self.noise_lib_Y.shape[1]
        self.lib_len_X = self.noise_lib_X.shape[0]
        self.embed_layer = nn.Embedding(args.option_num, args.option_embedding_dim)
        self.classify_X = MLP_RELU(
            input_dim=self.args.feature_dim+args.option_embedding_dim,
            hidden_dim=self.args.hidden_dim_classify,
            output_dim=self.lib_len_X,
            layer_num=self.args.layer_num_classify,
        )
        self.classify_Y = MLP_RELU(
            input_dim=self.args.feature_dim+args.option_embedding_dim,
            hidden_dim=self.args.hidden_dim_classify,
            output_dim=self.lib_len_Y,
            layer_num=self.args.layer_num_classify,
        )
    def forward(self, state, option):
        self.noise_lib_Y = self.noise_lib_Y.to(state.device) #(option_num, lib_len, action_dim)
        self.noise_lib_X = self.noise_lib_X.to(state.device) #(lib_len, action_dim)
        option_embed = self.embed_layer(option)
        all_state = torch.cat([state, option_embed], dim=-1)
        classify_out_X = self.classify_X(all_state)
        classify_out_Y = self.classify_Y(all_state)
        mask = one_hot(option, num_classes=self.num_classes).to(torch.float)
        tmp_Y = torch.einsum("bi,mij->bmj", classify_out_Y, self.noise_lib_Y) #(batch_size, option_num, action_dim)
        generate_noise_Y = torch.einsum("bmj,bm->bj", tmp_Y, mask) #(batch_size, action_dim)
        generate_noise_X = torch.einsum("bi,ij->bj", classify_out_X, self.noise_lib_X)
        return generate_noise_X, generate_noise_Y #, cosh, sinh

class DiffusionModel(nn.Module):
    def __init__(self, args):
        super(DiffusionModel, self).__init__()
        self.args = args
        self.embed_layer = nn.Embedding(args.option_num, args.option_embedding_dim)
        self.sde_X = DDPM(
            state_dim=self.args.feature_dim+args.option_embedding_dim,
            action_dim=int(self.args.action_dim/2),
            hidden_dim=self.args.sde_hidden_dim,
            layer_num=self.args.sde_layer_num,
            num_steps=self.args.sde_T,
        )
        self.sde_Y = DDPM(
           state_dim=self.args.feature_dim+args.option_embedding_dim,
            action_dim=int(self.args.action_dim/2),
            hidden_dim=self.args.sde_hidden_dim,
            layer_num=self.args.sde_layer_num,
            num_steps=self.args.sde_T,
        )

    # def forward(self, state, option, a0_X, a0_Y):
    #     '''
    #     用于扩散模型的bc训练
    #     '''
    #     a0X = regularize(a0_X, X_MIN, X_MAX).to(state.device)
    #     a0Y = regularize(a0_Y, Y_MIN, Y_MAX).to(state.device)
    #     embed_option = self.embed_layer(option)
    #     all_state = torch.cat([state, embed_option], dim=-1)
    #     return self.sde_X(a0X, all_state) + self.sde_Y(a0Y, all_state)
    
    def forward(self, state, option, noise_X, noise_Y):
        '''
        采样函数
        '''
        
        option_embed = self.embed_layer(option)
        all_state = torch.cat([state, option_embed], dim=-1)
        X0 = self.sde_X.sample(noise_X, all_state)
        Y0 = self.sde_Y.sample(noise_Y, all_state)
        return X0, Y0

class ActorOption(nn.Module):
    def __init__(self, args):
        super(ActorOption, self).__init__()
        self.args = args
        self.embed_layer = nn.Embedding(args.option_num + 1, args.option_embedding_dim)
        self.classify = MLP(
            input_dim=self.args.feature_dim+args.option_embedding_dim,
            hidden_dim=self.args.hidden_dim_option,
            output_dim=self.args.option_num,
            layer_num=self.args.layer_num_option,
        )

    def forward(self, state, last_option):
        '''
        state: (batch_size, feature_dim)
        last_option: (batch_size, )
        '''
        option_embed = self.embed_layer(last_option)
        all_state = torch.cat([state, option_embed], dim=-1)
        classify_out = self.classify(all_state)
        return classify_out # (batch_size, option_num)
