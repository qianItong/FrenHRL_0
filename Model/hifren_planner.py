import json
import torch
import torch.nn as nn

from Model.actor import Actor, ActorOption
from Model.feature_model import Feature, MLP

from HPs.my_args import get_args
from Model.util import de_regularize, X_MAX, X_MIN, Y_MAX, Y_MIN


class HiFren_planner(nn.Module):
    def __init__(
            self, args,
            noise_lib_X_path: str,
            noise_lib_Y_path: str,
        ):
        super(HiFren_planner, self).__init__()
        self.args = args
        self.Feature_model = Feature(args)
        self.option_model = ActorOption(
            args=args,
        )
        noise_lib_X = json.load(open(noise_lib_X_path, 'r'))
        noise_lib_Y = json.load(open(noise_lib_Y_path, 'r'))
        self.actor_model = Actor(
            args=args,
            noise_lib_X=noise_lib_X,
            noise_lib_Y=noise_lib_Y,
        )

    def forward(self, data, last_option):
        feature = self.Feature_model(data)
        option = self.option_model(feature, last_option)
        option = option.argmax(dim=-1)
        B = feature.shape[0]
        # 强制选择2
        option = torch.full((B,), 3).to(feature.device)
        norm_X, norm_Y = self.actor_model(feature, option)
        Frent_X = de_regularize(norm_X, X_MAX, X_MIN)
        Frent_Y = de_regularize(norm_Y, Y_MAX, Y_MIN)
        Frenet_trajectory = torch.stack([Frent_X, Frent_Y], dim=2)
        return Frenet_trajectory, option