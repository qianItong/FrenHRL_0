import json
import os
import numpy as np
from HPs.my_args import get_args
import torch
import random
import matplotlib.pyplot as plt
import copy
from Model.DDPM import DDPM

X_MAX = 0
X_MIN = 5
Y_MAX = -1
Y_MIN = 1.0
def regualize(x, min, max):
    now_max = max
    now_min = min
    x = np.array(x)
    for i in range(len(x)):
        x[i] = (x[i] - now_min)*2 / (now_max - now_min)-1  
        now_max += max
        now_min += min
    return x
def de_regualize(x, min, max):
    now_max = max
    now_min = min
    x = np.array(x)
    for i in range(len(x)):
        x[i] = (x[i] + 1) * (now_max - now_min) / 2 + now_min
        now_max += max
        now_min += min
    return x

if __name__ == '__main__':
    args = get_args()
    seed = args.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    save_dir = 'noise_lib_test'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    noise_lib_X = []
    noise_lib_Y = []

    with open('noise_lib/all_X.json') as f:
        x_traces = json.load(f)
    with open('noise_lib/all_Y.json') as f:
        y_traces = json.load(f)

    x_traces = np.array(x_traces)
    y_traces = np.array(y_traces)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_traces = torch.tensor(x_traces, dtype=torch.float32).to(device)
    y_traces = torch.tensor(y_traces, dtype=torch.float32).to(device)


    actor = DDPM(
        state_dim=args.feature_dim,
        action_dim=int(args.action_dim/2),
        hidden_dim=args.sde_hidden_dim,
        layer_num=args.sde_layer_num,
        num_steps=args.sde_T,
        device=device,
    )


    noise_lib_X,_,_ = actor.x_t(x_traces,actor.num_steps-1)
    noise_lib_X = noise_lib_X.tolist()

    for i in range(len(y_traces)):
        noise_Y,_,_ = actor.x_t(y_traces[i],actor.num_steps-1)
        noise_Y = noise_Y.tolist()
        noise_lib_Y.append(noise_Y)

    with open('noise_lib/noise_lib_X.json', 'w') as f:
        json.dump(noise_lib_X, f, indent=4)
    with open('noise_lib/noise_lib_Y.json', 'w') as f:
        json.dump(noise_lib_Y, f, indent=4)
    