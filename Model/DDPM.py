import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class SDE_MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, layer_num, t_dim=16):
        super(SDE_MLP, self).__init__()

        self.t_dim = t_dim
        self.a_dim = action_dim
        self.layer_num = layer_num

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential()
        for i in range(layer_num):
            self.mid_layer.add_module(f'hidden_{i}', nn.Linear(input_dim, hidden_dim))
            self.mid_layer.add_module(f'activation_{i}', nn.ReLU())
            input_dim = hidden_dim
        
        self.final_layer = nn.Linear(hidden_dim, action_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, time, state):
        time = time.to(x.device)
        time_embed = self.time_mlp(time)
        y = torch.cat([state, x, time_embed], dim=-1)
        y = self.mid_layer(y)
        y = self.final_layer(y)

        return y
    
class DDPM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, layer_num, num_steps, t_dim=16, device='cuda'):
        super(DDPM, self).__init__()
        self.model = SDE_MLP(state_dim, action_dim, hidden_dim, layer_num, t_dim)
        self.model.to(device)
        self.device = device
        self.num_steps = num_steps
        self.betas = torch.linspace(-6,6,num_steps)  # 使用linespace指定bata的值
        self.betas = torch.sigmoid(self.betas) * (0.5e-1 - 1e-4) + 1e-4 #(0.5e-2 - 1e-5) + 1e-5#torch.sigmoid(self.betas)*(0.05 - 1e-4)+1e-4  #(1e-2 - 1e-5) + 1e-5
        self.betas = self.betas.to(device)  # 将betas移动到指定设备
        self.alphas = 1.0-self.betas

        self.alphas_prod = torch.cumprod(self.alphas,0).to(device)  # 计算每个时间步的alpha积累值
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod).to(device)  # 计算每个时间步的alpha积累值的平方根
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod).to(device)  # 计算每个时间步的1-alpha积累值的平方根
        self.sample_coeff = self.betas/torch.sqrt(1 - self.alphas_prod).to(device)  # 计算每个时间步的采样系数

    def x_t(self, x_0, t=None):
        if t is None:   # 如果没有指定t，则使用最后一个时间步
            t = self.num_steps - 1
        noise = torch.randn_like(x_0)
        alphas_t = self.alphas_bar_sqrt[t]
        alphas_1_m_t = self.one_minus_alphas_bar_sqrt[t]
        mean = alphas_t * x_0
        std = alphas_1_m_t
        xt = alphas_t * x_0 + alphas_1_m_t * noise
        return xt, mean, std  # 在x[0]的基础上添加噪声
    
    def diffusion_loss_fn(self, x_0, state, mode='l2'):
        batch_size = x_0.shape[0]
        half_batch_size = (batch_size+1) // 2
        t = torch.randint(0, self.num_steps, size=(half_batch_size,))
        t = torch.cat([t, self.num_steps - 1 - t], dim=0)[:batch_size]# t的形状（bz）
        t = t.unsqueeze(-1).to(x_0.device)# t的形状（bz,1）

        a = self.alphas_bar_sqrt[t]
        aml = self.one_minus_alphas_bar_sqrt[t]
        e = torch.randn_like(x_0).to(self.device)

        x = x_0 * a + e * aml

        output = self.model(x, t.squeeze(-1), state)

        # 与真实噪声一起计算误差，求平均值
        if mode == 'l2':
            loss = torch.mean((e - output) ** 2)
        elif mode == 'l1':
            loss = torch.mean(torch.abs(e - output))
        else:
            raise ValueError("mode must be 'l2' or 'l1'")
        return loss
    
    def forward(self,x_0, state):
        return self.diffusion_loss_fn(x_0, state)
    
    def sample(self, x_t, state):
        for i in range(self.num_steps-1, -1, -1):
            t = torch.full((x_t.shape[0],), i, device=self.device, dtype=torch.long)
            x_t = x_t - extract(self.sample_coeff, t, x_t.shape) * self.model(x_t, t, state)
            x_t = x_t / self.alphas_bar_sqrt[t].unsqueeze(-1)
        return x_t