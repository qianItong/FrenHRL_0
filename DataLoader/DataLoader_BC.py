import os
from torch.utils.data import Dataset
import json
from collections import OrderedDict
import torch
from mmengine import fileio
import io
import numpy as np

def openjson(path):
    value  = fileio.get_text(path)
    dict = json.loads(value)
    return dict

def opendata(path):
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff)
    return npz_data

def linear_extrapolate_pad(points, target_rows=16):
    n = points.shape[0]
    if n >= target_rows:
        return points
    
    # 计算最后两个点的差值
    last_point = points[-1]
    if n >= 2:
        delta = points[-1] - points[-2]  # 最后两个点的差值
    else:
        delta = 0  # 如果只有1个点，差值为0
    
    # 生成新点：last_point + delta * (i+1)
    new_points = [
        last_point + delta * (i + 1)
        for i in range(target_rows - n)
    ]
    
    # 拼接原数组和新点
    padded_points = np.concatenate([points, new_points])
    return padded_points

class ChunkedDataset_BC(Dataset):
    def __init__(self, data_dir, data_list):
        self.data_dir = data_dir
        self.data_list = openjson(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        file_name = self.data_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        token,num = (file_name.split('.')[0]).split('_')
        num = int(num)
        last_file_name = '{}_{}.npz'.format(token,num-1)
        if os.path.exists(os.path.join(self.data_dir, last_file_name)):
            last_item = opendata(os.path.join(self.data_dir, last_file_name))
            last_option = torch.tensor(last_item['option'], dtype=torch.long)
        else:
            last_option = torch.tensor(7, dtype=torch.long)
        item = opendata(file_path)
        action_x = item['action_x']
        action_y = item['action_y']
        cosh = item['cosh']
        sinh = item['sinh']
        if len(action_x) < 16:
            action_x = linear_extrapolate_pad(action_x)
            action_y = linear_extrapolate_pad(action_y)
            cosh = linear_extrapolate_pad(cosh)
            sinh = linear_extrapolate_pad(sinh)
        data = {
            "ego_current_state": torch.tensor(item['ego_current_state'], dtype=torch.float32),
            "neighbor_agents_past": torch.tensor(item['neighbor_agents_past'], dtype=torch.float32),
            "static_objects": torch.tensor(item['static_objects'], dtype=torch.float32),
            'reference_line': torch.tensor(item['reference_line'], dtype=torch.float32).view(-1),
            'lanes': torch.tensor(item['lanes'], dtype=torch.float32),
            'route_lanes': torch.tensor(item['route_lanes'], dtype=torch.float32),
            'reward': torch.tensor(item['reward'], dtype=torch.float32),
            'action_x': torch.tensor(action_x, dtype=torch.float32),
            'action_y': torch.tensor(action_y, dtype=torch.float32),
            'cosh': torch.tensor(cosh, dtype=torch.float32),
            'sinh': torch.tensor(sinh, dtype=torch.float32),
            'option': torch.tensor(item['option'], dtype=torch.long),
            'last_option': last_option,
        }
        return data

def collate_fn(batch):

    collated = {
        'ego_current_state': torch.stack([item['ego_current_state'] for item in batch]),
        'neighbor_agents_past': torch.stack([item['neighbor_agents_past'] for item in batch]),
        'static_objects': torch.stack([item['static_objects'] for item in batch]),
        'reference_line': torch.stack([item['reference_line'] for item in batch]),
        'lanes': torch.stack([item['lanes'] for item in batch]),
        'route_lanes': torch.stack([item['route_lanes'] for item in batch]),
        'reward': torch.stack([item['reward'] for item in batch]),
        'action_x': torch.stack([item['action_x'] for item in batch]),
        'action_y': torch.stack([item['action_y'] for item in batch]),
        'cosh': torch.stack([item['cosh'] for item in batch]),
        'sinh': torch.stack([item['sinh'] for item in batch]),
        'option': torch.stack([item['option'] for item in batch]),
        'last_option': torch.stack([item['last_option'] for item in batch]),
    }

    return collated