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

class ChunkedDataset_single(Dataset):
    def __init__(self, file_pattern, total_files):
        """
        file_pattern: 文件路径模式，例如 /path/to/data_{}.pkl
        total_files: 总文件数
        """
        self.file_pattern = file_pattern
        self.total_files = total_files

    def __len__(self):
        return self.total_files

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        file_path = self.file_pattern.format(idx)
        item = opendata(file_path)

        return {
            "ego_current_state": torch.tensor(item['ego_current_state'], dtype=torch.float32),
            "neighbor_agents_past": torch.tensor(item['neighbor_agents_past'], dtype=torch.float32),
            "static_objects": torch.tensor(item['static_objects'], dtype=torch.float32),
            'reference_line': torch.tensor(item['reference_line'], dtype=torch.float32).view(-1),
            'lanes': torch.tensor(item['lanes'], dtype=torch.float32),
            'route_lanes': torch.tensor(item['route_lanes'], dtype=torch.float32),
            'reward': torch.tensor(item['reward'], dtype=torch.float32),
            'option_reward': torch.tensor(item['option_reward'], dtype=torch.float32),
            'action_x': torch.tensor(item['action_x'], dtype=torch.float32),
            'action_y': torch.tensor(item['action_y'], dtype=torch.float32),
            'option': torch.tensor(item['option'], dtype=torch.long),
        }

def collate_fn(batch):
    return {
        'ego_current_state': torch.stack([item['ego_current_state'] for item in batch]),
        'neighbor_agents_past': torch.stack([item['neighbor_agents_past'] for item in batch]),
        'static_objects': torch.stack([item['static_objects'] for item in batch]),
        'reference_line': torch.stack([item['reference_line'] for item in batch]),
        'lanes': torch.stack([item['lanes'] for item in batch]),
        'route_lanes': torch.stack([item['route_lanes'] for item in batch]),
        'reward': torch.stack([item['reward'] for item in batch]),
        'option_reward': torch.stack([item['option_reward'] for item in batch]),
        'action_x': torch.stack([item['action_x'] for item in batch]),
        'action_y': torch.stack([item['action_y'] for item in batch]),
        'option': torch.stack([item['option'] for item in batch])
    }