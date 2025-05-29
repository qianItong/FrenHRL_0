import os
import json
import numpy as np
import torch
import random
import time
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from HPs.my_args import get_args
from BC.BC_trainer import Diffusion_Trainer, Actor_Trainer, Actor_Option_Trainer, end2end_Trainer
from RL.RL_trainer import RL_Trainer
from DataLoader.DataLoader_BC import ChunkedDataset_BC
from DataLoader.DataLoader_RL import ChunkedDataset_RL
import multiprocessing

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "4054"
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            init_process_group(backend="nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            break
        except RuntimeError as e:
            if "Cannot assign requested address" in str(e) and attempt < max_retries - 1:
                time.sleep(1)  # 等待1秒后重试
                continue
            raise e

def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)

    my_args = get_args()
    pl.seed_everything(my_args.random_seed)

    dataset = ChunkedDataset_RL(
        data_dir='/root/autodl-tmp/nuplan/processed_datas/',
        data_list='/root/codes/Hifren/file_names_random1M.json'
    )
    # dataset = ChunkedDataset_BC(
    #     data_dir='/root/autodl-tmp/nuplan/processed_datas/',
    #     data_list='/root/codes/nuplan_OC/file_names_random1M.json'
    # )
    train_size = int(len(dataset) * 0.8)
    print(f"train_size: {train_size}")  
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False,sampler=DistributedSampler(train_dataset),pin_memory=True,num_workers=6,persistent_workers=True)#
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, sampler=DistributedSampler(val_dataset),pin_memory=True,num_workers=6,persistent_workers=True)#
    noise_lib_X = json.load(open('noise_lib/noise_lib_X.json', 'r'))
    noise_lib_Y = json.load(open('noise_lib/noise_lib_Y.json', 'r'))
    noise_lib_X = np.array(noise_lib_X)
    noise_lib_Y = np.array(noise_lib_Y)
    now_time = time.strftime("%m-%d-%H-%M", time.localtime())

    # trainer = Diffusion_Trainer(
    #     args=my_args,
    #     gpu_id=rank,
    #     train_data=train_loader,
    #     val_data=val_loader,
    #     save_path='/root/autodl-tmp/save_model/diffusion_bc_' + now_time,
    #     load_path='/root/autodl-tmp/save_model/diffusion_bc_05-07-21-55',
    #     load_epoch=59,
    # )
    # trainer = Actor_Trainer(
    #     args=my_args,
    #     gpu_id=rank,
    #     train_data=train_loader,
    #     val_data=val_loader,
    #     noise_lib_X=noise_lib_X,
    #     noise_lib_Y=noise_lib_Y,
    #     save_path='/root/autodl-tmp/save_model/Actor_bc_' + now_time,
    #     load_diffusion_path=None,
    #     load_diffusion_epoch=None,
    #     load_actor_path=None,
    #     load_actor_epoch=None,
    # )

    # trainer = Actor_Option_Trainer(
    #     args=my_args,
    #     gpu_id=rank,
    #     train_data=train_loader,
    #     val_data=val_loader,
    #     save_path='/root/autodl-tmp/save_model/option_' + now_time,
    #     load_path=None,
    #     load_epoch=None,
    # )

    # trainer = end2end_Trainer(
    #     args=my_args,
    #     gpu_id=rank,
    #     train_data=train_loader,
    #     val_data=val_loader,
    #     noise_lib_X=noise_lib_X,
    #     noise_lib_Y=noise_lib_Y,
    #     save_path='/root/autodl-tmp/save_model/end_val_' + now_time,
    #     load_path=None,
    #     load_epoch=None,
    # )
    # trainer.run(
    #     total_epoch=my_args.total_epochs,
    #     save_every=my_args.save_every,
    #     val_every=my_args.val_every,
    # )

    trainer = RL_Trainer(
        args=my_args,
        gpu_id=rank,
        train_data=train_loader,
        valid_data=val_loader,
        noise_lib_X=noise_lib_X,
        noise_lib_Y=noise_lib_Y,
        checkpoint_path=None,
        bc_checkpoint_path=None,
        save_path='/root/autodl-tmp/save_model/RL' + now_time,
        begin_epoch=0,
    )
    trainer.train(
        total_epoch=my_args.total_epochs,
        save_every=my_args.save_every,
        val_every=my_args.val_every,
        critic_epochs=my_args.critic_epochs,
    )
    destroy_process_group()
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)