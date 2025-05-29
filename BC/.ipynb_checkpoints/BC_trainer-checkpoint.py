'''
对扩散模型进行预训练,模仿学习
'''
import numpy as np
import torch
import torch.distributed as dist
from Model.feature_model import Feature
from Model.actor import ActorAction
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import tqdm


class Diffusion_Trainer():
    def __init__(
        self,
        args,
        gpu_id,
        noise_lib_X: np.ndarray,
        noise_lib_Y: np.ndarray,
        train_data: DataLoader,
        val_data: DataLoader,
        save_path: str,
        load_path: str = None,
        load_epoch: int = None,
    ):
        self.args = args
        self.gpu_id = gpu_id
        self.world_size = dist.get_world_size()
        self.feature_model = Feature(args, gpu_id, load_path, load_epoch)
        self.actor = ActorAction(args, noise_lib_X, noise_lib_Y).to(gpu_id)
        if load_path is not None and load_epoch is not None:
            self.actor.load_state_dict(torch.load(f'{load_path}/actor_{load_epoch}.pth'))

        self.actor = DDP(self.actor, device_ids=[gpu_id])
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = torch.optim.Adam(
            list(self.actor.module.parameters()) + self.feature_model.parameters(),
            lr=args.diffusion_bc_lr,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.diffusion_bc_step_size, gamma=args.diffusion_bc_gamma
        )
        self.save_path = save_path
        if gpu_id == 0:
            self.writer = SummaryWriter(f'/root/tf-logs')
        else:
            self.writer = None
        self.epoch = 0


    def val(self):
        self.feature_model.eval()
        self.actor.eval()
        average_loss = 0.0

        with torch.no_grad():
            for data in self.val_data:
                feature = self.feature_model.get_feature(data)
                a0_X = torch.tensor(data['action_x']).to(self.gpu_id)
                a0_Y = torch.tensor(data['action_y']).to(self.gpu_id)
                option = torch.tensor(data['option']).to(self.gpu_id)
                loss = self.actor.module.diffusion_loss(feature, option, a0_X, a0_Y)
                average_loss += loss.item()
        
            average_loss /= len(self.val_data)
            average_loss = torch.tensor(average_loss, device=self.gpu_id)
            dist.all_reduce(average_loss, op=dist.ReduceOp.SUM)
            global_average_loss = average_loss.item() / self.world_size


        if self.gpu_id == 0:
            self.writer.add_scalar('val_loss', global_average_loss, self.epoch)
            print(f'epoch: {self.epoch}, val_loss: {global_average_loss}')

    def run(self, total_epoch, val_every, save_every, train_part='diffusion'):
        for i in range(total_epoch):
            self.feature_model.train()
            self.actor.train()
            self.train_data.sampler.set_epoch(i)
            average_loss = 0.0
            if self.gpu_id == 0:
                train_bar = tqdm.tqdm(self.train_data, desc='Training')
            else:
                train_bar = self.train_data
            for data in train_bar:
                feature = self.feature_model.get_feature(data)
                a0_X =data['action_x'].to(self.gpu_id)
                a0_Y =data['action_y'].to(self.gpu_id)
                option =data['option'].to(self.gpu_id)
                if train_part == 'diffusion':
                    loss = self.actor.module.diffusion_loss(feature, option, a0_X, a0_Y)
                elif train_part == 'actor':
                    loss = self.actor.module.actor_loss(feature, option, a0_X, a0_Y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                average_loss += loss.item()
                if self.gpu_id == 0:
                    train_bar.set_postfix(loss=loss.item())

            average_loss /= len(self.train_data)

            average_loss_tensor = torch.tensor(average_loss, device=self.gpu_id, dtype=torch.float32)
            dist.all_reduce(average_loss_tensor, op=dist.ReduceOp.SUM)
            global_average_loss = average_loss_tensor.item() / self.world_size

            if self.gpu_id == 0:
                self.writer.add_scalar('train_loss', global_average_loss, self.epoch)
                print(f'epoch: {self.epoch}, loss: {global_average_loss}')
                
            if (i+1) % val_every == 0:
                self.val()
            
            if (i+1) % save_every == 0 and self.gpu_id == 0:
                self.save_checkpoint()

            self.epoch += 1
        # self.save_checkpoint()

    def save_checkpoint(self):
        if self.gpu_id == 0:
            torch.save(self.actor.module.state_dict(), f'{self.save_path}/actor_{self.epoch}.pth')
            self.feature_model.save_model(self.save_path, self.epoch)

    def load_checkpoint(self, save_path, epoch):
        self.actor.module.load_state_dict(torch.load(f'{save_path}/actor_{epoch}.pth'))
        self.feature_model.load_model(save_path, epoch)

        