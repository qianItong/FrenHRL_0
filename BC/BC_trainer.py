'''
对扩散模型进行预训练,模仿学习
'''
import numpy as np
import torch
import torch.distributed as dist
from Model.feature_model import Feature, MLP
from Model.actor import ActorAction, DiffusionModel, Actor, HiFren_planner_TRAIN, ActorOption
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.optim import lr_scheduler
import tqdm
import os
from Model.util import regularize, de_regularize, X_MIN, X_MAX, Y_MIN, Y_MAX

class Diffusion_Trainer():
    def __init__(
        self,
        args,
        gpu_id,
        train_data: DataLoader,
        val_data: DataLoader,
        save_path: str,
        load_path: str = None,
        load_epoch: int = None,
    ):
        self.args = args
        self.gpu_id = gpu_id
        self.world_size = dist.get_world_size()
        self.feature_model = Feature(args, gpu_id).to(gpu_id)
        self.diffusion_model = DiffusionModel(args).to(gpu_id)
        if load_path is not None and load_epoch is not None:
            self.diffusion_model.load_state_dict(torch.load(f'{load_path}/diffusion_{load_epoch}.pth'))
            self.feature_model.load_state_dict(torch.load(f'{load_path}/feature_{load_epoch}.pth'))
        self.diffusion_model = DDP(self.diffusion_model, device_ids=[gpu_id])
        self.feature_model = DDP(self.feature_model, device_ids=[gpu_id], find_unused_parameters=True)
        
        self.train_data = train_data
        self.val_data = val_data

        self.optimizer = torch.optim.Adam(
            list(self.feature_model.parameters())+ 
            list(self.diffusion_model.parameters()),
            lr=args.bc_lr,
        )
        if load_path is not None and load_epoch is not None:
            self.optimizer.load_state_dict(torch.load(f'{load_path}/diffusion_optimizer_{load_epoch}.pth'))
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[20], gamma=0.5)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        if gpu_id == 0:
            self.writer = SummaryWriter(f'/root/tf-logs')
        else:
            self.writer = None
        self.epoch = 0


    def val(self):
        self.feature_model.eval()
        self.diffusion_model.eval()
        average_loss = 0.0

        with torch.no_grad():
            for data in self.val_data:
                feature = self.feature_model(data)
                a0_X = data['action_x'].to(self.gpu_id)
                a0_Y = data['action_y'].to(self.gpu_id)
                
                option = data['option'].to(self.gpu_id)
                loss = self.diffusion_model(feature, option, a0_X, a0_Y)
                average_loss += loss.item()
        
            average_loss /= len(self.val_data)
            average_loss = torch.tensor(average_loss, device=self.gpu_id)
            dist.all_reduce(average_loss, op=dist.ReduceOp.SUM)
            global_average_loss = average_loss.item() / self.world_size


        if self.gpu_id == 0:
            self.writer.add_scalar('val_loss', global_average_loss, self.epoch)
            print(f'epoch: {self.epoch}, val_loss: {global_average_loss}')

    def run(self, total_epoch, val_every, save_every):
        for i in range(total_epoch):
            self.feature_model.train()
            self.diffusion_model.train()
            self.train_data.sampler.set_epoch(i)
            average_loss = 0.0
            if self.gpu_id == 0:
                train_bar = tqdm.tqdm(self.train_data, desc='Training')
            else:
                train_bar = self.train_data
            for data in train_bar:
                feature = self.feature_model(data)
                a0_X =data['action_x'].to(self.gpu_id)
                a0_Y =data['action_y'].to(self.gpu_id)
                option =data['option'].to(self.gpu_id)
                loss = self.diffusion_model(feature, option, a0_X, a0_Y)
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
            self.scheduler.step()

    def save_checkpoint(self):
        if self.gpu_id == 0:
            torch.save(self.diffusion_model.module.state_dict(), f'{self.save_path}/diffusion_{self.epoch}.pth')
            torch.save(self.feature_model.module.state_dict(), f'{self.save_path}/feature_{self.epoch}.pth')
            torch.save(self.optimizer.state_dict(), f'{self.save_path}/diffusion_optimizer_{self.epoch}.pth')
            torch.save(self.diffusion_model, f'{self.save_path}/diffusion_model_{self.epoch}.pth')

class Actor_Trainer():
    def __init__(
        self,
        args,
        gpu_id,
        train_data: DataLoader,
        val_data: DataLoader,
        noise_lib_X: np.ndarray,
        noise_lib_Y: np.ndarray,
        save_path: str,
        load_diffusion_path: str = None,
        load_diffusion_epoch: int = None,
        load_actor_path: str = None,
        load_actor_epoch: int = None,
    ):
        self.args = args
        self.gpu_id = gpu_id
        self.world_size = dist.get_world_size()
        self.feature_model = Feature(args, gpu_id).to(gpu_id)
        self.actor = Actor(
            args=args,
            noise_lib_X=noise_lib_X,
            noise_lib_Y=noise_lib_Y,
        ).to(gpu_id)

        if load_diffusion_path is not None and load_diffusion_epoch is not None:
            self.feature_model.load_state_dict(torch.load(f'{load_diffusion_path}/feature_{load_diffusion_epoch}.pth'))
            self.actor.diffusion_model.load_state_dict(torch.load(f'{load_diffusion_path}/diffusion_{load_diffusion_epoch}.pth'))
            
        if load_actor_path is not None and load_actor_epoch is not None:
            self.feature_model.load_state_dict(torch.load(f'{load_actor_path}/feature_{load_actor_epoch}.pth'))
            actor_pth = torch.load(f'{load_actor_path}/actor_{load_actor_epoch}.pth')
            self.actor.load_state_dict(actor_pth, strict=True)
            
        self.feature_model = DDP(self.feature_model, device_ids=[gpu_id], find_unused_parameters=True)
        self.actor = DDP(self.actor, device_ids=[gpu_id])

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters())+list(self.feature_model.parameters()),
            lr=args.bc_lr,
        )

        if load_actor_path is not None and load_actor_epoch is not None:
            self.optimizer.load_state_dict(torch.load(f'{load_actor_path}/actor_optimizer_{load_actor_epoch}.pth'))
        self.train_data = train_data
        self.val_data = val_data

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[5,20], gamma=0.5)

        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        if gpu_id == 0:
            self.writer = SummaryWriter(f'/root/tf-logs')
        else:
            self.writer = None
        self.epoch = 0


    def val(self):
        self.feature_model.eval()
        self.actor.eval()
        average_loss_xy = 0.0

        with torch.no_grad():
            for data in self.val_data:
                feature = self.feature_model(data)
                a0_X = data['action_x'].to(self.gpu_id)
                a0_Y = data['action_y'].to(self.gpu_id)
                option = data['option'].to(self.gpu_id)
                generate_X, generate_Y = self.actor(feature, option)
                original_X = de_regularize(generate_X, X_MIN, X_MAX)
                original_Y = de_regularize(generate_Y, Y_MIN, Y_MAX)
                loss_xy = F.mse_loss(original_X, a0_X) + F.mse_loss(original_Y, a0_Y)
                # loss_xy = F.mse_loss(original_X[:6], a0_X[:6]) + F.mse_loss(original_Y[:6], a0_Y[:6])
                average_loss_xy += loss_xy.item()
        
            average_loss_xy /= len(self.val_data)
            average_loss_xy = torch.tensor(average_loss_xy, device=self.gpu_id)
            dist.all_reduce(average_loss_xy, op=dist.ReduceOp.SUM)
            global_average_loss_xy = average_loss_xy.item() / self.world_size


        if self.gpu_id == 0:
            self.writer.add_scalar('val_loss_xy', global_average_loss_xy, self.epoch)
            print(f'epoch: {self.epoch}, val_loss: {global_average_loss_xy}')

    def run(self, total_epoch, val_every, save_every):
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
                feature = self.feature_model(data)
                a0_X = data['action_x'].to(self.gpu_id)
                a0_Y = data['action_y'].to(self.gpu_id)
                regularize_X = regularize(a0_X, X_MIN, X_MAX)
                regularize_Y = regularize(a0_Y, Y_MIN, Y_MAX)
                option = data['option'].to(self.gpu_id)
                generate_X, generate_Y = self.actor(feature, option)
                loss = F.mse_loss(generate_X, regularize_X) + F.mse_loss(generate_Y, regularize_Y)
                # loss_1 = F.mse_loss(generate_X[:6], regularize_X[:6]) + F.mse_loss(generate_Y[:6], regularize_Y[:6])
                # loss_2 = F.mse_loss(generate_X[6:], regularize_X[6:]) + F.mse_loss(generate_Y[6:], regularize_Y[6:])
                # K = 2.0  # 权重倍数
                # weight_1 = K * 10 / (6 + K * 10)  # 前6个的权重
                # weight_2 = 6 / (6 + K * 10)       # 后10个的权重
                # loss = (loss_1 * weight_1 / 6) + (loss_2 * weight_2 / 10)
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
            # self.scheduler.step()

    def save_checkpoint(self):
        if self.gpu_id == 0:
            torch.save(self.actor.module.state_dict(), f'{self.save_path}/actor_{self.epoch}.pth')
            torch.save(self.optimizer.state_dict(), f'{self.save_path}/actor_optimizer_{self.epoch}.pth')
            torch.save(self.feature_model.module.state_dict(), f'{self.save_path}/feature_{self.epoch}.pth')
            
class Actor_Option_Trainer():
    def __init__(
        self,
        args,
        gpu_id,
        train_data: DataLoader,
        val_data: DataLoader,
        save_path: str,
        load_path: str = None,
        load_epoch: int = None,
    ):
        self.args = args
        self.gpu_id = gpu_id
        self.world_size = dist.get_world_size()
        self.feature_model = Feature(args, gpu_id).to(gpu_id)
        self.actor_option = ActorOption(args).to(gpu_id)

        if load_path is not None and load_epoch is not None:
            self.feature_model.load_state_dict(torch.load(f'{load_path}/feature_{load_epoch}.pth'))
            # self.actor_option.load_state_dict(torch.load(f'{load_path}/actor_option_{load_epoch}.pth'))
           
        self.actor_option = DDP(self.actor_option, device_ids=[gpu_id])
        self.feature_model = DDP(self.feature_model, device_ids=[gpu_id], find_unused_parameters=True)

        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = torch.optim.Adam(
            self.actor_option.parameters(),
            lr=args.bc_lr,
        )
        # if load_path is not None and load_epoch is not None:
            # self.optimizer.load_state_dict(torch.load(f'{load_path}/actor_option_optimizer_{load_epoch}.pth'))
        self.save_path = save_path
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,30], gamma=0.5)
        os.makedirs(save_path, exist_ok=True)
        if gpu_id == 0:
            self.writer = SummaryWriter(f'/root/tf-logs')
        else:
            self.writer = None
        self.epoch = 0
        self.loss_f = torch.nn.CrossEntropyLoss()
        for param in self.feature_model.parameters():
            param.requires_grad = False
        
    def val(self):
        self.feature_model.eval()
        self.actor_option.eval()
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for data in tqdm.tqdm(self.val_data):
                feature = self.feature_model(data)
                option = data['option'].to(self.gpu_id)
                last_option = data['last_option'].to(self.gpu_id)
                output = self.actor_option(feature, last_option)
                _, predicted = torch.max(output, 1)
                correct += (predicted == option).sum().item()        
                total_samples += len(option)

            print(f'correct: {correct}, total_samples: {total_samples}')
            correct_rate = correct / total_samples
            correct_rate = torch.tensor(correct_rate, device=self.gpu_id)
            dist.all_reduce(correct_rate, op=dist.ReduceOp.SUM)
            global_correct_rate = correct_rate.item() / self.world_size


        if self.gpu_id == 0:
            self.writer.add_scalar('val_loss', global_correct_rate, self.epoch)
            print(f'epoch: {self.epoch}, val_loss: {global_correct_rate}')

    def run(self, total_epoch, val_every, save_every):
        for i in range(total_epoch):
            self.feature_model.eval()
            self.actor_option.train()
            self.train_data.sampler.set_epoch(i)
            average_loss = 0.0
            if self.gpu_id == 0:
                train_bar = tqdm.tqdm(self.train_data, desc='Training')
            else:
                train_bar = self.train_data
            for data in train_bar:
                feature = self.feature_model(data)
                option =data['option'].to(self.gpu_id)
                last_option = data['last_option'].to(self.gpu_id)
                output = self.actor_option(feature, last_option)
                loss = self.loss_f(output, option)

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
            self.scheduler.step()

    def save_checkpoint(self):
        if self.gpu_id == 0:
            torch.save(self.actor_option.module.state_dict(), f'{self.save_path}/actor_option_{self.epoch}.pth')
            torch.save(self.optimizer.state_dict(), f'{self.save_path}/actor_option_optimizer_{self.epoch}.pth')

class end2end_Trainer():
    def __init__(
        self,
        args,
        gpu_id,
        train_data: DataLoader,
        val_data: DataLoader,
        noise_lib_X: np.ndarray,
        noise_lib_Y: np.ndarray,
        save_path: str,
        load_path: str = None,
        load_epoch: int = None,
    ):
        self.args = args
        self.gpu_id = gpu_id
        self.world_size = dist.get_world_size()
        self.planner = HiFren_planner_TRAIN(args, gpu_id,noise_lib_X, noise_lib_Y).to(gpu_id)


        if load_path is not None and load_epoch is not None:
            # load_dict = torch.load(f'{load_path}/planner_{load_epoch}.pth')
            # dict_without_feature = {k: v for k, v in load_dict.items() if 'Feature_model' not in k and 'actoraction' not in k}
            # self.planner.load_state_dict(load_dict)
            self.planner.Feature_model.load_state_dict(torch.load(f'{load_path}/feature_{load_epoch}.pth'))
            self.planner.actor_model.load_state_dict(torch.load(f'{load_path}/actor_{load_epoch}.pth'))
            self.planner.option_model.load_state_dict(torch.load(f'{load_path}/actor_option_{load_epoch}.pth'))

        self.planner = DDP(self.planner, device_ids=[gpu_id], find_unused_parameters=True)

        self.optimizer = torch.optim.Adam(
            self.planner.parameters(),
            lr=args.bc_lr,
        )
        self.train_data = train_data
        self.val_data = val_data
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[5,20], gamma=0.5)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        if gpu_id == 0:
            self.writer = SummaryWriter(f'/root/tf-logs')
        else:
            self.writer = None
        self.epoch = 0
        self.loss_f = torch.nn.CrossEntropyLoss()

    def val(self):
        self.planner.eval()
        average_loss = 0.0
        average_loss_2 = 0.0

        with torch.no_grad():
            for data in self.val_data:
                a0_X = data['action_x'].to(self.gpu_id)
                a0_Y = data['action_y'].to(self.gpu_id)
                generate_X, generate_Y, _ = self.planner(data)
                original_X = de_regularize(generate_X, X_MIN, X_MAX)
                original_Y = de_regularize(generate_Y, Y_MIN, Y_MAX)
                loss = F.mse_loss(original_X, a0_X) + F.mse_loss(original_Y, a0_Y)
                loss_2 = F.mse_loss(original_X[:6], a0_X[:6]) + F.mse_loss(original_Y[:6], a0_Y[:6])
                # loss = loss_1

                average_loss += loss.item()
                average_loss_2 += loss_2.item()
        
            average_loss /= len(self.val_data)
            average_loss_2 /= len(self.val_data)
            average_loss = torch.tensor(average_loss, device=self.gpu_id)
            average_loss_2 = torch.tensor(average_loss_2, device=self.gpu_id)
            dist.all_reduce(average_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(average_loss_2, op=dist.ReduceOp.SUM)
            global_average_loss = average_loss.item() / self.world_size
            global_average_loss_2 = average_loss_2.item() / self.world_size


        if self.gpu_id == 0:
            self.writer.add_scalar('val_loss', global_average_loss, self.epoch)
            self.writer.add_scalar('val_loss_2', global_average_loss_2, self.epoch)
            print(f'epoch: {self.epoch}, val_loss: {global_average_loss}', f'val_loss_2: {global_average_loss_2}')

    def run(self, total_epoch, val_every, save_every):
        for i in range(total_epoch):
            self.planner.train()
            self.train_data.sampler.set_epoch(i)
            average_loss = 0.0
            if self.gpu_id == 0:
                train_bar = tqdm.tqdm(self.train_data, desc='Training')
            else:
                train_bar = self.train_data
            for data in train_bar:
                a0_X = data['action_x'].to(self.gpu_id)
                a0_Y = data['action_y'].to(self.gpu_id)
                regularize_X = regularize(a0_X, X_MIN, X_MAX)
                regularize_Y = regularize(a0_Y, Y_MIN, Y_MAX)
                generate_X, generate_Y, _ = self.planner(data)
                loss = F.mse_loss(generate_X, regularize_X) + F.mse_loss(generate_Y, regularize_Y)
                # loss_1 = F.mse_loss(generate_X[:6], regularize_X[:6]) + F.mse_loss(generate_Y[:6], regularize_Y[:6])
                # loss_2 = F.mse_loss(generate_X[6:], regularize_X[6:]) + F.mse_loss(generate_Y[6:], regularize_Y[6:])
                # loss = loss_1 * 0.8 + loss_2 * 0.2
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
            # self.scheduler.step()

    def save_checkpoint(self):
        if self.gpu_id == 0:
            torch.save(self.planner.module.state_dict(), f'{self.save_path}/planner_{self.epoch}.pth')
            torch.save(self.planner.module.Feature_model.state_dict(), f'{self.save_path}/feature_{self.epoch}.pth')
            torch.save(self.planner.module.actor_model.state_dict(), f'{self.save_path}/actor_{self.epoch}.pth')
            torch.save(self.planner.module.option_model.state_dict(), f'{self.save_path}/actor_option_{self.epoch}.pth')
            
            # torch.save(self.planner, f'{self.save_path}/planner_model_{self.epoch}.pth')
            torch.save(self.optimizer.state_dict(), f'{self.save_path}/actor_optimizer_{self.epoch}.pth')