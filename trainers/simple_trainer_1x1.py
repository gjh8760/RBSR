# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import OrderedDict
from trainers.base_trainer import BaseTrainer
from admin.stats import AverageMeter, StatValue
from admin.tensorboard import TensorboardWriter
from admin import multigpu
import torch
import time
import tqdm
import numpy as np

from dataset.synthetic_burst_val_set import SyntheticBurstVal
from models.loss.image_quality_v2 import PSNR, SSIM, LPIPS

class SimpleTrainer_1x1(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        # if self.epoch <= 100:
        #     for group in self.optimizer.param_groups:
        #         group['lr'] = 5e-5
        # if self.epoch <= 300 and self.epoch>100:
        #     for group in self.optimizer.param_groups:
        #         group['lr'] = 2e-5
        # if self.epoch <= 400 and self.epoch>300:
        #     for group in self.optimizer.param_groups:
        #         group['lr'] = 4e-6

        for i, data in enumerate(loader, 1):
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # forward pass
            loss, stats = self.actor(data)


            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not torch.isnan(loss):   # 如果loss是nan，就跳过此次迭代
                    loss.backward()
                self.optimizer.step()

            # update statistics
            # batch_size = data['train_images'].shape[loader.stack_dim]
            batch_size = self.settings.batch_size
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
    
    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""
        """Also remains a best checkpoint."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)

        # Save best checkpoint.
        # psnr = self.stats['train']['Stat/psnr'].history[-1]
        # if psnr > self.best_psnr:
        #     if self.best_epoch != 0:
        #         os.remove('{}/{}_best_ep{:04d}.pth.tar'.format(directory, net_type, self.best_epoch))
        #     self.best_epoch = self.epoch
        #     self.best_psnr = psnr
        #     state['best_epoch'] = self.best_epoch
        #     state['best_psnr'] = self.best_psnr
        #     tmp_file_path = '{}/{}_best_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        #     torch.save(state, tmp_file_path)
        #     file_path = '{}/{}_best_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
        #     os.rename(tmp_file_path, file_path)

        # Save best checkpoint.
        dataset = SyntheticBurstVal()
        metrics = ('psnr',)
        boundary_ignore = 40
        metrics_all = {}
        scores = {}
        for m in metrics:
            if m == 'psnr':
                loss_fn = PSNR(boundary_ignore=boundary_ignore)
            elif m == 'ssim':
                loss_fn = SSIM(boundary_ignore=boundary_ignore, use_for_loss=False)
            elif m == 'lpips':
                loss_fn = LPIPS(boundary_ignore=boundary_ignore)
                loss_fn.to('cuda')
            else:
                raise Exception
            metrics_all[m] = loss_fn
            scores[m] = []
        
        scores_all = {}
        scores = {k: [] for k, v in scores.items()}
        for idx in tqdm.tqdm(range(len(dataset))):
            burst, gt, meta_info = dataset[idx]
            burst_name = meta_info['burst_name']

            burst = burst.to('cuda').unsqueeze(0)
            gt = gt.to('cuda')

            with torch.no_grad():
                burst = burst[:, :14, ...]
                net_pred, _, _ = net(burst)
                net_pred = net_pred[0]
            
            # Perform quantization to be consistent with evaluating on saved images
            net_pred_int = (net_pred.clamp(0.0, 1.0) * 2 ** 14).short()
            net_pred = net_pred_int.float() / (2 ** 14)

            for m, m_fn in metrics_all.items():
                metric_value = m_fn(net_pred, gt.unsqueeze(0)).cpu().item()
                scores[m].append(metric_value)
        psnr_mean = np.mean(scores['psnr'])

        if psnr_mean > self.best_psnr:
            print(f'PSNR: {psnr_mean}(best)')
            if self.best_epoch != 0:
                os.remove('{}/{}_best_ep{:04d}.pth.tar'.format(directory, net_type, self.best_epoch))
            self.best_epoch = self.epoch
            self.best_psnr = psnr_mean
            state['best_epoch'] = self.best_epoch
            state['best_psnr'] = self.best_psnr
            tmp_file_path = '{}/{}_best_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
            torch.save(state, tmp_file_path)
            file_path = '{}/{}_best_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
            os.rename(tmp_file_path, file_path)
        else:
            print(f'PSNR: {psnr_mean}')