from model import *
from dataset import MMWHS
from configs import logx
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.utils import cal_metric
from torchvision.utils import make_grid
import torch.nn.functional as F
import numpy as np
import torch
import json
import itertools
import time

class Trainer:
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        self.device = opt['device']
        self.word_size = opt['word_size']
        self.rank = opt['rank']
        self.build_datasets()
        self.build_model()
        self.best_result = 0.0
        self.start_epoch = 0

    def build_datasets(self):
        start = time.time()
        self.Atrain = MMWHS(self.cfg.DATA.SOURCE, 'train', dataroot=self.cfg.ROOT + 'data/')
        self.Btrain = MMWHS(self.cfg.DATA.TARGET, 'train', dataroot=self.cfg.ROOT + 'data/')   
        self.Aval = MMWHS(self.cfg.DATA.SOURCE, 'val', dataroot=self.cfg.ROOT + 'data/')  
        self.Bval = MMWHS(self.cfg.DATA.TARGET, 'val', dataroot=self.cfg.ROOT + 'data/')
        Asampler = DistributedSampler(self.Atrain, num_replicas=self.word_size, rank=self.rank)
        Bsampler = DistributedSampler(self.Btrain, num_replicas=self.word_size, rank=self.rank)
        self.Atrain_loader = DataLoader(self.Atrain, shuffle=False, batch_size=self.cfg.TRAIN.BATCH_SIZE, drop_last=True, num_workers=self.cfg.WORKERS, sampler=Asampler)
        self.Btrain_loader = DataLoader(self.Btrain, shuffle=False, batch_size=self.cfg.TRAIN.BATCH_SIZE, drop_last=True, num_workers=self.cfg.WORKERS, sampler=Bsampler)
        self.Aval_loader = DataLoader(self.Aval, shuffle=False, batch_size=self.cfg.TRAIN.BATCH_SIZE, drop_last=False, num_workers=self.cfg.WORKERS)
        self.Bval_loader = DataLoader(self.Bval, shuffle=False, batch_size=self.cfg.TRAIN.BATCH_SIZE, drop_last=False, num_workers=self.cfg.WORKERS)
        logx.msg(f'--> Building dataloader elapsed: {time.time()-start} seconds.')
        logx.msg(f'Length of Atrain: {len(self.Atrain)}')
        logx.msg(f'Length of Btrain: {len(self.Btrain)}')
        logx.msg(f'Length of Atrain_loader: {len(self.Atrain_loader)}')
        logx.msg(f'Length of Btrain_loader: {len(self.Btrain_loader)}')

    def build_model(self):
        logx.msg(f'--> Building models: {self.cfg.MODEL.NAME}')
        self.model = eval(self.cfg.MODEL.NAME)(self.cfg, self.opt)

    def plot_images(self, visuals, step):
        for key, t in visuals.items():
            visuals[key] = make_grid(t)
        for label, image_grid in visuals.items():
            logx.add_image(label, image_grid, step)

    def valid_model(self, epoch):
        self.model.set_mode('eval')
        dice_list = []
        with torch.no_grad():
            for i, (inputA, inputB) in enumerate(zip(self.Aval_loader, self.Bval_loader)):
                img_A = inputA[0].to(self.device)
                seg_A = inputA[1].to(self.device)
                img_B = inputB[0].to(self.device)
                seg_B = inputB[1].to(self.device)
                if self.cfg.MODEL.NAME == 'Model3' or self.cfg.MODEL.NAME == 'Model4':
                    style_A, _, _ = self.model.E_style_A(img_A)
                    contents_B, _ = self.model.E_content_B(img_B)
                    if self.cfg.MODEL.NAME =='Model4':
                        contents_B = torch.round(F.softmax(contents_B, dim=1))
                    fake_A = self.model.G_A(contents_B, style_A)
                    _, pred_seg_B = self.model.E_content_A(fake_A)
                elif self.cfg.MODEL.NAME == 'Model2':
                    _, contents_B = self.model.E_content(img_A, img_B)
                    pred_seg_B = self.model.segmenter(contents_B)
                elif self.cfg.MODEL.NAME == 'Model':
                    style_A, _, _ = self.model.E_style_A(img_A)
                    contents_B, _ = self.model.E_content_B(img_B)
                    fake_A = self.model.G_A(style_A, contents_B)
                    _, pred_seg_B = self.model.E_content_A(fake_A)
                pred_seg_B = torch.argmax(pred_seg_B.detach(), dim=1)
                if i == 50:
                    visual_dict = self.model.visual_images(img_A, seg_A, img_B, seg_B)
                    self.plot_images(visual_dict, self.current_iter)
                result = cal_metric(seg_B.cpu().numpy(), pred_seg_B.cpu().numpy())
                dice_list.append((result["lv"] + result["myo"] + result["la"] + result['aa']) / 4.)
        monitor = np.mean(np.array(dice_list))
        if monitor > self.best_result:
            self.best_result = monitor
            self.model.save(prefix='best')
            logx.msg(f'epoch:{epoch} best validation dice: {self.best_result}')

    def plot_current_losses(self, step):
        losses = self.model.get_current_losses().copy()
        for tag, value in losses.items():
            value = value.mean().float()
            logx.add_scalar(tag, value, step)
        for k, v in losses.items():
            losses[k] = v.mean().float().item()
        logx.msg(f'--> printing from {step}')
        logx.msg(json.dumps(losses, indent=4))

    def train(self):
        self.current_iter = self.start_epoch * max(len(self.Atrain_loader), len(self.Btrain_loader))* self.cfg.TRAIN.BATCH_SIZE
        for epoch in range(self.start_epoch, self.cfg.TRAIN.NEPOCHS):
            self.Atrain_loader.sampler.set_epoch(epoch)
            self.Btrain_loader.sampler.set_epoch(epoch)
            epoch_start_time = time.time()
            self.model.set_mode('train')
            for inputA, inputB in zip(self.Atrain_loader, itertools.cycle(self.Btrain_loader)):
                self.current_iter += self.cfg.TRAIN.BATCH_SIZE * 2
                self.model.set_input(inputA, inputB)
                self.model.step()
                if self.current_iter % self.cfg.TRAIN.DISPLAY_FREQ == 0:
                    self.plot_current_losses(self.current_iter)
            logx.msg(f'Epoch: {epoch} elapsed time: {time.time()-epoch_start_time}')
            if epoch % 2 == 0:
                self.valid_model(epoch)
                self.model.update_learning_rate(1e-3*(0.9**(epoch/2)))
            if (epoch + 1) % self.cfg.TRAIN.SAVE_EPOCH_FREQ == 0:
                self.model.save()


