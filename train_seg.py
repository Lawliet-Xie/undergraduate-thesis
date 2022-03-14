import kornia
from dataset import MMWHS
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
from utils.loss import CrossEntropyLossWeighted, OrthonormalityLoss
from architecture import EncoderContent
from medpy import metric
from utils.utils import *
from collections import namedtuple
import numpy as np
import json
import itertools
import time


class SegmentationTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.unet = EncoderContent(3, 5, opt.n_contents)

        #self.load_pretrained()
        self.unet = self.unet.cuda()
        self.criterian_dice = kornia.losses.DiceLoss()
        self.criterian_ce = CrossEntropyLossWeighted()

        self.criterian_ortho = OrthonormalityLoss(opt.n_contents)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr = opt.lr, betas=(0.9, 0.999), weight_decay=0.0002)
    
    def step(self, x, gt_seg):
        self.optimizer.zero_grad()
        contents, pred_seg = self.unet(x)

        dice = self.criterian_dice(pred_seg, gt_seg)
        cross_entropy = self.criterian_ce(pred_seg, gt_seg)
        loss_ortho = self.criterian_ortho(contents)
        loss = dice + cross_entropy + loss_ortho
        self.losses = {'cross_entropy': cross_entropy.detach(), 'dice': dice.detach(), 'ortho':loss_ortho.detach()}

        loss.backward()
        self.optimizer.step()

    def save(self, epoch, prefix='latest'):
        save_filename = f'{prefix}_E_content_{opt.modality}.pth'
        save_path = os.path.join(opt.output_ckpt, save_filename)
        torch.save({'unet':self.unet.state_dict()}, save_path)
        print(f'--> Saveing the model to {save_path} at epoch {epoch}')

    def load_pretrained(self):
        load_filename = f'latest_E_content_{opt.modality}.pth'
        load_path = os.path.join('./pretrained/', load_filename)
        state_dict = torch.load(load_path, map_location=str(torch.device('cuda')))
        self.unet.load_state_dict(state_dict)


    def load(self, prefix='latest'):
        load_filename = f'{prefix}_E_content_{opt.modality}.pth'
        load_path = os.path.join(opt.output_ckpt, load_filename)
        state_dict = torch.load(load_path, map_location=str(torch.device('cuda')))
        self.unet.load_state_dict(state_dict['unet'])

    def update_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def evaluate(model, modality, prefix):
    model.load(prefix=prefix)
    unet = model.unet
    unet.eval()
    test_folder = f'data/{modality}_test/'
    all_img_paths = sorted([f for f in os.listdir(os.path.join(test_folder, 'images')) if f.endswith('.nii')])
    all_seg_paths = sorted([f for f in os.listdir(os.path.join(test_folder, 'labels')) if f.endswith('.nii')])
    n_samples = len(all_img_paths)
    assert n_samples == len(all_seg_paths)

    evaluation_dice = np.zeros((n_samples, 4))
    evaluation_assd = np.zeros((n_samples, 4))


    for i in range(n_samples):
        img_vol = process_img(load_nii(os.path.join(test_folder, 'images', all_img_paths[i])), modality)
        seg_vol = process_seg(load_nii(os.path.join(test_folder, 'labels', all_seg_paths[i])))
        pred_vol = np.zeros_like(seg_vol)
        for j in range(img_vol.shape[0]):
            input_img = torch.from_numpy(img_vol[[j-1, j, (j+1) % img_vol.shape[0]]].copy()).to(torch.float32)
            input_img = input_img[None, ...]
            input_img = input_img.cuda()
            _, pred_seg = unet(input_img)
            pred_seg = torch.argmax(pred_seg, dim=1)
            pred_vol[j] = pred_seg.cpu().detach().numpy()

        pred_vol = keep_largest_connected_components(pred_vol)
        pred_vol = np.array(pred_vol).astype(np.uint16)

        for j in range(1, 5):
            seg_vol_class = seg_vol == j
            pred_vol_class = pred_vol == j
            dice = metric.binary.dc(pred_vol_class, seg_vol_class)
            if j in pred_vol:
                assd = metric.binary.asd(pred_vol_class, seg_vol_class)
            else:
                assd = np.nan
            evaluation_dice[i, j-1] = dice
            evaluation_assd[i, j-1] = assd

    print(f'=> Evaluating on {modality} set.')
    print(' => Results Dice mean = ', evaluation_dice.mean(axis=0))
    print(' => Results Dice std = ', evaluation_dice.std(axis=0))
    print('=> Overall mean = ', evaluation_dice.mean(axis=1).mean())
    print('=> Overall std  = ', evaluation_dice.mean(axis=1).std())

    print(' => Results ASSD mean = ', evaluation_assd.mean(axis=0))
    print(' => Results ASSD std = ', evaluation_assd.std(axis=0))
    print('=> Overall mean = ', evaluation_assd.mean(axis=1).mean())
    print('=> Overall std  = ', evaluation_assd.mean(axis=1).std())


def main(opt):
    train_set = MMWHS(opt.modality, 'train')
    train_loader = DataLoader(train_set, shuffle=True, batch_size=opt.batch_size, drop_last=True, num_workers=8, pin_memory=True)
    val_set = MMWHS(opt.modality, 'val')
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, drop_last=False, num_workers=4)

    with open(opt.output_log+ f'log_{opt.modality}.txt', 'w') as f:
        f.write(f'The length of training {opt.modality} loader: {len(train_loader)}.\n')
        f.write(f'The length of validation {opt.modality} loader: {len(val_loader)}.\n')

    best_result = 0.0
    trainer = SegmentationTrainer(opt)
    current_iter = 0
    for epoch in range(opt.nepochs):
        if epoch> opt.nepochs//2:
            lr = 2.0*opt.lr*(opt.nepochs + 1.0 - epoch)/(opt.nepochs + 2.0)
            trainer.update_learning_rate(lr)
        trainer.unet.train()
        for x, gt_seg in train_loader:
            current_iter += opt.batch_size
            x = x.cuda()
            gt_seg = gt_seg.cuda()
            trainer.step(x, gt_seg)
            #plot training loss
            if current_iter % 8192 == 0:
                losses = trainer.losses.copy()
                for k, v in losses.items():
                    losses[k] = v.mean().float().item()
                with open(opt.output_log+ f'log_{opt.modality}.txt', 'a') as f:
                    f.write(f'--> printing from {current_iter}\n')
                    f.write(json.dumps(losses, indent=4) + '\n')

        #validation
        trainer.unet.eval()
        dice_list = []
        with torch.no_grad():
            for val_x, val_gt_seg in val_loader:
                val_x = val_x.cuda()
                val_gt_seg = val_gt_seg.cuda()
                _, pred_seg = trainer.unet(val_x)
                pred_seg = torch.argmax(pred_seg.detach(), dim=1)
                result = cal_metric(val_gt_seg.cpu().numpy(), pred_seg.cpu().numpy())
                dice_list.append((result["lv"] + result["myo"] + result["la"] + result['aa']) / 4.)
        monitor = np.mean(np.array(dice_list))
        if monitor > best_result:
            with open(opt.output_log+ f'log_{opt.modality}.txt', 'a') as f:
                f.write(f'epoch:{epoch}, best result: {monitor}.\n')
            best_result = monitor
            trainer.save(epoch, prefix='best')
        if epoch == opt.nepochs - 1:
            trainer.save(epoch, prefix='latest')
    
    evaluate(trainer, 'ct', 'best')
    
if __name__ == '__main__':
    os.chdir('')
    time_str = time.strftime('%m-%d-%H-%M')
    options = {
    'modality': 'mr',
    'n_contents': 8,
    'batch_size': 32,
    'nepochs': 100, 
    'n_classes': 5,
    'lr': 4e-4,
    'output_ckpt':f'./outputs/seg/{time_str}/ckpt/',
    'output_log':f'./outputs/seg/{time_str}/log/'
    }
    opt = namedtuple("Option", options.keys())(*options.values())
    os.makedirs(opt.output_ckpt, exist_ok=True)
    os.makedirs(opt.output_log, exist_ok=True)
    main(opt)








