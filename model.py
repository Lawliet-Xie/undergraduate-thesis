import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture import *
from torch.nn.parallel import DistributedDataParallel
from utils.loss import *
import kornia
from configs import logx
import os
from utils.utils import color_seg, normalize_img
from collections import OrderedDict
import itertools


class Model:
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.device = opt['device']
        self.local_rank = opt['local_rank']
        self.rank = opt['rank']
        self.model_names = ['E_content_A', 'E_content_B', 'E_style_A', 'E_style_B', 'G_A', 'G_B', 'D_img_A', 'D_img_B', 'D_seg']
        
        self.E_content_A = EncoderContent(cfg.MODEL.IN_CH, cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CONTENTS)
        self.E_content_B = EncoderContent(cfg.MODEL.IN_CH, cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CONTENTS)
        self.E_style_A = EncoderStyle(cfg.MODEL.IN_CH, cfg.MODEL.Z_DIM)
        self.E_style_B = EncoderStyle(cfg.MODEL.IN_CH, cfg.MODEL.Z_DIM)
        self.G_A = SPADEGenerator(cfg.MODEL.Z_DIM, cfg.MODEL.NUM_CONTENTS)
        self.G_B = SPADEGenerator(cfg.MODEL.Z_DIM, cfg.MODEL.NUM_CONTENTS)
        self.D_img_A = Discriminator(cfg.MODEL.IN_CH)
        self.D_img_B = Discriminator(cfg.MODEL.IN_CH)
        self.D_seg = Discriminator(cfg.MODEL.NUM_CLASSES)
        
        self.E_content_A = self.load_network(f'E_content_{self.cfg.DATA.SOURCE}', cfg.TRAIN.PRETRAINED_CKPT)
        #self.E_content_B = self.load_network('E_content_A', cfg.TRAIN.PRETRAINED_CKPT)
        self.E_content_A = self.init_net(self.E_content_A)
        self.E_content_B= self.init_net(self.E_content_B)
        self.E_style_A = self.init_net(self.E_style_A)
        self.E_style_B = self.init_net(self.E_style_B)
        self.G_A = self.init_net(self.G_A)
        self.G_B = self.init_net(self.G_B)
        self.D_img_A = self.init_net(self.D_img_A)
        self.D_img_B = self.init_net(self.D_img_B)
        self.D_seg = self.init_net(self.D_seg)

        # freeze E_content_A
        for param in self.E_content_A.parameters():
            param.requires_grad = False

        self.criterian_gan = GANLoss(mode=self.cfg.LOSS.GAN_TYPE)
        self.criterian_l1 = nn.L1Loss()
        self.criterian_ce = CrossEntropyLossWeighted()
        self.criterian_dice = kornia.losses.DiceLoss()
        self.criterian_ortho = OrthonormalityLoss(cfg.MODEL.NUM_CONTENTS)
        self.criterian_kl = KLDLoss()

        self.optimizer_G, self.optimizer_D = self.build_optimizers()

    def init_net(self, net):
        net.to(self.device)
        net = DistributedDataParallel(net, device_ids=[self.local_rank])#, find_unused_parameters=True)
        return net

    def load_network(self, name, ckpt_dir, prefix='latest'):
        save_filename = f'{prefix}_{name}.pth'
        save_path = os.path.join(ckpt_dir, save_filename)
        weights = torch.load(save_path, map_location=str(self.device))
        net = self.E_content_A
        net.load_state_dict(weights['unet'])
        logx.msg(f'Loading {name} from {save_path}.')
        return net

    def build_optimizers(self):
        G_params = itertools.chain(
                        self.E_style_A.parameters(), self.E_style_B.parameters(),
                        self.G_A.parameters(), self.G_B.parameters())
        optimizer_G = torch.optim.Adam(G_params, lr=self.cfg.OPTIM.GEN_LR, betas=(self.cfg.OPTIM.BETA1, 0.999), weight_decay=self.cfg.OPTIM.WEIGHT_DECAY)
        optimizer_G.add_param_group({
            'params': self.E_content_B.parameters(),
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'weight_decay':0.0002
        })
        D_params = itertools.chain(self.D_img_A.parameters(), self.D_img_B.parameters(), self.D_seg.parameters())
        optimizer_D = torch.optim.Adam(D_params, lr=self.cfg.OPTIM.DIS_LR, betas=(self.cfg.OPTIM.BETA1, 0.999), weight_decay=self.cfg.OPTIM.WEIGHT_DECAY)

        return optimizer_G, optimizer_D

    def update_learning_rate(self, lr):
        self.optimizer_G.param_groups[1]['lr'] = lr


    def set_mode(self, phase='train'):
        assert phase == 'train' or phase == 'eval'
        for name in self.model_names:
            net = getattr(self, name)
            if phase == 'train':
                net.train()
            else:
                net.eval()

    def set_input(self, input_A, input_B):
        self.real_A = input_A[0].to(self.device)
        self.real_seg_A = input_A[1].to(self.device)
        self.real_B = input_B[0].to(self.device)
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        ## forward
        # first cross
        self.contents_A, self.logits_A = self.E_content_A(self.real_A)
        self.style_A, self.mu_A, self.logvar_A = self.E_style_A(self.real_A)
        self.contents_B, self.logits_B = self.E_content_B(self.real_B)
        self.style_B, self.mu_B, self.logvar_B = self.E_style_B(self.real_B)
        self.fake_A = self.G_A(self.style_A, self.contents_B)
        self.fake_B = self.G_B(self.style_B, self.contents_A)
        self.id_A = self.G_A(self.style_A, self.contents_A)
        self.id_B = self.G_B(self.style_B, self.contents_B)
        # second cross
        self.contents_fake_A, self.logits_fake_A = self.E_content_A(self.fake_A)
        self.style_fake_A, _, _ = self.E_style_A(self.fake_A)
        self.contents_fake_B, self.logits_fake_B = self.E_content_B(self.fake_B)
        self.style_fake_B, _, _ = self.E_style_B(self.fake_B)
        self.rec_A = self.G_A(self.style_fake_A, self.contents_fake_B)
        self.rec_B = self.G_B(self.style_fake_B, self.contents_fake_A)

    def update_G(self):
        self.optimizer_G.zero_grad()
        # losses
        # cycle loss
        loss_cycle = self.criterian_l1(self.rec_A, self.real_A) + self.criterian_l1(self.rec_B, self.real_B)
        # identity loss
        loss_id = self.criterian_l1(self.id_A, self.real_A) + self.criterian_l1(self.id_B, self.real_B)
        # seg loss
        loss_seg = self.criterian_ce(self.logits_fake_B, self.real_seg_A) + self.cfg.LOSS.LAMBDA_DICE * self.criterian_dice(self.logits_fake_B, self.real_seg_A)
        # pseudo loss
        loss_pseudo = self.criterian_ce(self.logits_B, torch.argmax(self.logits_fake_A.detach(), dim=1)) + self.cfg.LOSS.LAMBDA_DICE * self.criterian_dice(self.logits_B, torch.argmax(self.logits_fake_A.detach(), dim=1))
        # loss_pseudo_content
        # orthogonality of contents
        loss_ortho = self.criterian_ortho(self.contents_B)
        # adversarial
        loss_img_adv = self.criterian_gan(self.D_img_A(self.fake_A), True) + self.criterian_gan(self.D_img_B(self.fake_B), True)
        #loss_img_aux_adv = self.criterian_gan(self.D_img_A(self.fake_A), True)
        loss_seg_adv = self.criterian_gan(self.D_seg(F.softmax(self.logits_fake_A, dim=1)), True)
        loss_adv = loss_img_adv + loss_seg_adv #+ loss_img_aux_adv
        # KL loss
        loss_kl = self.criterian_kl(self.mu_A, self.logvar_A) + self.criterian_kl(self.mu_B, self.logvar_B)
        loss_G = self.cfg.LOSS.LAMBDA_CYCLE*loss_cycle + self.cfg.LOSS.LAMBDA_SEG*loss_seg + self.cfg.LOSS.LAMBDA_ADV * loss_adv + \
                self.cfg.LOSS.LAMBDA_ID * loss_id + self.cfg.LOSS.LAMBDA_PSEUDO * loss_pseudo + self.cfg.LOSS.LAMBDA_ORTHO * loss_ortho +\
                self.cfg.LOSS.LAMBDA_KL * loss_kl
        
        loss_G.backward()
        self.optimizer_G.step()
        self.G_losses = {
            'G': loss_G.detach(),
            'cycle': loss_cycle.detach(),
            'seg': loss_seg.detach(),
            'adv': loss_adv.detach(),
            'id': loss_id.detach(),
            'pseudo': loss_pseudo.detach(),
            'ortho': loss_ortho.detach(),
            'kl': loss_kl.detach()
        }
        
    def update_D(self):
        self.optimizer_D.zero_grad()
        loss_D_img = self.criterian_gan(self.D_img_A(self.real_A), True) + self.criterian_gan(self.D_img_A(self.fake_A.detach()), False) + \
            self.criterian_gan(self.D_img_B(self.real_B), True) + self.criterian_gan(self.D_img_B(self.fake_B.detach()), False)
        loss_D_seg = self.criterian_gan(self.D_seg(F.softmax(self.logits_A.detach(), dim=1)), True) + self.criterian_gan(self.D_seg(F.softmax(self.logits_fake_A.detach(), dim=1)), False)
        #loss_D_img_aux = self.criterian_gan(self.D_img_A(self.rec_A.detach()), True) + self.criterian_gan(self.D_img_A(self.fake_A.detach()), False)
        loss_D = self.cfg.LOSS.LAMBDA_DIS*(loss_D_img + loss_D_seg)# + loss_D_img_aux)
        loss_D.backward()
        self.optimizer_D.step()
        self.D_losses = {
            'D': loss_D.detach(),
            'D_img': loss_D_img.detach(),
            'D_seg': loss_D_seg.detach()
        }
        
    def step(self):
        self.forward()
        # update G
        self.set_requires_grad([self.D_img_A, self.D_img_B, self.D_seg], False)
        self.update_G()
        #update D
        self.set_requires_grad([self.D_img_A, self.D_img_B, self.D_seg], True)
        #self.forward()
        self.update_D()

    def get_current_losses(self):
        return {**self.G_losses, **self.D_losses}

    def save(self, prefix='latest'):
        if not (self.rank == 0):
            return
        for name in self.model_names:
            net = getattr(self, name)
            save_dict = {name: net.state_dict()}
            save_filename = f'{prefix}_{name}.pth'
            save_path = os.path.join(self.cfg.TRAIN.OUTPUT_CKPT, save_filename)
            torch.save(save_dict, save_path)

    def visual_images(self, img_A, seg_A, img_B, seg_B):
        # A to B
        contents_A,logits_A = self.E_content_A(img_A)
        style_A, _, _ = self.E_style_A(img_A)
        contents_B, logits_B = self.E_content_B(img_B)
        style_B, _, _ = self.E_style_B(img_B)
        fake_A = self.G_A(style_A, contents_B)
        fake_B = self.G_B(style_B, contents_A)
        # second cross
        contents_fake_A, logits_fake_A = self.E_content_A(fake_A)
        style_fake_A, _, _ = self.E_style_A(fake_A)
        contents_fake_B, logits_fake_B = self.E_content_B(fake_B)
        style_fake_B, _, _ = self.E_style_B(fake_B)
        rec_A = self.G_A(style_fake_A, contents_fake_B)
        rec_B = self.G_B(style_fake_B, contents_fake_A)

        visual_dict = OrderedDict([
            ('real_A', (normalize_img(img_A[:,1:2,:,:])*255).to(torch.uint8)),
            ('fake_B', (normalize_img(fake_B[:,1:2,:,:].detach())*255).to(torch.uint8)),
            ('rec_A', (normalize_img(rec_A[:,1:2,:,:].detach())*255).to(torch.uint8)),
            ('real_B', (normalize_img(img_B[:,1:2,:,:])*255).to(torch.uint8)),
            ('fake_A', (normalize_img(fake_A[:,1:2,:,:].detach())*255).to(torch.uint8)),
            ('rec_B', (normalize_img(rec_B[:,1:2,:,:].detach())*255).to(torch.uint8)),
            ('real_seg_A', color_seg(seg_A.unsqueeze(1))),
            ('pred_seg_A', color_seg(torch.argmax(logits_A.detach(), dim=1).unsqueeze(1))),
            ('pred_seg_fake_B', color_seg(torch.argmax(logits_fake_B.detach(), dim=1).unsqueeze(1))),
            ('real_seg_B', color_seg(seg_B.unsqueeze(1))),
            ('pred_seg_B', color_seg(torch.argmax(logits_B.detach(), dim=1).unsqueeze(1))),
            ('pred_seg_fake_A', color_seg(torch.argmax(logits_fake_A.detach(), dim=1).unsqueeze(1))),
            ('contents_A' , (2*F.softmax(contents_A.detach(), dim=1) -1).view(-1, 1, contents_A.size(2), contents_A.size(3))),
            ('contents_B' , (2*F.softmax(contents_B.detach(), dim=1) -1).view(-1, 1, contents_B.size(2), contents_B.size(3)))
        ])
        return visual_dict


class Model3:
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.device = opt['device']
        self.local_rank = opt['local_rank']
        self.rank = opt['rank']
        self.model_names = ['E_content_A', 'E_content_B', 'E_style_A', 'E_style_B', 'G_A', 'G_B', 'D_content', 'D_img_A', 'D_img_B', 'D_seg']
        
        self.E_content_A = EncoderContent(cfg.MODEL.IN_CH, cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CONTENTS)
        self.E_content_B = EncoderContent(cfg.MODEL.IN_CH, cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CONTENTS)
        self.E_style_A = EncoderStyle(cfg.MODEL.IN_CH, cfg.MODEL.Z_DIM)
        self.E_style_B = EncoderStyle(cfg.MODEL.IN_CH, cfg.MODEL.Z_DIM)
        self.G_A = AdaINDecoder(cfg.MODEL.NUM_CONTENTS)
        self.G_B = AdaINDecoder(cfg.MODEL.NUM_CONTENTS)
        self.D_content = Discriminator(cfg.MODEL.NUM_CONTENTS)
        self.D_img_A = Discriminator(cfg.MODEL.IN_CH)
        self.D_img_B = Discriminator(cfg.MODEL.IN_CH)
        self.D_seg = Discriminator(cfg.MODEL.NUM_CLASSES)
        
        self.E_content_A = self.load_network('E_content_A', cfg.TRAIN.PRETRAINED_CKPT)
        self.E_content_B = self.load_network('E_content_A', cfg.TRAIN.PRETRAINED_CKPT)
        self.E_content_A = self.init_net(self.E_content_A)
        self.E_content_B= self.init_net(self.E_content_B)
        self.E_style_A = self.init_net(self.E_style_A)
        self.E_style_B = self.init_net(self.E_style_B)
        self.G_A = self.init_net(self.G_A)
        self.G_B = self.init_net(self.G_B)
        self.D_content = self.init_net(self.D_content)
        self.D_img_A = self.init_net(self.D_img_A)
        self.D_img_B = self.init_net(self.D_img_B)
        self.D_seg = self.init_net(self.D_seg)

        # freeze E_content_A
        for param in self.E_content_A.parameters():
            param.requires_grad = False

        self.criterian_gan = GANLoss(mode=self.cfg.LOSS.GAN_TYPE)
        self.criterian_l1 = nn.L1Loss()
        self.criterian_ce = CrossEntropyLossWeighted()
        self.criterian_dice = kornia.losses.DiceLoss()
        self.criterian_ortho = OrthonormalityLoss(cfg.MODEL.NUM_CONTENTS)
        self.criterian_kl = KLDLoss()

        self.optimizer_G, self.optimizer_D = self.build_optimizers()

    def init_net(self, net):
        net.to(self.device)
        net = DistributedDataParallel(net, device_ids=[self.local_rank])#, find_unused_parameters=True)
        return net

    def load_network(self, name, ckpt_dir, prefix='latest'):
        save_filename = f'{prefix}_{name}.pth'
        save_path = os.path.join(ckpt_dir, save_filename)
        weights = torch.load(save_path, map_location=str(self.device))
        net = getattr(self, name)
        net.load_state_dict(weights)
        logx.msg(f'Loading {name} from {save_path}.')
        return net

    def build_optimizers(self):
        G_params = itertools.chain(
                        self.E_style_A.parameters(), self.E_style_B.parameters(),
                        self.G_A.parameters(), self.G_B.parameters())
        optimizer_G = torch.optim.Adam(G_params, lr=self.cfg.OPTIM.GEN_LR, betas=(self.cfg.OPTIM.BETA1, 0.999), weight_decay=self.cfg.OPTIM.WEIGHT_DECAY)
        optimizer_G.add_param_group({
            'params': self.E_content_B.parameters(),
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'weight_decay':0.0002
        })
        D_params = itertools.chain(self.D_content.parameters(), self.D_img_A.parameters(), self.D_img_B.parameters(), self.D_seg.parameters())
        optimizer_D = torch.optim.Adam(D_params, lr=self.cfg.OPTIM.DIS_LR, betas=(self.cfg.OPTIM.BETA1, 0.999), weight_decay=self.cfg.OPTIM.WEIGHT_DECAY)

        return optimizer_G, optimizer_D

    def update_learning_rate(self, lr):
        self.optimizer_G.param_groups[1]['lr'] = lr


    def set_mode(self, phase='train'):
        assert phase == 'train' or phase == 'eval'
        for name in self.model_names:
            net = getattr(self, name)
            if phase == 'train':
                net.train()
            else:
                net.eval()

    def set_input(self, input_A, input_B):
        self.real_A = input_A[0].to(self.device)
        self.real_seg_A = input_A[1].to(self.device)
        self.real_B = input_B[0].to(self.device)
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        ## forward
        # first cross
        self.contents_A, self.logits_A = self.E_content_A(self.real_A)
        self.style_A, self.mu_A, self.logvar_A = self.E_style_A(self.real_A)
        self.contents_B, self.logits_B = self.E_content_B(self.real_B)
        self.style_B, self.mu_B, self.logvar_B = self.E_style_B(self.real_B)
        self.fake_A = self.G_A(self.contents_B, self.style_A)
        self.fake_B = self.G_B(self.contents_A, self.style_B)
        self.id_A = self.G_A(self.contents_A, self.style_A)
        self.id_B = self.G_B(self.contents_B, self.style_B)
        # second cross
        self.contents_fake_A, self.logits_fake_A = self.E_content_A(self.fake_A)
        self.style_fake_A, _, _ = self.E_style_A(self.fake_A)
        self.contents_fake_B, self.logits_fake_B = self.E_content_B(self.fake_B)
        self.style_fake_B, _, _ = self.E_style_B(self.fake_B)
        self.rec_A = self.G_A(self.contents_fake_B, self.style_fake_A)
        self.rec_B = self.G_B(self.contents_fake_A, self.style_fake_B)

    def update_G(self):
        self.optimizer_G.zero_grad()
        # losses
        # cycle loss
        loss_cycle = self.criterian_l1(self.rec_A, self.real_A) + self.criterian_l1(self.rec_B, self.real_B)
        # identity loss
        loss_id = self.criterian_l1(self.id_A, self.real_A) + self.criterian_l1(self.id_B, self.real_B)
        # seg loss
        loss_seg = self.criterian_ce(self.logits_fake_B, self.real_seg_A) + self.cfg.LOSS.LAMBDA_DICE * self.criterian_dice(self.logits_fake_B, self.real_seg_A)
        # pseudo loss
        loss_pseudo = self.criterian_l1(self.logits_B, self.logits_fake_A.detach())  + self.criterian_ce(self.logits_B, torch.argmax(self.logits_fake_A.detach(), dim=1)) + self.cfg.LOSS.LAMBDA_DICE * self.criterian_dice(self.logits_B, torch.argmax(self.logits_fake_A.detach(), dim=1))
        # loss_pseudo_content
        loss_pseudo += self.criterian_l1(self.contents_B, self.contents_fake_A.detach()) + self.criterian_l1(self.contents_fake_B, self.contents_A.detach())
        # orthogonality of contents
        loss_ortho = self.criterian_ortho(self.contents_B)
        # adversarial
        loss_content_adv = self.criterian_gan(self.D_content(F.softmax(self.contents_B, dim=1)), True)
        loss_img_adv = self.criterian_gan(self.D_img_A(self.fake_A), True) + self.criterian_gan(self.D_img_B(self.fake_B), True)
        loss_seg_adv = self.criterian_gan(self.D_seg(F.softmax(self.logits_fake_A, dim=1)), True)
        loss_adv = loss_content_adv + loss_img_adv + loss_seg_adv
        # KL loss
        loss_kl = self.criterian_kl(self.mu_A, self.logvar_A) + self.criterian_kl(self.mu_B, self.logvar_B)
        loss_G = self.cfg.LOSS.LAMBDA_CYCLE*loss_cycle + self.cfg.LOSS.LAMBDA_SEG*loss_seg + self.cfg.LOSS.LAMBDA_ADV * loss_adv + \
                self.cfg.LOSS.LAMBDA_ID * loss_id + self.cfg.LOSS.LAMBDA_PSEUDO * loss_pseudo + self.cfg.LOSS.LAMBDA_ORTHO * loss_ortho +\
                self.cfg.LOSS.LAMBDA_KL * loss_kl
        
        loss_G.backward()
        self.optimizer_G.step()
        self.G_losses = {
            'G': loss_G.detach(),
            'cycle': loss_cycle.detach(),
            'seg': loss_seg.detach(),
            'adv': loss_adv.detach(),
            'id': loss_id.detach(),
            'pseudo': loss_pseudo.detach(),
            'ortho': loss_ortho.detach(),
            'kl': loss_kl.detach()
        }
        
    def update_D(self):
        self.optimizer_D.zero_grad()
        loss_D_content = self.criterian_gan(self.D_content(F.softmax(self.contents_A.detach(), dim=1)), True) + self.criterian_gan(self.D_content(F.softmax(self.contents_B.detach(), dim=1)), False)
        loss_D_img = self.criterian_gan(self.D_img_A(self.real_A), True) + self.criterian_gan(self.D_img_A(self.fake_A.detach()), False) + \
            self.criterian_gan(self.D_img_B(self.real_B), True) + self.criterian_gan(self.D_img_B(self.fake_B.detach()), False)
        loss_D_seg = self.criterian_gan(self.D_seg(F.softmax(self.logits_A.detach(), dim=1).detach()), True) + self.criterian_gan(self.D_seg(F.softmax(self.logits_fake_A.detach(), dim=1)), False)
        loss_D = self.cfg.LOSS.LAMBDA_DIS*(loss_D_content + loss_D_img + loss_D_seg)
        loss_D.backward()
        self.optimizer_D.step()
        self.D_losses = {
            'D': loss_D.detach(),
            'D_content': loss_D_content.detach(),
            'D_img': loss_D_img.detach(),
            'D_seg': loss_D_seg.detach()
        }
        
    def step(self):
        self.forward()
        # update G
        self.set_requires_grad([self.D_content, self.D_img_A, self.D_img_B, self.D_seg], False)
        self.update_G()
        #update D
        self.set_requires_grad([self.D_content, self.D_img_A, self.D_img_B, self.D_seg], True)
        self.update_D()

    def get_current_losses(self):
        return {**self.G_losses, **self.D_losses}

    def save(self, prefix='latest'):
        if not (self.rank == 0):
            return
        for name in self.model_names:
            net = getattr(self, name)
            save_dict = {name: net.state_dict()}
            save_filename = f'{prefix}_{name}.pth'
            save_path = os.path.join(self.cfg.TRAIN.OUTPUT_CKPT, save_filename)
            torch.save(save_dict, save_path)

    def visual_images(self, img_A, seg_A, img_B, seg_B):
        # A to B
        contents_A,logits_A = self.E_content_A(img_A)
        style_A, _, _ = self.E_style_A(img_A)
        contents_B, logits_B = self.E_content_B(img_B)
        style_B, _, _ = self.E_style_B(img_B)
        fake_A = self.G_A(contents_B, style_A)
        fake_B = self.G_B(contents_A, style_B)
        # second cross
        contents_fake_A, logits_fake_A = self.E_content_A(fake_A)
        style_fake_A, _, _ = self.E_style_A(fake_A)
        contents_fake_B, logits_fake_B = self.E_content_B(fake_B)
        style_fake_B, _, _ = self.E_style_B(fake_B)
        rec_A = self.G_A(contents_fake_B, style_fake_A)
        rec_B = self.G_B(contents_fake_A, style_fake_B)

        visual_dict = OrderedDict([
            ('real_A', (normalize_img(img_A[:,1:2,:,:])*255).to(torch.uint8)),
            ('fake_B', (normalize_img(fake_B[:,1:2,:,:].detach())*255).to(torch.uint8)),
            ('rec_A', (normalize_img(rec_A[:,1:2,:,:].detach())*255).to(torch.uint8)),
            ('real_B', (normalize_img(img_B[:,1:2,:,:])*255).to(torch.uint8)),
            ('fake_A', (normalize_img(fake_A[:,1:2,:,:].detach())*255).to(torch.uint8)),
            ('rec_B', (normalize_img(rec_B[:,1:2,:,:].detach())*255).to(torch.uint8)),
            ('real_seg_A', color_seg(seg_A.unsqueeze(1))),
            ('pred_seg_A', color_seg(torch.argmax(logits_A.detach(), dim=1).unsqueeze(1))),
            ('pred_seg_fake_B', color_seg(torch.argmax(logits_fake_B.detach(), dim=1).unsqueeze(1))),
            ('real_seg_B', color_seg(seg_B.unsqueeze(1))),
            ('pred_seg_B', color_seg(torch.argmax(logits_B.detach(), dim=1).unsqueeze(1))),
            ('pred_seg_fake_A', color_seg(torch.argmax(logits_fake_A.detach(), dim=1).unsqueeze(1))),
            ('contents_A' , F.softmax(contents_A.detach(), dim=1).view(-1, 1, contents_A.size(2), contents_A.size(3))),
            ('contents_B' , F.softmax(contents_B.detach(), dim=1).view(-1, 1, contents_B.size(2), contents_B.size(3)))
        ])
        return visual_dict     