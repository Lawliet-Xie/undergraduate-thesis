from matplotlib.pyplot import plot
import torch
import os
from utils.utils import *
from architecture import EncoderContent, AdaINDecoder, SPADEGenerator, EncoderStyle
from collections import namedtuple, OrderedDict
from nnunet import Generic_UNet
import torch.nn as nn
import argparse

def load_network(net, name, ckpt_dir, prefix='latest'):
    save_filename = f'{prefix}_{name}.pth'
    save_path = os.path.join(ckpt_dir, save_filename)
    weights = torch.load(save_path, map_location='cuda')
    new_state_dict = OrderedDict()
    if 'seg' in name or 'fake' in name or '03-13' in ckpt_dir:
        net.load_state_dict(weights['unet'])
    else:
        for k,v in weights[name].items():
            name = k[7:]
            new_state_dict[name] = v 
        net.load_state_dict(new_state_dict)
    return net


def evaluate(cfg, plot=False):
    print(f'--> Evaluating model: from {cfg.source} to {cfg.target}.')
    E_content_A = EncoderContent(3, 5, 8)
    E_content_B = EncoderContent(3, 5, 8)
    E_style_A = EncoderStyle(3)
    G_A = SPADEGenerator()
    if cfg.unet:
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        unet = Generic_UNet(3, 64, 5, 4,
                            num_conv_per_stage=2,
                            norm_op=nn.InstanceNorm2d, 
                            norm_op_kwargs=norm_op_kwargs,
                            dropout_op_kwargs=dropout_op_kwargs,
                            nonlin_kwargs=net_nonlin_kwargs,
                            final_nonlin=lambda x: x,
                            convolutional_pooling=True, 
                            convolutional_upsampling=True,
                            deep_supervision=False)
        unet = load_network(unet, f'{cfg.source}_seg_model',cfg.root + 'pretrained/', 'best')
        unet.cuda()                     
    E_content_A = load_network(E_content_A, 'E_content_A', cfg.log_dir + 'ckpt/', cfg.prefix)
    #E_content_A = load_network(E_content_A, 'E_content_mr_fake', cfg.E_content_A_ckpt, cfg.prefix)
    E_content_B = load_network(E_content_B, 'E_content_B', cfg.log_dir + 'ckpt/', cfg.prefix)
    E_style_A = load_network(E_style_A, 'E_style_A', cfg.log_dir+ 'ckpt/', cfg.prefix)
    G_A = load_network(G_A, 'G_A', cfg.log_dir+ 'ckpt/', cfg.prefix)
    E_content_A.cuda()
    E_content_B.cuda()
    E_style_A.cuda()
    G_A.cuda()
    E_content_A.eval()
    E_content_B.eval()
    E_style_A.eval()
    G_A.eval()

    os.makedirs(os.path.join(cfg.log_dir, 'results', 'images'), exist_ok=True)
    os.makedirs(os.path.join(cfg.log_dir, 'results', 'stats'), exist_ok=True)

    source_folder = cfg.root + f'data/{cfg.source}_test/'
    test_folder = cfg.root + f'data/{cfg.target}_test/'
    all_source_imgs = sorted([f for f in os.listdir(os.path.join(source_folder, 'images')) if f.endswith('.nii')])
    all_img_paths = sorted([f for f in os.listdir(os.path.join(test_folder, 'images')) if f.endswith('.nii')])
    all_seg_paths = sorted([f for f in os.listdir(os.path.join(test_folder, 'labels')) if f.endswith('.nii')])
    n_samples = len(all_img_paths)
    assert n_samples == len(all_seg_paths)

    evaluation_dice = np.zeros((n_samples, 4))
    evaluation_assd = np.zeros((n_samples, 4))

    for i in range(n_samples):
        source_vol = process_img(load_nii(os.path.join(source_folder, 'images', all_source_imgs[1])), cfg.source)
        img_vol = process_img(load_nii(os.path.join(test_folder, 'images', all_img_paths[i])), cfg.target)
        seg_vol = process_seg(load_nii(os.path.join(test_folder, 'labels', all_seg_paths[i])))
        pred_vol = np.zeros_like(seg_vol)
        for j in range(img_vol.shape[0]):
            input_B = torch.from_numpy(img_vol[[j-1, j, (j+1) % img_vol.shape[0]]].copy()).to(torch.float32)
            input_B = input_B[None, ...]
            input_B = input_B.cuda()
            contents_B, pred_seg = E_content_B(input_B)
            if cfg.gen:
                idx = np.random.randint(0, source_vol.shape[0])
                input_A = torch.from_numpy(source_vol[[idx-1, idx, (idx+1) % source_vol.shape[0]]].copy()).to(torch.float32)
                input_A = input_A[None, ...]
                input_A = input_A.cuda()
                style_A, _, _ = E_style_A(input_A)
                fake_A = G_A(style_A, contents_B)
                if plot:
                    viz_input = 255*normalize_img(img_vol[j])
                    fake_img = 255*normalize_img(fake_A[0,1].detach().cpu().numpy())
                    Image.fromarray(viz_input.astype(np.uint8)).save(os.path.join(cfg.log_dir,'results', 'images', str(i) + '_' + str(j) + 'input_ct.png'))
                    Image.fromarray(fake_img.astype(np.uint8)).save(os.path.join(cfg.log_dir,'results', 'images', str(i) + '_' + str(j) + 'fake_mr.png'))
                if cfg.unet:
                    pred_seg, _ = unet(fake_A)
                else:
                    _, pred_seg = E_content_A(fake_A)
            pred_seg = torch.argmax(pred_seg, dim=1)
            pred_vol[j] = pred_seg.detach().cpu().numpy()

        pred_vol = keep_largest_connected_components(pred_vol)
        pred_vol = np.array(pred_vol).astype(np.uint16)
        if plot:
            for j in range(img_vol.shape[0]):
                gt_seg_img = overlay_seg_img(255*normalize_img(img_vol[j]), seg_vol[j])
                pred_seg_img = overlay_seg_img(255*normalize_img(img_vol[j]), pred_vol[j])
                Image.fromarray(gt_seg_img.astype(np.uint8)).save(os.path.join(cfg.log_dir,'results', 'images', str(i) + '_' + str(j) + 'gt.png'))
                Image.fromarray(pred_seg_img.astype(np.uint8)).save(os.path.join(cfg.log_dir, 'results', 'images', str(i) + '_' + str(j) + 'pred.png'))
        for j in range(1, 5):
            seg_vol_class = seg_vol == j
            pred_vol_class = pred_vol == j

            dice = metric.binary.dc(pred_vol_class, seg_vol_class)
            assd = metric.binary.asd(pred_vol_class, seg_vol_class)
            evaluation_dice[i, j-1] = dice
            evaluation_assd[i, j-1] = assd

        
    print(' => Results Dice mean = ', evaluation_dice.mean(axis=0))
    print(' => Results Dice std = ', evaluation_dice.std(axis=0))
    print('=> Overall mean = ', evaluation_dice.mean(axis=1).mean())
    print('=> Overall std  = ', evaluation_dice.mean(axis=1).std())

    print(' => Results ASSD mean = ', evaluation_assd.mean(axis=0))
    print(' => Results ASSD std = ', evaluation_assd.std(axis=0))
    print('=> Overall mean = ', evaluation_assd.mean(axis=1).mean())
    print('=> Overall std  = ', evaluation_assd.mean(axis=1).std())
    if plot:
        np.save(os.path.join(cfg.log_dir,'results', 'stats',f'{cfg.target}_dice'), evaluation_dice)
        np.save(os.path.join(cfg.log_dir,'results', 'stats', f'{cfg.target}_assd'), evaluation_assd)

def evaluate_bound(cfg, plot=False):
    print(f'--> Evaluating model: from mr to ct.')
    E_content_A = EncoderContent(3, 5, 8)
    E_content_B = EncoderContent(3, 5, 8)
    E_style_A = EncoderStyle(3)
    G_A = SPADEGenerator()
                  
    E_content_A = load_network(E_content_A, 'E_content_mr', cfg.mr_dir + 'ckpt/', 'best')
    E_content_B = load_network(E_content_B, 'E_content_ct_fake', cfg.log_dir + 'ckpt/', 'best')
    E_content_A.cuda()
    E_content_B.cuda()
    E_content_A.eval()
    E_content_B.eval()

    os.makedirs(os.path.join(cfg.log_dir, 'results', 'images'), exist_ok=True)
    os.makedirs(os.path.join(cfg.log_dir, 'results', 'stats'), exist_ok=True)

    test_folder = cfg.root + f'data/ct_test/'
    all_img_paths = sorted([f for f in os.listdir(os.path.join(test_folder, 'images')) if f.endswith('.nii')])
    all_seg_paths = sorted([f for f in os.listdir(os.path.join(test_folder, 'labels')) if f.endswith('.nii')])
    n_samples = len(all_img_paths)
    assert n_samples == len(all_seg_paths)
    
    evaluation_dice_up = np.zeros((n_samples, 4))
    evaluation_assd_up = np.zeros((n_samples, 4))
    evaluation_dice_low = np.zeros((n_samples, 4))
    evaluation_assd_low = np.zeros((n_samples, 4))

    for i in range(n_samples):
        img_vol = process_img(load_nii(os.path.join(test_folder, 'images', all_img_paths[i])), 'ct')
        seg_vol = process_seg(load_nii(os.path.join(test_folder, 'labels', all_seg_paths[i])))
        pred_upper_vol = np.zeros_like(seg_vol)
        pred_lower_vol = np.zeros_like(seg_vol)
        for j in range(img_vol.shape[0]):
            input_B = torch.from_numpy(img_vol[[j-1, j, (j+1) % img_vol.shape[0]]].copy()).to(torch.float32)
            input_B = input_B[None, ...]
            input_B = input_B.cuda()
            _, pred_upper = E_content_B(input_B)
            _, pred_lower = E_content_A(input_B)
            pred_upper = torch.argmax(pred_upper, dim=1)
            pred_lower = torch.argmax(pred_lower, dim=1)
            pred_upper_vol[j] = pred_upper.detach().cpu().numpy()
            pred_lower_vol[j] = pred_lower.detach().cpu().numpy()

        pred_upper_vol = keep_largest_connected_components(pred_upper_vol)
        pred_upper_vol = np.array(pred_upper_vol).astype(np.uint16)
        pred_lower_vol = keep_largest_connected_components(pred_lower_vol)
        pred_lower_vol = np.array(pred_lower_vol).astype(np.uint16)
        if plot:
            for j in range(img_vol.shape[0]):
                pred_seg_lower = overlay_seg_img(255*normalize_img(img_vol[j]), pred_lower_vol[j])
                pred_seg_upper = overlay_seg_img(255*normalize_img(img_vol[j]), pred_upper_vol[j])
                Image.fromarray(pred_seg_lower.astype(np.uint8)).save(os.path.join(cfg.log_dir,'results', 'images', str(i) + '_' + str(j) + 'lower.png'))
                Image.fromarray(pred_seg_upper.astype(np.uint8)).save(os.path.join(cfg.log_dir, 'results', 'images', str(i) + '_' + str(j) + 'upper.png'))
        
        
        for j in range(1, 5):
            seg_vol_class = seg_vol == j
            pred_vol_class = pred_upper_vol == j
            dice = metric.binary.dc(pred_vol_class, seg_vol_class)
            if j in pred_upper_vol:
                assd = metric.binary.asd(pred_vol_class, seg_vol_class)
            else:
                assd = np.nan
            evaluation_dice_up[i, j-1] = dice
            evaluation_assd_up[i, j-1] = assd
        
        for j in range(1, 5):
            seg_vol_class = seg_vol == j
            pred_vol_class = pred_lower_vol == j
            dice = metric.binary.dc(pred_vol_class, seg_vol_class)
            if j in pred_lower_vol:
                assd = metric.binary.asd(pred_vol_class, seg_vol_class)
            else:
                assd = np.nan
            evaluation_dice_low[i, j-1] = dice
            evaluation_assd_low[i, j-1] = assd

    print('=> up')
    print(' => Results Dice mean = ', evaluation_dice_up.mean(axis=0))
    print(' => Results Dice std = ', evaluation_dice_up.std(axis=0))
    print('=> Overall mean = ', evaluation_dice_up.mean(axis=1).mean())
    print('=> Overall std  = ', evaluation_dice_up.mean(axis=1).std())

    print(' => Results ASSD mean = ', evaluation_assd_up.mean(axis=0))
    print(' => Results ASSD std = ', evaluation_assd_up.std(axis=0))
    print('=> Overall mean = ', evaluation_assd_up.mean(axis=1).mean())
    print('=> Overall std  = ', evaluation_assd_up.mean(axis=1).std())

    print('=> low')
    print(' => Results Dice mean = ', evaluation_dice_low.mean(axis=0))
    print(' => Results Dice std = ', evaluation_dice_low.std(axis=0))
    print('=> Overall mean = ', evaluation_dice_low.mean(axis=1).mean())
    print('=> Overall std  = ', evaluation_dice_low.mean(axis=1).std())

    print(' => Results ASSD mean = ', evaluation_assd_low.mean(axis=0))
    print(' => Results ASSD std = ', evaluation_assd_low.std(axis=0))
    print('=> Overall mean = ', evaluation_assd_low.mean(axis=1).mean())
    print('=> Overall std  = ', evaluation_assd_low.mean(axis=1).std())




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--unet', action='store_true')
    parser.add_argument('--prefix', type=str, default='latest')
    #parser.add_argument('--file', type=str, default='03-09-23-55')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = parse_args()
    opt.root = ''
    #opt.log_dir = ''
    opt.source = 'mr'
    opt.target = 'ct'
    opt.log_dir = ''
    opt.mr_dir = ''
    np.random.seed(42)
    evaluate_bound(opt, plot=True)







