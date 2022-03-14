import numpy as np
from skimage import measure
from medpy import metric
import medpy.io as medio
from PIL import Image
import os
import torch
import time
import random
import logging


def create_logger(log_dir):
    # create logger
    os.makedirs(log_dir, exist_ok=True)
    log_file = 'logging.log'
    final_log_file = os.path.join(log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = '[%(asctime)s] %(message)s'

    file = logging.FileHandler(filename=final_log_file, mode='a')
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file)

    return logger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def color_seg(seg):
    colors = torch.tensor([
    [0,     0,   0],
    [254, 232,  81], #LV-myo
    [145, 193,  62], #LA-blood
    [ 29, 162, 220], #LV-blood
    [238,  37,  36]]) #AA
    out = []
    for s in seg:
        out.append((colors[s[0]]).permute(2, 0, 1))
    return torch.stack(out, dim=0)


def overlay_seg_img(img, seg):
    colors = np.array([
    [0,     0,   0],
    [254, 232,  81], #LV-myo(yellow)
    [145, 193,  62], #LA-blood(green)
    [ 29, 162, 220], #LV-blood(blue)
    [238,  37,  36]]) #AA(red)
    # get unique labels
    seg = seg.astype(int)
    labels = np.unique(seg)

    # remove background
    labels = labels[labels !=0]

    # img backgournd
    img_b = img*(seg == 0)

    # final_image
    final_img = np.zeros([img.shape[0], img.shape[1], 3])

    final_img += img_b[:, :, np.newaxis]

    for l in labels:
        mask = seg == l
        img_f = img*mask

        # convert to rgb
        img_f = np.tile(img_f, (3, 1, 1)).transpose(1, 2, 0)

        # colored segmentation
        img_seg = colors[l*mask]

        # alpha overlay
        final_img += 0.5*img_f + 0.5*img_seg
    
    return final_img

def load_nii(img_path):
    nimg = medio.load(img_path)
    return nimg[0]

def process_img(vol, modality):
    vol = np.flip(vol, axis=0)
    vol = np.flip(vol, axis=1)
    if modality == 'ct':
        param1 = -2.8
        param2 = 3.2
    else:
        param1 = -1.8
        param2 = 4.4
    batch_wise = np.transpose(vol, (2, 0, 1))
    batch_wise = 2*(batch_wise - param1)/(param2 - param1)
    return batch_wise

def process_seg(vol):
    vol = np.flip(vol, axis=0)
    vol = np.flip(vol, axis=1)
    batch_wise = np.transpose(vol, (2, 0, 1))
    return batch_wise

def normalize_img(img):
    return (img - img.min()) / (img.max() - img.min())

def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    num_channel = mask.shape[1]
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in range(1, num_channel + 1):

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img

def to_categorical(mask, num_classes):
    eye = np.eye(num_classes, dtype='uint8')
    if mask.shape[1] == 1:
        mask = np.squeeze(mask, axis=1)
    output = eye[mask]
    output = np.moveaxis(output, -1, 1)
    return output

def cal_metric(img_gt, img_pred):
    # img_gt = to_categorical(gt, 5)
    # img_pred = to_categorical(pred, 5)
    assert img_gt.shape == img_pred.shape
    res = {}
    class_name = ["myo", "la", "lv", "aa"]
    # Loop on each classes of the input images
    for c, cls_name in zip([1, 2, 3, 4], class_name) :

        gt_c_i = np.where(img_gt == c, 1, 0)
        pred_c_i = np.where(img_pred == c, 1, 0)

        # Compute the Dice
        dice = metric.binary.dc(gt_c_i, pred_c_i)
        res[cls_name] = dice
    return res

def one_hot(targets):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), 5, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot