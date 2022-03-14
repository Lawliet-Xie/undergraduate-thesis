import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from dataset.transforms import RandomLRFlipPaired, RandomUDFlipPaired, ToTensorPaired, RandomRotatePaired

class MMWHS(Dataset):
    def __init__(self, modality, phase='train', dataroot='./data'):
        super(MMWHS, self).__init__()
        self.images = []
        self.labels = []
        self.phase = [phase] if not isinstance(phase, list) else phase
        assert modality == "ct" or modality == "mr"
        
        if modality == 'ct':
            self.param1 = -2.8
            self.param2 = 3.2
        else:
            self.param1 = -1.8
            self.param2 = 4.4
        
        for ph in self.phase:
            self.images += sorted([os.path.join(dataroot, f'{modality}_{ph}','images', f) for f in os.listdir(os.path.join(dataroot, f'{modality}_{ph}','images')) if f.endswith('npy')])
            self.labels += sorted([os.path.join(dataroot, f'{modality}_{ph}','labels', f) for f in os.listdir(os.path.join(dataroot, f'{modality}_{ph}','labels')) if f.endswith('npy')])
        #self.filter_images()
        # if phase == 'train':
        #     val_images = sorted([os.path.join(dataroot, f'{modality}_{self.phase}','images', f) for f in os.listdir(os.path.join(dataroot, f'{modality}_val','images')) if f.endswith('npy')])
        #     val_labels = sorted([os.path.join(dataroot, f'{modality}_{self.phase}','labels', f) for f in os.listdir(os.path.join(dataroot, f'{modality}_val','labels')) if f.endswith('npy')])
        #     self.images = sorted(list(set(self.images) - set(val_images)))
        #     self.labels = sorted(list(set(self.labels) - set(val_labels)))

        if len(self.images) != len(self.labels):
            raise Exception('Images and labels length do not match')

        self.transfrom = transforms.Compose([RandomRotatePaired(), RandomLRFlipPaired(), RandomUDFlipPaired(), ToTensorPaired()]) if 'train' in self.phase else transforms.Compose([ToTensorPaired()])

    def filter_images(self):
        for img_idx, label_idx in zip(self.images, self.labels):
            assert os.path.basename(img_idx) == os.path.basename(label_idx)
            label = np.load(label_idx)
            label = label[:, :, 1]
            label = np.round(label)
            if np.max(label) == 0:
                self.images.remove(img_idx)
                self.labels.remove(label_idx)

    def check_order(self):
        for img_idx, label_idx in zip(self.images, self.labels):
            if os.path.basename(img_idx) != os.path.basename(label_idx):
                return False
            else:
                return True

    def __getitem__(self, index):
        image = np.load(self.images[index])
        label = np.load(self.labels[index]) / 4.0

        if len(label.shape) == 3:
            image = 2.0*(image - self.param1)/(self.param2 - self.param1) - 1.0
            label = label[:, :, 1]
        
        image, label  = self.transfrom([image, label])

        label = label*4
        label = label.round()
        label = label[0] #totensor
        label = label.to(torch.long)
        return image.to(torch.float32), label
    
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = MMWHS('mr',['train', 'fake'])
    print(len(dataset))
    #image, label = dataset.__getitem__(2805)
    #print(image.shape, label.shape)
    #print(torch.max(label), torch.min(label))
    print(dataset.check_order())
















