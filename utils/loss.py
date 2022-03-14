from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, mode='lsgan'):
        super(GANLoss, self).__init__()
        if mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("invalid mode.")
        
    def get_target_tensor(self, prediction, is_real):
        if is_real:
            return torch.ones_like(prediction)
        else:
            return torch.zeros_like(prediction)
    
    def forward(self, prediction, is_real):
        target = self.get_target_tensor(prediction, is_real)
        loss = self.loss(prediction, target)
        return loss

class BCEGANLoss(nn.Module):
    def __init__(self):
        super(BCEGANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def get_target_tensor(self, prediction, is_real):
        if is_real:
            return torch.ones_like(prediction)
        else:
            return torch.zeros_like(prediction)
    
    def forward(self, prediction, is_real, prob=0.5):
        target = self.get_target_tensor(prediction, is_real)
        loss = self.loss(prediction, target*prob)
        return loss


class CrossEntropyLossWeighted(nn.Module):
    def __init__(self, n_classes=5):
        super(CrossEntropyLossWeighted, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.n_classes = n_classes

    def one_hot(self, targets):
        targets_extend=targets.clone()
        targets_extend.unsqueeze_(1) # convert to Nx1xHxW
        one_hot = torch.cuda.FloatTensor(targets_extend.size(0), self.n_classes, targets_extend.size(2), targets_extend.size(3)).zero_()
        one_hot.scatter_(1, targets_extend, 1)
        
        return one_hot
    
    def forward(self, inputs, targets):
        one_hot = self.one_hot(targets)

        # size is batch, nclasses, 256, 256
        weights = 1.0 - torch.sum(one_hot, dim=(2, 3), keepdim=True)/torch.sum(one_hot)
        one_hot = weights*one_hot

        loss = self.ce(inputs, targets).unsqueeze(1) # shape is batch, 1, 256, 256
        loss = loss*one_hot

        return torch.sum(loss)/(torch.sum(weights)*targets.size(0)*targets.size(1))

class OrthonormalityLoss(nn.Module):
    def __init__(self, size):
        super(OrthonormalityLoss, self).__init__()
        self.size = size
        self.lower_tr = torch.zeros(size, size).cuda()
        indices = torch.tril_indices(size, size, offset=-1)

        # # lower triangular matrix as mask
        # self.lower_tr[indices[0], indices[1]] = 1.0
        self.id = torch.eye(self.size).cuda()

    def forward(self, attentions):
        x = attentions.view(attentions.shape[0], attentions.shape[1], -1)

        # normalize xx
        x = nn.functional.normalize(x, dim=2)

        #batch_matrix = torch.matmul(x, torch.transpose(x, 1, 2))*self.lower_tr
        batch_matrix = (torch.matmul(x, torch.transpose(x, 1, 2)) - self.id)
        
        
        cost = 2*(batch_matrix**2).sum()/(self.size * (self.size - 1.0))

        return cost

class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())