import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
 

class BarlowTwins(nn.Module):
    def __init__(self, bs, lamda):
        super().__init__()
        self.bs = bs
        self.lamda = lamda
        self.bn = nn.BatchNorm1d(1024, affine=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, image_features1, image_features2):
        z1 = image_features1
        z2 = image_features2

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        #c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.bs)
        #torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamda * off_diag
        return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

