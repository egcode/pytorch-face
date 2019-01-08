
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import math
# from pdb import set_trace as bp

class Arcface_loss(nn.Module):
    def __init__(self, num_classes, feat_dim, device, s=7.0, m=0.2):
        super(Arcface_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.device = device

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi-m)*m
        self.threshold = math.cos(math.pi-m)

    def forward(self, feat, label):
        eps = 1e-4
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        feat_l2norm = torch.div(feat, norms)
        feat_l2norm = feat_l2norm * self.s

        norms_w = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        weights_l2norm = torch.div(self.weights, norms_w)

        fc7 = torch.matmul(feat_l2norm, torch.transpose(weights_l2norm, 0, 1))

        if torch.cuda.is_available():
            label = label.cuda()
            fc7 = fc7.cuda()
        else:
            label = label.cpu()
            fc7 = fc7.cpu()

        target_one_hot = torch.zeros(len(label), self.num_classes).to(self.device)
        target_one_hot = target_one_hot.scatter_(1, label.unsqueeze(1), 1.)        
        zy = torch.addcmul(torch.zeros(fc7.size()).to(self.device), 1., fc7, target_one_hot)
        zy = zy.sum(-1)

        cos_theta = zy/self.s
        cos_theta = cos_theta.clamp(min=-1+eps, max=1-eps) # for numerical stability

        theta = torch.acos(cos_theta)
        theta = theta+self.m

        body = torch.cos(theta)
        new_zy = body*self.s

        diff = new_zy - zy
        diff = diff.unsqueeze(1)

        body = torch.addcmul(torch.zeros(diff.size()).to(self.device), 1., diff, target_one_hot)
        output = fc7+body

        return output.to(self.device)
  
