from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import math
# from pdb import set_trace as bp

# class Arcface_loss(nn.Module):
#     # def __init__(self, num_classes, feat_dim, device, s=7.0, m=0.2):
#     def __init__(self, num_classes, feat_dim, device, s=64.0, m=0.5):
    
#         super(Arcface_loss, self).__init__()
#         self.feat_dim = feat_dim
#         self.num_classes = num_classes
#         self.s = s
#         self.m = m
#         self.weights = nn.Parameter(torch.randn(num_classes, feat_dim))
#         self.device = device

#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.mm = math.sin(math.pi-m)*m
#         self.threshold = math.cos(math.pi-m)

#         # For Softmax Feeding after model features
#         self.prelu = nn.PReLU().to(self.device)

#     def forward(self, feat, label):
#         # For Softmax Feeding after model features    
#         feat_prelu = self.prelu(feat)

#         eps = 1e-4
#         batch_size = feat_prelu.shape[0]
#         norms = torch.norm(feat_prelu, p=2, dim=-1, keepdim=True)
#         feat_l2norm = torch.div(feat_prelu, norms)
#         feat_l2norm = feat_l2norm * self.s

#         norms_w = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
#         weights_l2norm = torch.div(self.weights, norms_w)

#         fc7 = torch.matmul(feat_l2norm, torch.transpose(weights_l2norm, 0, 1))

#         if torch.cuda.is_available():
#             label = label.cuda()
#             fc7 = fc7.cuda()
#         else:
#             label = label.cpu()
#             fc7 = fc7.cpu()

#         target_one_hot = torch.zeros(len(label), self.num_classes).to(self.device)
#         target_one_hot = target_one_hot.scatter_(1, label.unsqueeze(1), 1.)        
#         zy = torch.addcmul(torch.zeros(fc7.size()).to(self.device), 1., fc7, target_one_hot)
#         zy = zy.sum(-1)

#         cos_theta = zy/self.s
#         cos_theta = cos_theta.clamp(min=-1+eps, max=1-eps) # for numerical stability

#         theta = torch.acos(cos_theta)
#         theta = theta+self.m

#         body = torch.cos(theta)
#         new_zy = body*self.s

#         diff = new_zy - zy
#         diff = diff.unsqueeze(1)

#         body = torch.addcmul(torch.zeros(diff.size()).to(self.device), 1., diff, target_one_hot)
#         output = fc7+body

#         return output.to(self.device)
  
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output
    
class Arcface_loss(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, num_classes, feat_dim, device, s=64.0, m=0.5):
        super(Arcface_loss, self).__init__()
        self.num_classes = num_classes
        self.kernel = nn.Parameter(torch.Tensor(feat_dim,num_classes))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
        self.device = device

    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
#         return output
        return output.to(self.device)
