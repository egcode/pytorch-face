from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
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
  
class Arcface_loss(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, num_classes, feat_dim, device, s=64.0, m=0.50, easy_margin = False):
        super(Arcface_loss, self).__init__()
        self.in_features = feat_dim
        self.out_features = num_classes

        self.device = device

        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
      
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        one_hot = one_hot.to(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        # return output
        return output.to(self.device)
