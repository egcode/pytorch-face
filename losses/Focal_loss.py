import torch
import torch.nn as nn


class Focal_loss(nn.Module):

    def __init__(self, gamma=0):
        super(Focal_loss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
