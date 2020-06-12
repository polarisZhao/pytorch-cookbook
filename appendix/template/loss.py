import torch
import torch.nn as nn

class RRLoss(nn.Module):
    def __init__(self):
        """ RRLoss"""
        super(RRLoss, self).__init__()

    def forward(self, predict, data):
        return torch.mean(torch.square(predict  - data))