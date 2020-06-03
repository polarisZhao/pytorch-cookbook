import torch
import torch.nn as nn
import torch.nn.functional as F

class XXXNet(nn.Module):
    """
    """
    def __init__(self):
        super(XXXNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size = 3, padding=1)
        # ...

    def forward(self, x): 
        out = self.conv1(x)
        # ...        
        return out