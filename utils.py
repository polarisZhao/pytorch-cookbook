import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# conv
def conv3x3(in_planes, out_planes, stride=1, padding=1, groups=1):
    """3x3 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                    bias=False)

# BasicConv2d
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# VGG Block: TBD

# Resnet Block
class ResnetBlockA(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(ResnetBlockA, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('ResnetBlockA only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResnetBlcokB(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Inception 
class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# SENet
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ShufflenetV1 & V2
class ShufflenetV1Block(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, flag=False):
        super(ShufflenetBlockV1, self).__init__()
        self.downsample = downsample
        group_num = 3
        self.flag = flag
        if self.flag:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, groups=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, groups=group_num, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, groups=group_num, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def _shuffle(self, features, g):
        channels = features.size()[1] 
        index = torch.from_numpy(np.asarray([i for i in range(channels)]))
        index = index.view(-1, g).t().contiguous()
        index = index.view(-1).cuda()
        features = features[:, index]
        return features

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.flag:
            out = self._shuffle(out, 3)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            out = torch.cat((out, residual), 1) 
        else:
            out += residual
        out = self.relu(out)

        return out

class ShufflenetV2Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ShufflenetBlockV2, self).__init__()
        self.downsample = downsample

        if not self.downsample: #---if not downsample, then channel split, so the channel become half
            inplanes = inplanes // 2
            planes = planes // 2
 
        self.conv1x1_1 = conv1x1(in_channels=inplanes, out_channels=planes)
        self.conv1x1_1_bn = nn.BatchNorm2d(planes)

        self.dwconv3x3 = conv3x3(in_channels=planes, out_channels=planes, stride=stride, groups=planes)
        self.dwconv3x3_bn= nn.BatchNorm2d(planes)

        self.conv1x1_2 = conv1x1(in_channels=planes, out_channels=planes)
        self.conv1x1_2_bn = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def _channel_split(self, features, ratio=0.5):
        size = features.size()[1]
        split_idx = int(size * ratio)
        return features[:,:split_idx], features[:,split_idx:]

    def _channel_shuffle(self, features, g=2):
        channels = features.size()[1] 
        index = torch.from_numpy(np.asarray([i for i in range(channels)]))
        index = index.view(-1, g).t().contiguous()
        index = index.view(-1).cuda()
        features = features[:, index]
        return features

    def forward(self, x):
        if  self.downsample:
            # x1 = x.clone() #----deep copy x, so where x2 is modified, x1 not be affected
            x1 = x
            x2 = x
        else:
            x1, x2 = self._channel_split(x)

        #----right branch----- 
        x2 = self.conv1x1_1(x2)
        x2 = self.conv1x1_1_bn(x2)
        x2 = self.relu(x2)
         
        x2 = self.dwconv3x3(x2)
        x2 = self.dwconv3x3_bn(x2)
    
        x2 = self.conv1x1_2(x2)
        x2 = self.conv1x1_2_bn(x2)
        x2 = self.relu(x2)

        #---left branch-------
        if self.downsample:
            x1 = self.downsample(x1)

        x = torch.cat([x1, x2], 1)
        x = self._channel_shuffle(x)
        return x
 
# Mobilenet V1
class MobileNetV1Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, 
                               stride=stride, padding=1,
                               groups=in_c, bias=False)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=1,
                              stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
    
    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

# Mobilenet V2
class MobilenetV2Block(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
 
