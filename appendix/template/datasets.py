import os
import random
from glob import glob

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

def transform_train(data, target):
    # random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        data, output_size=(96, 96))
    data = F.crop(data, i, j, h, w)
    target = F.crop(target, i, j, h, w)
    # hflip
    if random.random() > 0.5:
        data = F.hflip(data)
        target = F.hflip(target)
    return F.to_tensor(data), F.to_tensor(target)

def transform_test(data, target):
    return F.to_tensor(data), F.to_tensor(target)


class KKDataset(Dataset):
    """
    """
    def __init__(self, root_dir='./data/train', is_trainval = True, transform=None):
        """
        """
        self.root_dir = root_dir
        self.is_trainval = is_trainval
        self.transform = transform
        self.train_data = sorted(glob(os.path.join(self.root_dir, "low/*.png")))
        self.train_target = sorted(glob(os.path.join(self.root_dir, "high/*.png")))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = Image.open(self.train_data[idx])
        target = Image.open(self.train_target[idx])

        if self.transform:
            data, target = self.transform(data, target)

        sample = {'data': data, 'target': target}
        return sample