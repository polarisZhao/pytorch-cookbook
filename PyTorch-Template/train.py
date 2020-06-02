from __future__ import print_function

import os
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import KKDataset, transform_train, transform_test #
from models import XXXNet # 
from loss import RRLoss #
from metrics import PSNR # 
from utils import * 

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Template training')
# env
parser.add_argument('--use_gpu', action='store_true', default=True, help='enables/disables CUDA training')
parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed (default: 2020)')
parser.add_argument('--log_path', type=str, default='./logs', metavar='PATH', help='path to save logs (default: current directory/logs)')
# data
parser.add_argument('--train_dataset', type=str, default='./data/train', help='training dataset path')
parser.add_argument('--valid_dataset', type=str, default='./data/valid', help='test dataset path')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
parser.add_argument('--val_batch_size', type=int, default=16, metavar='N', help='input batch size for testing (default: 16)')
# models
parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to checkpoint path(default: none)')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', metavar='PATH', help='path to save prune model (default: current directory/checkpoints)')
# optimizer & lr
parser.add_argument('--init_lr', type=float, default=0.01, metavar='LR', help='init learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
# epoch & save-interval
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 160)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--save_interval', type=int, default=1, metavar='N', help='how many batches to wait before logging training status')
args = parser.parse_args()


# global setting
torch.manual_seed(args.seed)

device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
if device == "cuda":
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.ckpt_path)
if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)


# dataset
train_dataset = KKDataset(args.train_dataset, is_trainval = True, transform = transform_train) #
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=0, drop_last=False) #
valid_dataset = KKDataset(args.valid_dataset, is_trainval = True, transform = transform_test) # 
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.val_batch_size, 
                                           shuffle=True, num_workers=0) #
# model & loss
model = XXXNet().to(device) #
lossfunc = RRLoss() #
criterion = PSNR() #
# lr & optimizer
optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)

# load resume
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

def train(epoch):
    model.train()

    avg_loss = 0.0
    train_acc = 0.0
    for batch_idx, batchdata in enumerate(train_loader):
        data, target = batchdata["data"], batchdata["target"] #
        data, target = data.to(device), target.to(device)  #
        optimizer.zero_grad()

        predict = model(data) # 
        loss = lossfunc(predict, target) #
        avg_loss += loss.item() #

        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    if (epoch + 1) %  args.save_interval == 0:
        state = { 'epoch': epoch + 1,
                   'state_dict': model.state_dict(),
                   'best_prec': 0.0,
                   'optimizer': optimizer.state_dict()}
        model_path = os.path.join(args.checkpoint_dir, 'model_' + str(epoch) + '.pth')
        torch.save(state, model_path)


def test():
    model.eval()

    test_loss = 0
    for batch_idx, batchdata in enumerate(valid_loader):
        data, target = batchdata["data"], batchdata["target"] #
        data, target = data.to(device), target.to(device) #
        predict = model(data) # 
        test_loss += lossfunc(predict, target) #
        psnr = criterion(predict * 255, target * 255) #

    test_loss /= len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, loss:{}, PSNR: ({:.1f})\n'.format(
        test_loss, test_loss / len(valid_loader.dataset), psnr / len(valid_loader.dataset)))
    return psnr / float(len(valid_loader.dataset))


best_prec = 0.0
for epoch in range(args.start_epoch, args.epochs):
    train(epoch)
    scheduler.step()
    print(print(optimizer.state_dict()['param_groups'][0]['lr']))

    current_prec = test() 
    is_best = current_prec > best_prec #　更改大小写 !
    best_prec = max(best_prec, best_prec) #  max or min

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec': best_prec,
        'optimizer': optimizer.state_dict(),
    }, is_best, args.checkpoint_dir)
