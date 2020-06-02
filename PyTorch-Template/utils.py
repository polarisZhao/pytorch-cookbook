import os
import shutil
import torch

def save_checkpoint(state, is_best, checkpoint_dir):
    torch.save(state, os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar'),
                        os.path.join(checkpoint_dir, 'checkpoint__best.pth.tar'))