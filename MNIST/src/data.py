#!/usr/bin/env python3

import torch
import os

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


class ContinualDataset(Dataset):
    def __init__(self, cfg, train, tid, transform=None, target_transform=None):
        super(ContinualDataset, self).__init__()
        self.cfg = cfg
        if train:
            self.path = cfg.DATA.TRAIN.ROOT
            self.n_samples = cfg.DATA.TRAIN.NUM_SAMPLES
        else:
            self.path = cfg.DATA.TEST.ROOT
            self.n_samples = -1 # Using all test data
        self.data, self.targets = torch.load(self.path)[tid]
        self.tid = tid
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        # if self.num_samples > 0:
        #     return self.cfg.DATA.TRAIN.NUM_SAMPLES * (self.cfg.DATA.NUM_CLASSES // self.cfg.SOLVER.NUM_TASKS)

        # else:
        return len(self.data)


def get_loader(cfg, train, tid):
    transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
    transform = None
    if train:
        loader = DataLoader(ContinualDataset(cfg, True, tid, transform, target_transform=None), batch_size=cfg.SOLVER.BATCH_SIZE,
                            drop_last=False, num_workers=0, shuffle=True)
    else:
        loader = DataLoader(ContinualDataset(cfg, False, tid, transform, target_transform=None), batch_size=cfg.SOLVER.BATCH_SIZE,
                            drop_last=False, num_workers=0, shuffle=True)

    return loader
