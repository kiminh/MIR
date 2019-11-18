#!/usr/bin/env python3

import time
import torch
import numpy as np
import random
import os
import shutil
from datetime import datetime

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum().item()
            res.append(correct_k*100.0 / batch_size)

        if len(res)==1:
            return res[0]
        else:
            return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Timer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval


def save_config(save_root, cfg, clean_run):
    if clean_run:
        if os.path.exists(os.path.join(cfg.SYSTEM.SAVE_DIR, cfg.SYSTEM.EXP_NAME)):
            shutil.rmtree(os.path.join(cfg.SYSTEM.SAVE_DIR, cfg.SYSTEM.EXP_NAME))
        if os.path.exists(cfg.SYSTEM.SAVE_DIR+f'/runs/{cfg.SYSTEM.EXP_NAME}'):
           shutil.rmtree(cfg.SYSTEM.SAVE_DIR+f'/runs/{cfg.SYSTEM.EXP_NAME}')
           time.sleep(5)

    dir_name = 'runs'
    dir_name = os.path.join(save_root, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fn = cfg.SYSTEM.EXP_NAME
    log_dir = os.path.join(dir_name, fn)

    dir_name = cfg.SYSTEM.EXP_NAME
    dir_name = os.path.join(save_root, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    chk_dir = os.path.join(dir_name, 'chkpt')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(chk_dir):
        os.makedirs(chk_dir)

    with open(os.path.join(dir_name, 'config.txt'), 'w') as f:
        f.write(str(cfg))

    return log_dir, chk_dir

def set_seed(cfg):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = cfg.SYSTEM.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
