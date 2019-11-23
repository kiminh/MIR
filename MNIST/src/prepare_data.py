#!/usr/bin/env python3
# Modified from Code of GEM.


from config import cfg
import os.path
import torch
import argparse
import random
import numpy as np
from utils.utils import set_seed


def save_permuted_dataset():
    train_data = {}
    test_data = {}

    x_tr, y_tr = torch.load(os.path.join(cfg.DATA.ROOT, 'training.pt'))
    x_te, y_te = torch.load(os.path.join(cfg.DATA.ROOT, 'test.pt'))

    x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
    x_te= x_te.float().view(x_te.size(0), -1) / 255.0
    y_tr = y_tr.view(-1).long()
    y_te = y_te.view(-1).long()

    for tid in range(cfg.SOLVER.NUM_TASKS):
        p = torch.randperm(x_tr.size(1)).long().view(-1)
        train_data[tid] = (x_tr.index_select(1, p), y_tr)
        test_data[tid] = (x_te.index_select(1, p), y_te)

    torch.save(train_data, cfg.DATA.SAVE_FILE + 'train.pt')
    torch.save(train_data, cfg.DATA.SAVE_FILE + 'test.pt')


def save_split_dataset():
    train_data = {}
    test_data = {}

    x_tr, y_tr = torch.load(os.path.join(cfg.DATA.ROOT, 'training.pt'))
    x_te, y_te = torch.load(os.path.join(cfg.DATA.ROOT, 'test.pt'))
    x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
    x_te= x_te.float().view(x_te.size(0), -1) / 255.0
    y_tr = y_tr.view(-1).long()
    y_te = y_te.view(-1).long()

    num_classes = cfg.DATA.NUM_CLASSES
    s_cls = torch.arange(num_classes)
        n_labels = cfg.DATA.NUM_CLASSES // cfg.SOLVER.NUM_TASKS
        cids = s_cls[tid*n_labels: (tid+1)*n_labels]
        idx_tr = []
        idx_te = []
        for cid in cids:
            idx_tr.extend(torch.nonzero(y_tr == cid).view(-1))
            idx_te.extend(torch.nonzero(y_te==cid).view(-1))
        train_data[tid] = (x_tr[idx_tr], y_tr[idx_tr])
        test_data[tid] = (x_te[idx_te], y_te[idx_te])

    torch.save(train_data, cfg.DATA.SAVE_FILE + 'train.pt')
    torch.save(test_data, cfg.DATA.SAVE_FILE + 'test.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_type', help='Type of dataset to prepare', default='split', type=str)
    parser.add_argument('--n_task', help='Number of tasks', default=5, type=int)
    parser.add_argument('--save_file', help='save file', default='./torch_data/mnist/split/', type=str)

    args = parser.parse_args()
    opts = ['SOLVER.NUM_TASKS', args.n_task, 'DATA.SAVE_FILE', args.save_file]
    cfg.merge_from_list(opts)
    cfg.freeze()
    # set_seed(cfg)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    if args.d_type == 'permute':
        save_permuted_dataset()
    else:
        save_split_dataset()
