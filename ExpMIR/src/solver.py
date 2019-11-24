#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import torch.optim as optim

from config import cfg
from data import get_loader
from model import Model
from memory_buffer import Buffer
from utils.loss import BCEauto
from utils.utils import AverageMeter
from utils.logger import setup_logger
import logging


def update_virtual(model, optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def restore_model(model, params):
    model.load_state_dict(params)

def train_one_epoch(cfg, model, train_loader, optimizer, mem, logger):
    criterion_per_item, criterion_avg = BCEauto(reduction='none'), BCEauto(reduction='mean')
    avg_loss = AverageMeter()
    for bid, data in enumerate(train_loader):
        x, y = data
        x = x.view(min(x.shape[0], cfg.SOLVER.BATCH_SIZE), -1)
        params = model.state_dict()
        r_sample_batch = mem.sample()
        if r_sample_batch:
            if cfg.SOLVER.SAMPLING_CRITERION == 2:
                x_c, y_c, best_loss_c = r_sample_batch
            else:
                r_sample_batch = mem.sample()
                x_c, y_c, _ = r_sample_batch

            pred = model(x_c)
            y_c = torch.squeeze(y_c)
            loss_per_item = torch.mean(criterion_per_item(pred, y_c), axis=1)
            loss_mean = criterion_avg(pred, y_c)
            # Update Model
            update_virtual(model, optimizer, loss_mean)
            pred = model(x_c)
            virt_loss_per_item = torch.mean(criterion_per_item(pred, y_c), axis=1)
            _, bc_idx = torch.topk(virt_loss_per_item - loss_per_item, cfg.SOLVER.BUDGET)
            # Restore Model
            restore_model(model, params)
            x_union, y_union= torch.cat((x, x_c[bc_idx])), torch.cat((y, y_c[bc_idx]))
        else:
            x_union, y_union= x, y

        pred = model(x_union)
        loss_mean= criterion_avg(pred, y_union)
        avg_loss.update(loss_mean)
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()
        loss_per_item = torch.mean(criterion_per_item(model(x), y), axis=1)
        mem.fill(x, y, loss_per_item)
    logger.debug(f'Epoch ended, Average Loss:{avg_loss.avg}')

def train(cfg, model, train_loader, logger):
    optimizer = optim.SGD(model.parameters(), lr = cfg.OPTIMIZER.LR, momentum = cfg.OPTIMIZER.MOMENTUM)
    mem = Buffer(cfg)
    for epoch_idx in range(0, cfg.SOLVER.NUM_EPOCHS):
        train_one_epoch(cfg, model, train_loader, optimizer, mem, logger)


def main():
    model = Model(cfg.MODEL.MLP.INPUT_SIZE, cfg.MODEL.MLP.HIDDEN_SIZE, cfg.MODEL.MLP.OUTPUT_SIZE)
    logger = setup_logger(cfg.SYSTEM.EXP_NAME, cfg.SYSTEM.SAVE_DIR, 0)

    for tid in range(cfg.SOLVER.NUM_TASKS):
        train_loader = get_loader(cfg, tid)
        train(cfg, model, train_loader, logger)
    # test_loader = get_loader(cfg, -1)

if __name__ == "__main__":
        main()
