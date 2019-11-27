#!/usr/bin/env python3
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from data import get_loader
from memory_buffer import Buffer
from model import Model
from utils.logger import setup_logger
from utils.loss import BCEauto
from utils.metrics import Metrics
from utils.utils import AverageMeter, save_config, set_seed
from utils.basic import get_counts, get_counts_labels

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def update_virtual(model, optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def restore_model(model, params):
    model.load_state_dict(params)

def train(cfg, model, train_loader, tid, mem, logger, writer, metrics):
    criterion_avg =  torch.nn.CrossEntropyLoss()
    criterion_per_item = torch.nn.CrossEntropyLoss(reduction='none')
    avg_loss = AverageMeter()
    batch_size = cfg.SOLVER.BATCH_SIZE
    num_batches = cfg.DATA.TRAIN.NUM_SAMPLES // batch_size
    optimizer = optim.SGD(model.parameters(), lr = cfg.OPTIMIZER.LR, momentum = cfg.OPTIMIZER.MOMENTUM)
    acc = 0.0
    for epoch_idx in range(0, cfg.SOLVER.NUM_EPOCHS):
        for batch_idx, data in enumerate(train_loader):
            writer_idx = batch_idx * batch_size + (epoch_idx * num_batches * batch_size)
            x, y = data
            x = x.to(device)
            y = y.to(device)
            x_orig, y_orig = x.clone(), y.clone()
            # Safe gaurading against for incomplete batches
            x = x.view(min(x.shape[0], cfg.SOLVER.BATCH_SIZE), -1)
            params = model.state_dict()
            # Random sampled C items
            sampled_mem, loss = mem.sample()
            if sampled_mem is not None:
                x_c = torch.stack([x[0] for x in sampled_mem]).to(device)
                y_c = torch.stack([x[1] for x in sampled_mem]).to(device)
                if cfg.SOLVER.SAMPLING_CRITERION == 2 and not loss == -1:
                    best_loss_per_item = torch.tensor(loss)
                # loss for sampled before virtual updates
                output = model(x_c)
                y_c = torch.squeeze(y_c)
                loss_per_item = criterion_per_item(output, y_c)
                loss_mean = criterion_avg(output, y_c)

                # Virtual Update
                update_virtual(model, optimizer, loss_mean)

                # loss for sampled after virtual updates
                output = model(x_c)
                virt_loss_per_item = criterion_per_item(output, y_c)
                if cfg.SOLVER.SAMPLING_CRITERION == 1:
                    _, bc_idx = torch.topk(virt_loss_per_item - loss_per_item, cfg.SOLVER.BUDGET)
                elif cfg.SOLVER.SAMPLING_CRITERION == 2:
                    _, bc_idx = torch.topk(virt_loss_per_item - torch.min(loss_per_item, best_loss_per_item), cfg.SOLVER.BUDGET)
                # Restore Model
                restore_model(model, params)
                x_union, y_union= torch.cat((x, x_c[bc_idx])), torch.cat((y, torch.squeeze(y_c[bc_idx])))
            else:
                x_union, y_union= x, y

            output = model(x_union)
            loss_mean= criterion_avg(output, y_union)
            pred = output.argmax(dim=1, keepdim=True)
            acc += pred.eq(y_union.view_as(pred)).sum().item() / x_union.shape[0]
            avg_loss.update(loss_mean)
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            # import ipdb; ipdb.set_trace()
            loss_per_item = criterion_per_item(model(x_orig), y_orig)
            mem.fill(x, y, loss_per_item, tid)
            writer.add_scalar(f'loss-{tid}', loss_mean, writer_idx)
            mem.num_seen += batch_size
            if batch_idx % cfg.SYSTEM.LOG_FREQ==0:
                logger.debug(f'Batch Id:{batch_idx}, Average Loss:{avg_loss.avg}')
                print(f'Labels: {get_counts_labels(y_union)},\
                        Memory: {get_counts(mem.memory)},\
                        Eff Size: {mem.eff_size},\
                        Memory Size: {len(mem.memory)},\
                        Num Seen:{mem.num_seen}')
        logger.info(f'Task Id:{tid}, Acc:{acc/len(train_loader)}')
    test(cfg, model, logger, writer, metrics, tid)

def test(cfg, model, logger, writer, metrics, tid_done):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loaders = [(tid, get_loader(cfg, False, tid)) for tid in range(tid_done+1)]
    avg_meter = AverageMeter()
    for tid, test_loader in test_loaders:
        avg_meter.reset()
        for idx, data in enumerate(test_loader):
            x, y = data
            output = model(x)
            test_loss = criterion(output, y)
            pred = output.argmax(dim=1, keepdim=True)
            acc = metrics.accuracy(tid, tid_done, pred, y)
        metrics.avg_accuracy(tid, tid_done, len(test_loader.dataset))
        metrics.forgetting(tid, tid_done)
    logger.info(f'Task Done:{tid_done},\
                  Test Acc:{metrics.acc_task(tid_done)},\
                  Test Forgetting:{metrics.forgetting_task(tid_done)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_run', type=bool, default=True)
    parser.add_argument('--config_file', type=str, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg)
    log_dir, chkpt_dir = save_config(cfg.SYSTEM.SAVE_DIR, cfg, args.clean_run)
    logger = setup_logger(cfg.SYSTEM.EXP_NAME, os.path.join(cfg.SYSTEM.SAVE_DIR, cfg.SYSTEM.EXP_NAME), 0)
    writer = SummaryWriter(log_dir)
    metrics = Metrics(cfg.SOLVER.NUM_TASKS)

    model = Model(cfg.MODEL.MLP.INPUT_SIZE, cfg.MODEL.MLP.HIDDEN_SIZE, cfg.MODEL.MLP.OUTPUT_SIZE)
    mem = Buffer(cfg)
    for tid in range(cfg.SOLVER.NUM_TASKS):
        train_loader = get_loader(cfg, True, tid)
        print(torch.unique(train_loader.dataset.targets))
        train(cfg, model, train_loader, tid, mem, logger, writer, metrics)
    logger.info(f'Avg Acc:{metrics.acc_task(cfg.SOLVER.NUM_TASKS-1)},\
                  Avg Forgetting:{metrics.forgetting_task(cfg.SOLVER.NUM_TASKS-1)}')
