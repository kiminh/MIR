#!/usr/bin/env python
from collections import defaultdict

import numpy as np
from operator import itemgetter
import torch

class Buffer:
    def __init__(self, cfg):
        self.size = cfg.BUFFER.SIZE
        self.C = cfg.BUFFER.C
        self.memory = []
        self.loss = defaultdict(lambda:1e+6)
        self.num_seen = 0
        self.num_tasks = cfg.SOLVER.NUM_TASKS
        self.num_cls = cfg.DATA.NUM_CLASSES

    def fill(self, x, y, loss=None, tid=None):
        """
        Reservoir Sampling for selecting which tensor make it
        into the memory buffer. Here we use id function for
        hashing tensor. See https://github.com/pytorch/pytorch/pull/10625
        # TODO: Check the speed of this function
        Args
        ---
        x: [None, tensor], Batch of tensor data
        y: [None, tensor], Batch of tensor labels
        loss: [None, tensor], Loss associated with each tensor data value
        """
        if tid is not None:
            mult = self.num_cls / self.num_tasks
            self.eff_size = (tid+1) * int(mult*(self.size / (self.num_cls)))
        if loss:
            loss = loss.detach()
        for i in range(0, x.shape[0]):
            if len(self.memory) < self.eff_size:
                self.memory.append((x[i], y[i]))
                if loss:
                    self.loss[id(x[i])] = min(self.loss[id(x[i])], loss[i])
            else:
                if np.random.randint(0, self.num_seen + i) < len(self.memory):
                    self.memory[i] = (x[i], y[i])
                if loss:
                    self.loss[id(x[i])] = min(self.loss[id(x[i])], loss[i])

    def sample(self):
        if len(self.memory) == 0:
            return None
        idx = torch.randperm(len(self.memory))[:self.C].tolist()
        mem_sampled = [self.memory[i] for i in idx]
        if len(self.loss)==0:
            return mem_sampled, -1
        else:
            loss_keys = [id(x) for x in self.memory[idx][0]]
            loss_idx = list(itemgetter(*loss_keys)(self.loss))
            return mem_sampled, loss_idx

    def reset(self):
        self.memory = []
        self.best_loss = defaultdict(np.inf)
        self.num_seen = 0
