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
        self.loss = []
        self.num_seen = 0
        self.num_tasks = cfg.SOLVER.NUM_TASKS
        self.num_cls = cfg.DATA.NUM_CLASSES

    def fill(self, x, y, use_loss=False, loss=None, tid=None):
        """
        Reservoir Sampling for selecting which tensor make it
        into the memory buffer. Check Notes for explanation
        # TODO: Check the speed of this function
        Args
        ---
        x: [None, tensor], Batch of tensor data
        y: [None, tensor], Batch of tensor labels
        loss: [None, tensor], Loss associated with each tensor data value
        """
        if tid is not None:
            mult = self.num_cls / self.num_tasks
            self.eff_size = (tid+1) * int(mult * self.size / self.num_cls)
        if use_loss:
            loss = loss.detach()
        for i in range(0, x.shape[0]):
            if len(self.memory) < self.size:
                self.memory.append((x[i], y[i]))
                if use_loss and not len(loss) == 0:
                    self.loss.append(loss[i])
            else:
                loc = np.random.randint(0, self.num_seen + i)
                if loc < len(self.memory):
                    self.memory[loc] = (x[i], y[i])
                    if use_loss and not len(loss) == 0:
                        self.loss[i] = min(self.loss[i], loss[i])
            # self.num_seen += 1

    def sample(self):
        if len(self.memory) == 0:
            return (None, -1)
        idx = torch.randperm(len(self.memory))[:self.C].tolist()
        mem_sampled = [self.memory[i] for i in idx]
        if len(self.loss)==0:
            return (mem_sampled, -1)
        else:
            loss = [self.loss[i] for i in idx]
            return (mem_sampled, loss)

    def reset(self):
        self.memory = []
        self.best_loss = defaultdict(np.inf)
        self.num_seen = 0
