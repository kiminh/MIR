#!/usr/bin/env python
from collections import defaultdict

import numpy as np
from operator import itemgetter
import torch

class Buffer:
    def __init__(self, cfg):
        self.size = cfg.BUFFER.SIZE
        self.C = cfg.BUFFER.C
        self.memory_x = torch.zeros((self.size, cfg.MODEL.MLP.INPUT_SIZE))
        self.memory_y = torch.zeros((self.size, 1), dtype=torch.long)
        self.loss = defaultdict(lambda:1e+6)
        self.num_seen = 0
        self.task_count = 0

    def fill(self, x, y, loss):
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
        loss = loss.detach()
        for i in range(0, x.shape[0]):
            if len(self.loss) < self.size:
                self.memory_x[i] = x[i]
                self.memory_y[i] = y[i]
                self.loss[id(x[i])] = min(self.loss[id(x[i])], loss[i])
            else:
                if np.random.randint(0, self.num_seen + i) < len(self.loss):
                    self.memory_x[i] = x[i]
                    self.memory_y[i] = y[i]
                    self.loss[id(x[i])] = min(self.loss[id(x[i])], loss[i])
            self.num_seen += 1

    def sample(self):
        if len(self.loss) == 0:
            return None # Empty Memory, Indicates first task
        idx = torch.randperm(self.size)[:self.C]
        loss_keys = [id(x) for x in self.memory_x[idx]]
        loss_idx = list(itemgetter(*loss_keys)(self.loss))
        return self.memory_x[idx], self.memory_y[idx], loss_idx

    def reset(self):
        self.memory = np.zeros(self.size)
        self.best_loss = defaultdict(np.inf)
        self.num_seen = 0
