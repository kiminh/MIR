#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.log_softmax(self.layer2(x), dim=1)
        return x
