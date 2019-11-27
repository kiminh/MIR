#!/usr/bin/env python3
import numpy as np


def to_np(x):
    return x.cpu().numpy()


def get_counts(mem):
    y = [y for x, y in mem]
    y = np.array(y)
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def get_counts_labels(y):
    y = y.cpu().numpy()
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))
