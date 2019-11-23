#!/usr/bin/env python3
import numpy as np

class Metrics:
    def __init__(self, num_tasks):
        self.acc = np.zeros((num_tasks, num_tasks))
        self.forget = np.zeros((num_tasks, num_tasks))

    def accuracy(self, tid, tid_done, pred, y):
        self.acc[tid_done, tid]+=pred.eq(y.view_as(pred)).sum().item()

    def avg_accuracy(self, tid, tid_done, val):
        # print(f'tid_done:{tid_done}, tid:{tid}, val:{val}')
        self.acc[tid_done, tid] /= val

    def acc_task(self, tid_done):
        return np.sum(self.acc[tid_done, :]) / (tid_done+1)

    def forgetting(self, tid, tid_done):
        if tid_done > 0 and not tid == tid_done:
            self.forget[tid_done][tid] = np.max(self.acc[:tid_done, tid]) - self.acc[tid_done, tid]
        else:
            self.forget[tid_done][tid] = 0

    def forgetting_task(self, tid_done):
        if tid_done > 0:
            return np.sum(self.forget[tid_done, :tid_done]) / tid_done
        else:
            return 0


if __name__ == "__main__":
    metrics = Metrics(4)
    y = np.array([1, 0, 2, 4])
    pred = np.array([2, 0, 2, 4])
    tid_done = 3
    for tid in range(4):
        metrics.accuracy(tid, 3, pred, y)
    print(metrics.acc)
