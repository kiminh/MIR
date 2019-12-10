#!/usr/bin/env python
from joblib import Parallel, delayed
import Queue
import os

# Define number of GPUs available
N_GPU = 6

# Put indices in queue
q = Queue.Queue(maxsize=N_GPU)
for i in range(N_GPU):
    q.put(i)

def runner(x):
    gpu = q.get()
    print (x, gpu)

    # Put here your job cmd
    cmd = "python main.py %s" % x
    os.system("CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd))

    # return gpu id to queue
    q.put(gpu)

# Change loop
Parallel(n_jobs=N_GPU, backend="threading")(
    delayed(runner)(i) for i in range(100))
