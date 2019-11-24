#!/usr/bin/env python3

from joblib import Parallel, delayed
import queue as Queue
import os

# Define number of GPUs available
seeds = [465, 387,  53, 121, 999, 848, 960, 193,
         996, 341, 142, 657, 712, 674,  98,  25,
         349, 664, 365, 783]

# Put indices in queue
def runner(i):
    # Put here your job cmd
    seed = seeds[i]
    exp_name = f'permute-er-str{seed}'
    args = ['SYSTEM.SEED', seed, 'SYSTEM.EXP_NAME', exp_name]
    cmd = f'python src/solver_er.py --clean_run True --config_file ./configs/mnist-permute/experience_replay.yaml {args[0]} {args[1]} {args[2]} {args[3]}'
    print(cmd)
    os.system(cmd)

# Change loop
Parallel(n_jobs=1, backend="threading")(
    delayed(runner)(i) for i in range(len(seeds)))
