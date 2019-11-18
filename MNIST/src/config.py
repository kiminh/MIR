#!/usr/bin/env python

from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.SEED = 43
_C.SYSTEM.SAVE_DIR = './experiments/'
_C.SYSTEM.EXP_NAME = 'default'
_C.SYSTEM.LOG_FREQ = 100


# Dataset
_C.DATA = CN()
_C.DATA.ROOT = './torch_data/mnist/'
_C.DATA.SAVE_FILE = './torch_data/mnist/split/'
_C.DATA.NUM_CLASSES = 10
_C.DATA.SHAPE = (28, 28)
_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.ROOT = './torch_data/mnist/split/train.pt'
_C.DATA.TRAIN.NUM_SAMPLES = 1000
_C.DATA.TEST = CN()
_C.DATA.TEST.ROOT = './torch_data/mnist/split/test.pt'



# Model
_C.MODEL = CN()
_C.MODEL.MLP = CN()
_C.MODEL.MLP.INPUT_SIZE = 784
_C.MODEL.MLP.OUTPUT_SIZE = 10
_C.MODEL.MLP.HIDDEN_SIZE = 400
# Solver
_C.SOLVER = CN()
_C.SOLVER.NUM_TASKS = 5
_C.SOLVER.NUM_EPOCHS = 1
_C.SOLVER.BATCH_SIZE = 1000
_C.SOLVER.SAMPLING_CRITERION = 2
_C.SOLVER.BUDGET = 50
# Optimizer
_C.OPTIMIZER = CN()
_C.OPTIMIZER.MOMENTUM = 0.0
_C.OPTIMIZER.LR = 0.005
# Buffer
_C.BUFFER = CN()
_C.BUFFER.SIZE = 50
_C.BUFFER.C = 50

cfg = _C
