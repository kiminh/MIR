#!/usr/bin/env bash

python src/solver_er_mir.py --clean_run True --config_file ./configs/mnist-permute/experience_replay_mir.yaml SOLVER.SAMPLING_CRITERION 2
