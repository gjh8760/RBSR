import os
import sys
import argparse
import importlib
import multiprocessing
import cv2 as cv
import torch.backends.cudnn

import numpy as np
import random

def setup_seed(seed=0):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	# torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

env_path = os.path.join(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)

import admin.settings as ws_settings


def run_training(train_module, train_name, exp_name, align_type, cudnn_benchmark=True):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        exp_name: Name of this experiment.
        align_type: flow_guided_dcn_alignment, flow_alignment, dcn_alignment. Default is flow_guided_dcn_alignment.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    setup_seed(0)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    print('Training:  {}  {}  {}'.format(train_module, train_name, exp_name))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = '{}/{}/{}'.format(train_module, train_name, exp_name)
    settings.align_type = align_type

    expr_module = importlib.import_module('train_settings.{}.{}'.format(train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('exp_name', type=str, help='Name of this experiment.')
    # parser.add_argument('align_type', type=str, default='flow_guided_dcn_alignment', help='Feature alignment type.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')

    args = parser.parse_args()

    run_training(args.train_module, args.train_name, args.exp_name, args.cudnn_benchmark)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
