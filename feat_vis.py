
# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from dataset.synthetic_burst_val_set import SyntheticBurstVal
import torch

from models.loss.image_quality_v2 import PSNR, SSIM, LPIPS
from evaluation.common_utils.display_utils import generate_formatted_report
import time
import argparse
import importlib
import cv2
import numpy as np
import tqdm
import random
import imageio

from data.postprocessing_functions import SimplePostProcess

def setup_seed(seed=0):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	# torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    from models.RBSR_test import RBSR
    
    net = RBSR(align_type=None)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    setup_seed(seed=0)

    device = 'cuda'
    model_path = './pretrained_networks/RBSR_best_ep0400.pth.tar'
    checkpoint_dict = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint_dict['net'])
    net = net.to(device).train(False)

    dataset = SyntheticBurstVal()

    for idx in range(len(dataset)):
        burst, gt, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']

        burst = burst.to(device).unsqueeze(0)
        gt = gt.to(device)

        with torch.no_grad():
            burst = burst[:, :14, ...]
            net_pred, _ = net(burst)
            print()
    