import torch
import numpy as np
import os, glob
import pickle as pkl
import cv2
import imageio
from PIL import Image
import tqdm

from data.postprocessing_functions import SimplePostProcess

from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004


def to_one_channel(image, pattern='RGGB'):
    """
    Convert 4-channel raw to single channel raw.
    """
    if pattern != 'RGGB':
        raise NotImplementedError
    
    if pattern == 'RGGB':
        h, w, _ = image.shape
        im_ = np.empty((2*h, 2*w), dtype=np.uint16)
        im_[0::2, 0::2] = image[:, :, 0]
        im_[0::2, 1::2] = image[:, :, 1]
        im_[1::2, 0::2] = image[:, :, 2]
        im_[1::2, 1::2] = image[:, :, 3]
        return im_


pp = SimplePostProcess(return_np=True)

linear_folder_dir = '/data/gjh8760/datasets/SyntheticBurstVal'
srgb_folder_dir = '/data/gjh8760/datasets/SyntheticBurstVal_srgb_temp'

subfolders = sorted(os.listdir(os.path.join(linear_folder_dir, 'gt')))

for subfolder in subfolders:
    gt_folder_dir = os.path.join(linear_folder_dir, 'gt', subfolder)
    meta_data = pkl.load(open('{}/meta_info.pkl'.format(gt_folder_dir), "rb", -1))
    burst_folder_dir = os.path.join(linear_folder_dir, 'bursts', subfolder)
    
    gt_img_path = os.path.join(gt_folder_dir, 'im_rgb.png')
    burst_img_paths = sorted(glob.glob(os.path.join(burst_folder_dir, '*.png')))
    
    # Burst
    for burst_img_path in burst_img_paths:
        dst_folder_dir = os.path.join(srgb_folder_dir, 'bursts', subfolder)
        if not os.path.exists(dst_folder_dir):
            os.makedirs(dst_folder_dir)
        
        im = cv2.imread(burst_img_path, cv2.IMREAD_UNCHANGED)   # dtype: np.uint16
        # Demosaic to linear
        im_1 = to_one_channel(im, 'RGGB')
        im_d = demosaicing_CFA_Bayer_Malvar2004(im_1, 'RGGB')
        im_d = np.clip(im_d, 0, im_d.max())
        # linear to srgb
        im_t = (torch.from_numpy(im_d.astype(np.float32)) / 2 ** 14).permute(2, 0, 1).float()
        res = pp.process(im_t, meta_data)
        # imageio.imwrite('{}/{}'.format(dst_folder_dir, os.path.basename(burst_img_path)), res)
        cv2.imwrite('{}/{}'.format(dst_folder_dir, os.path.basename(burst_img_path)), res)

    # gt
    dst_folder_dir = os.path.join(srgb_folder_dir, 'gt', subfolder)
    if not os.path.exists(dst_folder_dir):
        os.makedirs(dst_folder_dir)
    
    gt_img_path = glob.glob(os.path.join(gt_folder_dir, '*.png'))[0]
    im = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)   # linear
    # linear to srgb
    im_t = (torch.from_numpy(im.astype(np.float32)) / 2 ** 14).permute(2, 0, 1).float()
    res = pp.process(im_t, meta_data)
    # imageio.imwrite('{}/{}'.format(dst_folder_dir, os.path.basename(gt_img_path)), res)
    cv2.imwrite('{}/{}'.format(dst_folder_dir, os.path.basename(gt_img_path)), res)

    print('{} Done'.format(subfolder))