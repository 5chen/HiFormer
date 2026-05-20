"""
Testing script for HiFormer
"""

import argparse
import subprocess
from tqdm import tqdm
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import TestDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.hiformer import HiFormer


class HiFormerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = HiFormer(
            inp_channels=3,
            out_channels=3,
            dim=24,
            num_blocks=[4, 4, 8],
            heads=[1, 2, 4],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            window_size=4
        )
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x):
        return self.net(x)

def load_checkpoint(model, ckpt_path):
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--valid_data_dir', type=str, default="data/Test/UHD_haze/", 
                        help='save path of test images')
    parser.add_argument('--output_path', type=str, default="output/", 
                        help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/hiformer/hiformer-epoch-499.ckpt", 
                        help='checkpoint path')
    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = testopt.ckpt_path
    print("CKPT name : {}".format(ckpt_path))

    # Load model
    net = load_checkpoint(HiFormerModel(), ckpt_path).cuda()
    net.eval()

    data_path = testopt.valid_data_dir
    dataset_name = data_path.split('/')[-2]
    print(f'Test: {dataset_name}')
    data_set = TestDataset(testopt)
    
    output_path = testopt.output_path + dataset_name + '/'
    testloader = DataLoader(data_set, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

    psnr = AverageMeter()
    ssim = AverageMeter()
    with torch.no_grad():
        for (degraded_name, degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
        print("PSNR: %.4f, SSIM: %.4f" % (psnr.avg, ssim.avg))

