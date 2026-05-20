"""
Configuration options for HiFormer training
Ultra-High-Definition Image Restoration via High Frequency Enhanced Transformer
"""

import argparse

parser = argparse.ArgumentParser(description='HiFormer Training Options')

# Training Parameters
parser.add_argument('--cuda', type=int, default=0, help='CUDA device number')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch size per GPU')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--num_workers', type=int, default=16, help='number of data loading workers')
parser.add_argument('--num_gpus', type=int, default=2, help='number of GPUs to use for training')

# Model Parameters
parser.add_argument('--patch_size', type=int, default=768, help='patch size of input images')


# Data Paths
parser.add_argument('--data_file_dir', type=str, default='data_dir/', 
                    help='directory containing data file lists')

# Output Paths
parser.add_argument('--output_path', type=str, default='output/', 
                    help='output save path')
parser.add_argument('--ckpt_dir', type=str, default='ckpt/hiformer/', 
                    help='checkpoint save directory')

# Logging
parser.add_argument('--wblogger', type=str, default='hiformer', 
                    help='wandb project name (None to disable wandb)')

# Training Options
parser.add_argument('--use_amp', action='store_true', 
                    help='use automatic mixed precision training')
parser.add_argument('--gradient_clip', action='store_true',
                    help='enable gradient clipping')
parser.add_argument('--resume_from', type=str, default=None,
                    help='path to checkpoint to resume from')

# Testing
parser.add_argument('--test_only', action='store_true',
                    help='only run testing')
parser.add_argument('--test_path', type=str, default='test/',
                    help='path to test images')

options = parser.parse_args()

