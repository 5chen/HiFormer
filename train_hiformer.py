"""
Training script for HiFormer
Based on PyTorch Lightning
"""

import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset
from net.hiformer import HiFormer
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options_hiformer import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch.fft
import os
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'


class HiFormerModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
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
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        (degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        
        # Logging to TensorBoard/WandB
        self.log("train_loss", loss, prog_bar=True)
        
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr)
        
        # Dynamic warmup and max epochs based on total epochs
        warm_up = int(self.opt.epochs * 0.15)  # 15% warmup
        max_e = int(self.opt.epochs * 0.75)     # 75% for cosine annealing
        
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=warm_up,
            max_epochs=max_e
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1            
            }
        }


def main():
    print("=" * 50)
    print("HiFormer Training Configuration")
    print("=" * 50)
    print(opt)
    print("=" * 50)
    
    # Setup logger
    if opt.wblogger is not None:
        logger = WandbLogger(
            project=opt.wblogger,
            name="HiFormer-Train",
            # mode="offline"  # uncomment for offline mode
        )
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    # Setup dataset
    trainset = TrainDataset(opt)
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )
    
    print(f"\nDataset size: {len(trainset)}")
    print(f"Number of batches per epoch: {len(trainloader)}")
    print(f"Total training steps: {len(trainloader) * opt.epochs}\n")
    
    # Setup model
    model = HiFormerModel(opt)
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        every_n_epochs=5,
        save_top_k=-1,
        filename='hiformer-{epoch:03d}'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp" if opt.num_gpus > 1 else "auto",
        logger=logger,
        callbacks=[checkpoint_callback],
        precision='16-mixed' if opt.use_amp else 32,  # Mixed precision training
        gradient_clip_val=0.01 if opt.gradient_clip else 0,
        log_every_n_steps=50
    )
    
    # Start training
    print("\nStarting training...")
    trainer.fit(model=model, train_dataloaders=trainloader)
    
    print("\nTraining completed!")
    print(f"Checkpoints saved to: {opt.ckpt_dir}")


if __name__ == '__main__':
    main()

