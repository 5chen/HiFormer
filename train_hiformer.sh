#!/bin/bash

# Training script for HiFormer
# Ultra-High-Definition Image Restoration
# Usage: bash train_hiformer.sh

echo "Starting HiFormer Training..."
echo "=============================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1  # Modify based on available GPUs

# Training configuration
EPOCHS=500
BATCH_SIZE=8
PATCH_SIZE=768
NUM_GPUS=2
NUM_WORKERS=16
LR=0.0002


# Paths (modify according to your data location)
DATA_DIR="data_dir/"

# Checkpoint directory
CKPT_DIR="ckpt/hiformer/"
mkdir -p ${CKPT_DIR}

# WandB project name (set to "none" to disable wandb)
WBLOGGER="hiformer"

# Run training
python train_hiformer.py \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --patch_size ${PATCH_SIZE} \
    --num_gpus ${NUM_GPUS} \
    --num_workers ${NUM_WORKERS} \
    --lr ${LR} \
    --de_type ${DE_TYPES} \
    --data_file_dir ${DATA_DIR} \
    --ckpt_dir ${CKPT_DIR} \
    --wblogger ${WBLOGGER} \
    --use_amp \
    --gradient_clip

echo ""
echo "=============================="
echo "Training completed!"
echo "Checkpoints saved to: ${CKPT_DIR}"

