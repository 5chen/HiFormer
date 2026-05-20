#!/bin/bash

# Testing script for HiFormer
# Ultra-High-Definition Image Restoration
# Usage: bash test_hiformer.sh

echo "Starting HiFormer Testing..."
echo "=============================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Testing configuration
CUDA_ID=0
MODE=2  # 0: denoise, 1: derain, 2: dehaze (UHD), 3: all

# Checkpoint path
CKPT_NAME="hiformer-epoch-499.ckpt"

# Test data paths
DATA_DIR="test/denoise/"
OUTPUT_PATH="output/"

# Create output directory
mkdir -p ${OUTPUT_PATH}

# Run testing
python test_hiformer.py \
    --cuda ${CUDA_ID} \
    --mode ${MODE} \
    --valid_data_dir ${DATA_DIR} \
    --output_path ${OUTPUT_PATH} \
    --ckpt_name ${CKPT_NAME}

echo ""
echo "=============================="
echo "Testing completed!"
echo "Results saved to: ${OUTPUT_PATH}"

