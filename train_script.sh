#!/bin/bash

# Train the face beautification GAN using distributed training on all available GPUs
# Usage: bash scripts/train_distributed.sh

# Get the number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Check if GPUs are available
if [ $NUM_GPUS -le 0 ]; then
    echo "Error: No GPUs found. This model requires at least one GPU for training."
    exit 1
fi

echo "Found $NUM_GPUS GPUs. Starting distributed training..."

# Create directories if they don't exist
mkdir -p /home/ubuntu/results
mkdir -p /home/ubuntu/model_progress
mkdir -p /home/ubuntu/logs

# Set environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Launch training with torchrun
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train.py --resume
