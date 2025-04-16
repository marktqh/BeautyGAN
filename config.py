"""
Configuration parameters for face beautification model.
"""
import os

# Paths
DATA_DIR = os.environ.get("DATA_DIR", "/home/ubuntu/data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
AFT_DIR = os.path.join(DATA_DIR, "aft")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/home/ubuntu/results")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/home/ubuntu/model_progress")
LOG_DIR = os.environ.get("LOG_DIR", "/home/ubuntu/logs")

# Training parameters
BATCH_SIZE = 80
BATCH_SIZE_TEST = 1
LEARNING_RATE = 6e-4
NUM_EPOCHS = 800
SEED = 42

# Logging and saving
LOG_INTERVAL = 100
SAMPLE_INTERVAL = 250
SAVE_INTERVAL = 1000
LR_DECAY_STEPS = 2000

# Model parameters
IMG_SIZE = 256
TRANSFORMER_BLOCKS = 3
TRANSFORMER_HEADS = 8
FFN_RATIO = 4.0
DROPOUT = 0.1
INIT_ALPHA = 0.5  # Initial alpha value for DyT
WEIGHT_DECAY = 1e-4  # Weight decay for regularization

# Loss weights
ADVERSARIAL_WEIGHT = 1.0  # Weight for GAN loss
RECONSTRUCTION_WEIGHT = 10.0  # Weight for L1 loss
PERCEPTUAL_WEIGHT = 0.5  # Weight for perceptual loss
TV_WEIGHT = 0.1  # Weight for total variation loss

# Optimizer parameters
BETA1 = 0.5  # Beta1 for Adam optimizer
BETA2 = 0.999  # Beta2 for Adam optimizer

# Distributed training
MASTER_ADDR = "localhost"
MASTER_PORT = "12355"

def update_config(opts=None):
    """Update config parameters from command line options."""
    if opts is None:
        return
    
    global DATA_DIR, RAW_DIR, AFT_DIR, RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR
    global BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
    global LOG_INTERVAL, SAMPLE_INTERVAL, SAVE_INTERVAL
    
    # Example: Handle command line arguments that modify config
    # This function can be expanded based on your needs
    if hasattr(opts, 'data_dir') and opts.data_dir:
        DATA_DIR = opts.data_dir
        RAW_DIR = os.path.join(DATA_DIR, "raw")
        AFT_DIR = os.path.join(DATA_DIR, "aft")
    
    if hasattr(opts, 'batch_size') and opts.batch_size:
        BATCH_SIZE = opts.batch_size
    
    if hasattr(opts, 'lr') and opts.lr:
        LEARNING_RATE = opts.lr
