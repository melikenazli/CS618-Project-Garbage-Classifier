from pathlib import Path

# Paths
SPLIT_DIR   = Path("data/split")   # expects train/ val/ test/ inside
RESULTS_DIR = Path("results")

# Training
IMG_SIZE    = 256
BATCH_SIZE  = 32
EPOCHS      = 10    # During experimentation for resnet50=10, baseline=40
LR          = 1e-3
NUM_WORKERS = 4
SEED        = 42

# Model choice: "baseline", "resnet50"
MODEL_NAME  = "resnet50"
