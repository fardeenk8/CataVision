"""
Configuration for Cataract Detection System.
Paths, hyperparameters, and image size.
"""
import os
from pathlib import Path

# --- Paths (relative to project root) ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Roboflow OculaCare (naked-eye) is now the only supported dataset.
# Expect Folder Structure export with train/valid/test under this root.
RAW_ROBOFLOW_DIR = DATA_DIR / "raw_roboflow"  # put OculaCare.v1i.folder here
SPLITS_ROBOFLOW_PATH = DATA_DIR / "splits_roboflow.json"  # output from prepare_roboflow_binary.py

# Primary splits path used by training/evaluation scripts (points to Roboflow splits).
SPLITS_PATH = SPLITS_ROBOFLOW_PATH

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Image ---
IMAGE_SIZE = 224  # for both baseline and EfficientNet (EfficientNet expects 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet
IMAGE_STD = [0.229, 0.224, 0.225]

# --- Training ---
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20  # capped for laptop; early stopping (patience=7) usually stops sooner
PATIENCE = 7  # early stopping
SEED = 42

# EfficientNet: freeze backbone for this many epochs, then fine-tune
FREEZE_EPOCHS = 5

# --- Model checkpoints (trained on Roboflow OculaCare) ---
BASELINE_CKPT = CHECKPOINTS_DIR / "baseline_roboflow_best.pt"
EFFICIENTNET_CKPT = CHECKPOINTS_DIR / "efficientnet_roboflow_best.pt"

# --- Evaluation output ---
EVAL_OUTPUT_DIR = RESULTS_DIR / "evaluation"

def ensure_dirs():
    """Create necessary directories if they do not exist."""
    for d in (CHECKPOINTS_DIR, RESULTS_DIR, EVAL_OUTPUT_DIR, RAW_ROBOFLOW_DIR):
        d.mkdir(parents=True, exist_ok=True)
