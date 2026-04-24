"""
Configuration settings for Brain Tumor Detection System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model configuration
IMG_WIDTH = 224
IMG_HEIGHT = 224
CHANNELS = 3
BATCH_SIZE = 16  # Reduced for stability
EPOCHS = 30  # Reduced for faster testing
LEARNING_RATE = 1e-4

# Model checkpoint path
BEST_MODEL_PATH = MODELS_DIR / "brain_tumor_model_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "brain_tumor_model_final.keras"
HISTORY_PATH = MODELS_DIR / "training_history.pkl"

# Class labels
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)

# Data augmentation parameters
AUGMENTATION = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "horizontal_flip": True,
    "vertical_flip": True,
    "zoom_range": 0.2,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest"
}

# Training parameters
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 10  # Reduced for faster stopping
REDUCE_LR_PATIENCE = 3  # Reduced for more aggressive LR reduction
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Threshold for confidence
CONFIDENCE_THRESHOLD = 0.60
