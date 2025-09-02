# Configuration file for background removal inference

import os

# Dictionary containing output folder names as keys and model paths as values
MODEL_CONFIGS = {
    "model_output/ft_best": "trained_weights/bg_15k_best.pth",
    "model_output/bria": "trained_weights/bria.pth",
    "model_output/ft_58": "trained_weights/bg_15k_58.pth",
    "model_output/ft_12": "trained_weights/bg_15k_12.pth",
}




INPUT_DATASET_PATH = "../test_images"

# Model input size
INPUT_SIZE = [1024, 1024]

# Static threshold value (simplified)
THRESHOLD = 150

# Image extensions to process
IMAGE_EXTENSIONS = [
    "*.jpg", "*.JPG", "*.jpeg", "*.JPEG",
    "*.png", "*.PNG"
]

# GCP Storage Configuration
GS_BUCKET_PATH = "gs://retouching-ai/Background Removal/MODNet_inference_result/"

# GSync options
GSUTIL_RSYNC_OPTIONS = [
    "-m",  # Multi-threaded
    "-r",  # Recursive
    "-d",  # Delete files in destination that are not in source
    "-x", ".*\\.tmp$|.*\\.temp$"  # Exclude temporary files
]

# Inference output base directory (under <repo>/inference_output)
INFERENCE_OUTPUT_DIR = "../inference_output"

# Default epochs to run inference on (checkpoint files must exist)
EPOCHS = [5, 10, 15, 20]

# Model checkpoint directory and filename prefix used to compose paths like:
#   {MODEL_SAVE_DIR}/{MODEL_PREFIX}_epoch_{E}.pth
# Example: "../saved_models_simple_001/modnet_test/modnet_epoch_10.pth"
# MODEL_SAVE_DIR = "../saved_models/isnet_test"
# MODEL_PREFIX = "main_model"

# Inference enhancements
# Multi-scale ensemble: run at multiple scales, upsample to original, and average
MULTI_SCALE_ENABLED = True
MULTI_SCALE_SCALES = [0.75, 1.0, 1.25]

# Test-time augmentation: horizontal flip
TTA_HFLIP_ENABLED = True

# ───────────────────────────────
# Trimap generation parameters
# ───────────────────────────────

# Generic defaults (fallbacks)
TRIMAP_BG_THRESHOLD = 0.7
TRIMAP_FG_THRESHOLD = 0.8
TRIMAP_KERNEL_SIZE = 9
TRIMAP_UNKNOWN_DILATE_ITERS = 5

# ISNet-specific trimap settings
TRIMAP_BG_THRESHOLD_ISNET = 0.7
TRIMAP_FG_THRESHOLD_ISNET = 0.8
TRIMAP_KERNEL_SIZE_ISNET = 9
TRIMAP_UNKNOWN_DILATE_ITERS_ISNET = 5

# MODNet-specific trimap settings
TRIMAP_BG_THRESHOLD_MODNET = 0.7
TRIMAP_FG_THRESHOLD_MODNET = 0.8
TRIMAP_KERNEL_SIZE_MODNET = 9
TRIMAP_UNKNOWN_DILATE_ITERS_MODNET = 5

FBA_MODEL_DIR = '../model_weights'
