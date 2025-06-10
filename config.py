#config.py
"""
Configuration settings for the Iterative Crowd Counting Model.
"""
import os
import torch

# --- Dataset Paths ---
# Adjust these paths based on your environment
BASE_DATA_DIR = "C:\\Users\\Mehmet_Postdoc\\Desktop\\datasets_for_experiments\\ShanghaiTech_Crowd_Counting_Dataset" # Base directory after unzipping
IMAGE_DIR_TRAIN_VAL = os.path.join(BASE_DATA_DIR, "part_A_final\\train_data\\images")
GT_DIR_TRAIN_VAL = os.path.join(BASE_DATA_DIR, "part_A_final\\train_data\\ground_truth")
IMAGE_DIR_TEST = os.path.join(BASE_DATA_DIR, "part_A_final\\test_data\\images")
GT_DIR_TEST = os.path.join(BASE_DATA_DIR, "part_A_final\\test_data\\ground_truth")
OUTPUT_DIR = "C:\\Users\\Mehmet_Postdoc\\Desktop\\python_set_up_code\\iterative_crowd_counting_with_confidence\\crowd_counting_outputs" # For logs, plots, models

# --- Data Preprocessing ---
AUGMENTATION_SIZE = 256 # Intermediate size during augmentation before final crop (e.g., 256 for 224 input)
MODEL_INPUT_SIZE = 224 # Final input size to the model (after center crop) - Changed to 224
MIN_DIM_RESCALE = 256 # Minimum dimension allowed after random scaling (should be >= AUGMENTATION_SIZE) - Adjusted
GT_PSF_SIGMA = 1      # Sigma for Gaussian kernel to generate GT PSFs

# --- Training ---
TOTAL_ITERATIONS = 100000
BATCH_SIZE = 8 # You might need to reduce batch size if 224x224 images cause memory issues
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VALIDATION_INTERVAL = 100 # How often to run validation
VALIDATION_BATCHES = 50 # Number of batches for validation evaluation
SCHEDULER_PATIENCE = 10 # ReduceLROnPlateau patience (in terms of validation intervals)
SEED = 42

# --- Model ---
PSF_HEAD_TEMP = 0.1 # Temperature for softmax in PSFHead

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Output File Paths ---
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "training_log.txt")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_fpn.pth")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Model input size set to: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
print(f"Augmentation size set to: {AUGMENTATION_SIZE}x{AUGMENTATION_SIZE}")