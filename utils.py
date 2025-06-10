# utils.py
"""
Utility functions for the Iterative Crowd Counting Model.
"""
import random
import numpy as np
import torch
import os
import glob

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
        # Optional: Might slow down training but ensures reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    #print(f"Seed set to {seed}")

def find_and_sort_paths(directory, pattern):
    """Finds and sorts paths matching a pattern in a directory."""
    paths = sorted(glob.glob(os.path.join(directory, '**', pattern), recursive=True))
    if not paths:
        print(f"Warning: No files found for pattern '{pattern}' in directory '{directory}'")
    return paths

def split_train_val(sorted_image_paths, sorted_gt_paths, val_ratio=0.1, seed=42):
    """Splits sorted image and ground truth paths into training and validation sets."""
    assert len(sorted_image_paths) == len(sorted_gt_paths), "Image and GT paths must have the same length."
    assert len(sorted_image_paths) > 0, "Input path lists are empty."

    local_random = random.Random(seed) # Use a local Random instance
    indices = list(range(len(sorted_image_paths)))
    local_random.shuffle(indices)

    num_val = int(len(indices) * val_ratio)
    if num_val == 0 and len(indices) > 0:
         print(f"Warning: Validation set size is zero with ratio {val_ratio} and {len(indices)} samples. Adjust val_ratio or dataset size.")
         # Handle small datasets: ensure at least one validation sample if possible
         num_val = 1 if len(indices) > 1 else 0

    train_indices = indices[num_val:]
    val_indices = indices[:num_val]

    train_image_paths = [sorted_image_paths[i] for i in train_indices]
    train_gt_paths = [sorted_gt_paths[i] for i in train_indices]
    val_image_paths = [sorted_image_paths[i] for i in val_indices]
    val_gt_paths = [sorted_gt_paths[i] for i in val_indices]

    print(f"Data split: {len(train_image_paths)} train, {len(val_image_paths)} validation samples.")
    return train_image_paths, train_gt_paths, val_image_paths, val_gt_paths

def generate_anchor_grid_centers_pixel(feature_map_height, feature_map_width, fpn_output_stride):
    """
    Generates anchor center pixel coordinates for a given feature map size and stride.
    The coordinates are relative to the model input image space.
    Args:
        feature_map_height (int): Height of the feature map.
        feature_map_width (int): Width of the feature map.
        fpn_output_stride (int): The stride of this feature map relative to the input image.
    Returns:
        torch.Tensor: Anchor coordinates (N_anchors, 2) where N_anchors = H_feat * W_feat.
                      Coordinates are (x_pixel, y_pixel).
    """
    anchors = []
    for j in range(feature_map_height): # y_grid_idx (row index)
        for i in range(feature_map_width): # x_grid_idx (column index)
            # Center of the grid cell in pixel coordinates on the input image
            # (i + 0.5) is the center of the i-th cell in grid coordinates
            # fpn_output_stride maps grid coordinates to input image pixel coordinates
            pixel_x = (i + 0.5) * fpn_output_stride
            pixel_y = (j + 0.5) * fpn_output_stride
            anchors.append([pixel_x, pixel_y])
    return torch.tensor(anchors, dtype=torch.float32)