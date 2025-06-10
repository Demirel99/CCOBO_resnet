# dataset.py
"""
Dataset preparation, augmentation, and sample generation functions.
"""
import numpy as np
import cv2
import random
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import os
import glob
import matplotlib.pyplot as plt

from config import (AUGMENTATION_SIZE, MODEL_INPUT_SIZE, MIN_DIM_RESCALE, GT_PSF_SIGMA, IMAGE_DIR_TRAIN_VAL, GT_DIR_TRAIN_VAL, DEVICE)

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def prepare_data_augmentations(image, gt_coor, target_size=AUGMENTATION_SIZE, min_dim=MIN_DIM_RESCALE,
                               skew_patch_to_top=False, top_patch_bias_range_percentage=0.25):
    if image is None:
        print("Warning: prepare_data_augmentations received None image.")
        return None, None
    h, w, c = image.shape
    if c != 3:
        print(f"Warning: Expected a 3-channel image (H, W, 3), but got {c} channels. Attempting to proceed.")

    scale_factor = random.uniform(0.7, 1.3)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    if min(h, w) > 0 and min(new_h, new_w) < min_dim:
        scale_factor = min_dim / min(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    elif min(h, w) == 0:
        print("Warning: Image has zero dimension before scaling.")
        return None, None
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    try:
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except cv2.error as e:
        print(f"Error during cv2.resize: {e}. Original shape: {(h,w)}, Target shape: {(new_h, new_w)}")
        return None, None
    scaled_gt_coor = gt_coor * scale_factor if gt_coor is not None and gt_coor.size > 0 else np.array([])

    curr_h, curr_w = scaled_image.shape[:2]
    crop_h = min(target_size, curr_h)
    crop_w = min(target_size, curr_w)
    cropped_gt_coor = np.array([])

    if curr_h >= target_size and curr_w >= target_size:
        max_x_start = curr_w - target_size
        max_y_start = curr_h - target_size
        x_min = random.randint(0, max_x_start)
        
        if skew_patch_to_top and max_y_start > 0:
            # Bias y_min to the top part of the image
            upper_bound_y_start = max(0, int(max_y_start * top_patch_bias_range_percentage))
            y_min = random.randint(0, upper_bound_y_start)
        else:
            y_min = random.randint(0, max_y_start)
        cropped_image = scaled_image[y_min : y_min + target_size, x_min : x_min + target_size]
        if scaled_gt_coor.size > 0:
            keep_mask = (scaled_gt_coor[:, 0] >= x_min) & (scaled_gt_coor[:, 0] < x_min + target_size) & \
                        (scaled_gt_coor[:, 1] >= y_min) & (scaled_gt_coor[:, 1] < y_min + target_size)
            cropped_gt_coor = scaled_gt_coor[keep_mask]
            if cropped_gt_coor.size > 0:
                cropped_gt_coor[:, 0] -= x_min
                cropped_gt_coor[:, 1] -= y_min
        final_h, final_w = target_size, target_size
    else:
        print(f"Warning: Scaled image ({curr_h}x{curr_w}) smaller than target ({target_size}x{target_size}). Resizing up.")
        try:
             cropped_image = cv2.resize(scaled_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
             print(f"Error during cv2.resize (upscaling): {e}. Scaled shape: {(curr_h, curr_w)}, Target shape: {(target_size, target_size)}")
             return None, None
        if curr_w > 0 and curr_h > 0 and scaled_gt_coor.size > 0:
            scale_x_final = target_size / float(curr_w)
            scale_y_final = target_size / float(curr_h)
            cropped_gt_coor = scaled_gt_coor.copy()
            cropped_gt_coor[:, 0] *= scale_x_final
            cropped_gt_coor[:, 1] *= scale_y_final
            cropped_gt_coor[:, 0] = np.clip(cropped_gt_coor[:, 0], 0, target_size - 1)
            cropped_gt_coor[:, 1] = np.clip(cropped_gt_coor[:, 1], 0, target_size - 1)
        final_h, final_w = target_size, target_size

    if random.random() < 0.5:
        cropped_image = cv2.flip(cropped_image, 1)
        if cropped_gt_coor.size > 0:
            cropped_gt_coor[:, 0] = (final_w - 1) - cropped_gt_coor[:, 0]

    if cropped_image.shape[0] != target_size or cropped_image.shape[1] != target_size:
         print(f"Warning: Final augmented image shape {cropped_image.shape} does not match target {target_size}x{target_size}. Resizing again.")
         try:
            cropped_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
         except cv2.error as e:
            print(f"Error during final cv2.resize: {e}. Shape before: {cropped_image.shape}, Target shape: {(target_size, target_size)}")
            return None, None
    return cropped_image, cropped_gt_coor

def generate_single_psf(coord, image_shape, sigma):
    height, width = image_shape
    psf = np.zeros((height, width), dtype=np.float32)
    x = np.clip(int(round(coord[0])), 0, width - 1)
    y = np.clip(int(round(coord[1])), 0, height - 1)
    psf[y, x] = 1.0
    psf = gaussian_filter(psf, sigma=sigma, order=0, mode='constant', cval=0.0)
    psf_sum = np.sum(psf)
    if psf_sum > 1e-7:
        psf /= psf_sum
    return psf

def get_center_crop_coords(image_size, crop_size):
    img_h, img_w = image_size
    crop_h, crop_w = crop_size
    start_y = max(0, (img_h - crop_h) // 2)
    start_x = max(0, (img_w - crop_w) // 2)
    return start_y, start_x

def get_random_coord_index_in_center(coordinates, image_shape, center_crop_shape, skew_to_top=False): # Added skew_to_top
    if coordinates is None or coordinates.shape[0] == 0: return None
    img_h, img_w = image_shape
    crop_h, crop_w = center_crop_shape
    start_y, start_x = get_center_crop_coords(image_shape, center_crop_shape)
    end_y, end_x = start_y + crop_h, start_x + crop_w
    indices_in_center = [
        i for i, (x, y) in enumerate(coordinates)
        if start_x <= x < end_x and start_y <= y < end_y
    ]
    if not indices_in_center:
        return None
    
    if skew_to_top:
        # Create a list of (index_in_coordinates, y_coordinate) for candidates
        candidate_targets_with_y = []
        for idx in indices_in_center:
            # coordinates[idx] is (x,y), so coordinates[idx][1] is the y-coordinate
            candidate_targets_with_y.append((idx, coordinates[idx][1]))
        
        # Sort candidates by y-coordinate (ascending, so top-most first)
        sorted_candidate_targets = sorted(candidate_targets_with_y, key=lambda item: item[1])
        
        num_candidates = len(sorted_candidate_targets)
        if num_candidates == 1:
            return sorted_candidate_targets[0][0] # Return the only candidate's index
            
        # Generate weights: higher weight for smaller rank (top-most points)
        # Using 1/(rank+1) as weights. Can be adjusted (e.g., (1/(rank+1))**2 for stronger skew)
        weights = [1.0 / (rank + 1) for rank in range(num_candidates)]
        
        # Choose one candidate based on the weights
        # random.choices returns a list of k elements, so take the first one
        chosen_candidate = random.choices(sorted_candidate_targets, weights=weights, k=1)[0]
        return chosen_candidate[0] # Return the original index from 'coordinates'
    else:
        # Original random selection
        return random.choice(indices_in_center)


def generate_train_sample(image_paths, gt_paths, augment_size=AUGMENTATION_SIZE,
                          model_input_size=MODEL_INPUT_SIZE, psf_sigma=GT_PSF_SIGMA,
                          negative_prob=0.1, skew_target_selection_to_top=False,
                          skew_patch_to_top=False, top_patch_bias_range_percentage=0.25): # Added patch skewing args
    max_retries = 10
    for _ in range(max_retries):
        rand_idx = random.randint(0, len(image_paths) - 1)
        image_path = image_paths[rand_idx]
        img_filename = os.path.basename(image_path)
        gt_filename = "GT_" + os.path.splitext(img_filename)[0] + ".mat"
        gt_path = os.path.join(os.path.dirname(gt_paths[0]), gt_filename) if gt_paths else os.path.join(os.path.dirname(image_path).replace("images", "ground_truth"), gt_filename)

        if not os.path.exists(gt_path):
             if rand_idx < len(gt_paths):
                 gt_path_fallback = gt_paths[rand_idx]
                 if os.path.exists(gt_path_fallback): gt_path = gt_path_fallback
                 else: print(f"Warning: GT file not found for {image_path}. Skipping."); continue
             else: print(f"Warning: GT index out of bounds for {image_path}. Skipping."); continue

        image = cv2.imread(image_path)
        if image is None: print(f"Warning: Failed to load image {image_path}. Skipping."); continue
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] != 3: print(f"Warning: Image {image_path} has unexpected shape {image.shape}. Skipping."); continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            mat_data = loadmat(gt_path)
            if 'image_info' in mat_data: gt_coor = mat_data['image_info'][0, 0][0, 0][0].astype(np.float32)
            elif 'annPoints' in mat_data: gt_coor = mat_data['annPoints'].astype(np.float32)
            else:
                 found_coords = False
                 for key, value in mat_data.items():
                     if isinstance(value, np.ndarray) and len(value.shape) == 2 and value.shape[1] == 2:
                         gt_coor = value.astype(np.float32); found_coords = True; break
                 if not found_coords: print(f"Warning: Could not find coordinate data in {gt_path}. Skipping."); continue
        except Exception as e: print(f"Warning: Error loading/parsing .mat {gt_path}: {e}. Skipping."); continue

        if gt_coor.shape[0] == 0: continue # Skip images with no ground truth points initially

        aug_image, aug_gt_coor = prepare_data_augmentations(
            image, gt_coor.copy(), target_size=augment_size,
            skew_patch_to_top=skew_patch_to_top, top_patch_bias_range_percentage=top_patch_bias_range_percentage)
        if aug_image is None or aug_gt_coor is None: continue
        
        if aug_gt_coor.shape[0] < 1: continue 

        img_h, img_w = aug_image.shape[:2]
        if img_h == 0 or img_w == 0: print(f"Warning: Augmented image has zero dimension for {image_path}. Skipping."); continue

        sorted_indices = np.lexsort((aug_gt_coor[:, 0], -aug_gt_coor[:, 1]))
        sorted_aug_gt_coor = aug_gt_coor[sorted_indices]
        num_actual_points_in_aug_crop = len(sorted_aug_gt_coor)

        is_negative_sample = random.random() < negative_prob
        num_previous_points_for_input_psf = 0 

        if is_negative_sample:
            confidence_target = 0.0
            target_psf_full = np.zeros((img_h, img_w), dtype=np.float32) 
            num_previous_points_for_input_psf = num_actual_points_in_aug_crop
        else: 
            confidence_target = 1.0
            center_crop_shape = (model_input_size, model_input_size)
            
            # Pass skew_target_selection_to_top to get_random_coord_index_in_center
            timestep_k_for_target = get_random_coord_index_in_center(
                sorted_aug_gt_coor, 
                (img_h, img_w), 
                center_crop_shape,
                skew_to_top=skew_target_selection_to_top # New argument passed here
            )
            
            if timestep_k_for_target is None: 
                 continue 

            target_psf_full = generate_single_psf(sorted_aug_gt_coor[timestep_k_for_target], (img_h, img_w), psf_sigma)
            num_previous_points_for_input_psf = timestep_k_for_target 

        input_psf_full = np.zeros((img_h, img_w), dtype=np.float32)
        if num_previous_points_for_input_psf > 0:
            previous_points_map = np.zeros((img_h, img_w), dtype=np.float32)
            for i in range(num_previous_points_for_input_psf):
                coord = sorted_aug_gt_coor[i]
                x = np.clip(int(round(coord[0])), 0, img_w - 1)
                y = np.clip(int(round(coord[1])), 0, img_h - 1)
                previous_points_map[y, x] += 1.0
            if np.sum(previous_points_map) > 1e-7:
                 input_psf_full = gaussian_filter(previous_points_map, sigma=psf_sigma, order=0, mode='constant', cval=0.0)

        center_crop_shape = (model_input_size, model_input_size)
        start_y, start_x = get_center_crop_coords((img_h, img_w), center_crop_shape)
        end_y, end_x = start_y + model_input_size, start_x + model_input_size
        if start_y < 0 or start_x < 0 or end_y > img_h or end_x > img_w: print(f"Warning: Invalid crop coordinates for {image_path}. Skipping."); continue

        final_image = aug_image[start_y:end_y, start_x:end_x]
        final_input_psf = input_psf_full[start_y:end_y, start_x:end_x]
        final_target_psf = target_psf_full[start_y:end_y, start_x:end_x]

        if final_image.shape[:2] != center_crop_shape or \
           final_input_psf.shape != center_crop_shape or \
           final_target_psf.shape != center_crop_shape: print(f"Warning: Cropped shape mismatch for {image_path}. Skipping."); continue

        final_image_tensor = torch.from_numpy(final_image.copy()).permute(2, 0, 1).float() / 255.0
        final_image_tensor = (final_image_tensor - IMG_MEAN) / IMG_STD

        max_val_in = np.max(final_input_psf)
        if max_val_in > 1e-7: final_input_psf = final_input_psf / max_val_in
        final_input_psf_tensor = torch.from_numpy(final_input_psf).float().unsqueeze(0)

        if not is_negative_sample: 
            target_psf_sum = np.sum(final_target_psf)
            if target_psf_sum > 1e-7:
                final_target_psf = final_target_psf / target_psf_sum
        final_target_psf_tensor = torch.from_numpy(final_target_psf).float().unsqueeze(0)
        
        confidence_target_tensor = torch.tensor(confidence_target, dtype=torch.float32)

        expected_shape = (model_input_size, model_input_size)
        if final_image_tensor.shape != (3, *expected_shape) or \
           final_input_psf_tensor.shape != (1, *expected_shape) or \
           final_target_psf_tensor.shape != (1, *expected_shape):
            print(f"Warning: Final shape mismatch for {image_path}. Retrying."); continue

        return final_image_tensor, final_input_psf_tensor, final_target_psf_tensor, confidence_target_tensor

    print(f"Warning: Failed to generate a valid sample after {max_retries} retries. Returning None tuple.")
    return None, None, None, None


def generate_batch(image_paths, gt_paths, batch_size, generation_fn=generate_train_sample, **kwargs):
    """Generates a batch of data including confidence targets."""
    image_batch, input_psf_batch, output_psf_batch, confidence_target_batch = [], [], [], []
    attempts = 0
    max_attempts = batch_size * 10 

    while len(image_batch) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            # Pass kwargs (like negative_prob, skew_target_selection_to_top) to generation_fn
            sample = generation_fn(image_paths, gt_paths, **kwargs) 

            if sample is not None and sample[0] is not None: 
                img, in_psf, out_psf, conf_tgt = sample
                if isinstance(img, torch.Tensor) and isinstance(in_psf, torch.Tensor) and \
                   isinstance(out_psf, torch.Tensor) and isinstance(conf_tgt, torch.Tensor):
                    image_batch.append(img)
                    input_psf_batch.append(in_psf)
                    output_psf_batch.append(out_psf)
                    confidence_target_batch.append(conf_tgt)
                else:
                    print(f"Warning: generation_fn returned non-Tensor data. Skipping. Types: {type(img)}, {type(in_psf)}, {type(out_psf)}, {type(conf_tgt)}")
        except Exception as e:
            import traceback
            print(f"Error during sample generation: {e}. Skipping sample.")
            print(traceback.format_exc())
            continue

    if not image_batch: 
        print(f"Warning: Failed to generate any valid samples for a batch after {max_attempts} attempts.")
        return None, None, None, None

    try:
        final_image_batch = torch.stack(image_batch)
        final_input_psf_batch = torch.stack(input_psf_batch)
        final_output_psf_batch = torch.stack(output_psf_batch)
        final_confidence_target_batch = torch.stack(confidence_target_batch) 
        return final_image_batch, final_input_psf_batch, final_output_psf_batch, final_confidence_target_batch
    except Exception as e:
        print(f"Error during torch.stack: {e}")
        if image_batch: print("Individual image shapes:", [t.shape for t in image_batch])
        if input_psf_batch: print("Individual input_psf shapes:", [t.shape for t in input_psf_batch])
        if output_psf_batch: print("Individual output_psf shapes:", [t.shape for t in output_psf_batch])
        if confidence_target_batch: print("Individual confidence target shapes:", [t.shape for t in confidence_target_batch])
        return None, None, None, None

if __name__ == "__main__":
    print("Running dataset.py test...")
    test_image_dir = IMAGE_DIR_TRAIN_VAL
    test_gt_dir = GT_DIR_TRAIN_VAL
    num_samples_to_show = 5 

    image_paths = sorted(glob.glob(os.path.join(test_image_dir, '*.jpg')))
    gt_paths = sorted(glob.glob(os.path.join(test_gt_dir, '*.mat')))
    if not image_paths: print(f"Error: No images found in {test_image_dir}"); exit()
    print(f"Found {len(image_paths)} images.")

    for i in range(num_samples_to_show):
        # Test with skewing enabled for one of the samples
        skew_target_test = True if i >= num_samples_to_show - 2 else False # Enable for the last two samples
        skew_patch_test = True if i == num_samples_to_show -1 else False # Enable for the last sample
        
        print(f"\n--- Generating Sample {i+1}/{num_samples_to_show} (neg_prob=0.3, skew_target_pt={skew_target_test}, skew_patch_loc={skew_patch_test}) ---")
        sample_data = generate_train_sample(
            image_paths, gt_paths,
            augment_size=AUGMENTATION_SIZE, model_input_size=MODEL_INPUT_SIZE,
            psf_sigma=GT_PSF_SIGMA, negative_prob=0.3,
            skew_target_selection_to_top=skew_target_test,
            skew_patch_to_top=skew_patch_test
        )

        if sample_data is None or sample_data[0] is None:
            print("Failed to generate a valid sample. Skipping visualization.")
            continue

        img_tensor, input_psf_tensor, target_psf_tensor, confidence_target_tensor = sample_data

        print(f"Generated Tensor Shapes:")
        print(f"  Image:      {img_tensor.shape}")
        print(f"  Input PSF:  {input_psf_tensor.shape}")
        print(f"  Target PSF: {target_psf_tensor.shape}")
        print(f"  Confidence Target: {confidence_target_tensor.shape}, Value: {confidence_target_tensor.item():.1f}")

        img_vis = img_tensor.cpu() * IMG_STD.cpu() + IMG_MEAN.cpu()
        img_vis = torch.clamp(img_vis, 0, 1)
        img_vis_np = img_vis.permute(1, 2, 0).numpy()        

        input_psf_np = input_psf_tensor.squeeze().cpu().numpy()
        target_psf_np = target_psf_tensor.squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        title_suffix = "POSITIVE" if confidence_target_tensor.item() == 1.0 else "NEGATIVE (No Next Person)"
        skew_target_title = " (SkewTgt)" if skew_target_test and confidence_target_tensor.item() == 1.0 else ""
        skew_patch_title = " (SkewPatch)" if skew_patch_test else ""
        fig.suptitle(f'Sample {i+1} ({title_suffix}{skew_target_title}{skew_patch_title})', fontsize=16)

        axes[0].imshow(img_vis_np)
        axes[0].set_title(f'Input Image ({MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE})')
        axes[0].axis('off')

        im_in = axes[1].imshow(input_psf_np, cmap='viridis', vmin=0)
        axes[1].set_title(f'Input PSF (Max: {np.max(input_psf_np):.4f})')
        axes[1].axis('off')
        fig.colorbar(im_in, ax=axes[1], fraction=0.046, pad=0.04)

        im_tgt = axes[2].imshow(target_psf_np, cmap='viridis', vmin=0)
        axes[2].set_title(f'Target PSF (Sum: {np.sum(target_psf_np):.4f})')
        axes[2].axis('off')
        fig.colorbar(im_tgt, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    print("\nDataset test finished.")