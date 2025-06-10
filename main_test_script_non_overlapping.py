# iterative_patch_processing_multi_peak_v2.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max # For finding multiple peaks

# --- Configuration and Model Imports ---
from config import DEVICE, MODEL_INPUT_SIZE, GT_PSF_SIGMA
from model import VGG19FPNASPP


BEST_MODEL_PATH=r"C:\Users\Mehmet_Postdoc\Desktop\python_set_up_code\iterative_crowd_counting_with_confidence\crowd_counting_outputs\best_model_fpn_100k.pth"
print(f"INFO: Script will use device: {DEVICE}")

# ImageNet Mean/Std for normalization/unnormalization
IMG_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Helper function to create a PSF from a list of points ---
def create_psf_from_points(points_list, shape, sigma):
    height, width = shape
    aggregate_delta_map = np.zeros((height, width), dtype=np.float32)
    if not points_list:
        return gaussian_filter(aggregate_delta_map, sigma=sigma, order=0, mode='constant', cval=0.0)

    for coord in points_list:
        x = np.clip(int(round(coord[0])), 0, width - 1)
        y = np.clip(int(round(coord[1])), 0, height - 1)
        aggregate_delta_map[y, x] += 1.0

    psf_filtered = gaussian_filter(aggregate_delta_map, sigma=sigma, order=0, mode='constant', cval=0.0)
    max_val = np.max(psf_filtered)
    if max_val > 1e-7:
        psf_filtered /= max_val
    return psf_filtered

# --- Helper function to calculate distance between two points ---
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- Helper function to load and prepare GT points for the patch ---
def load_and_prepare_gt_points_for_patch(gt_file_path, r_start_patch, c_start_patch, patch_size_h, patch_size_w):
    """
    Loads ground truth points from a .mat file and filters/transforms them for a specific patch.
    Args:
        gt_file_path (str): Path to the .mat ground truth file.
        r_start_patch (int): Starting row index of the patch in the full image.
        c_start_patch (int): Starting column index of the patch in the full image.
        patch_size_h (int): Height of the patch.
        patch_size_w (int): Width of the patch.
    Returns:
        list: A list of (x, y) tuples representing GT points within the patch,
              with coordinates relative to the patch's top-left corner.
    """
    gt_points_in_patch = []
    if not os.path.exists(gt_file_path):
        print(f"Warning: GT file not found at {gt_file_path}")
        return gt_points_in_patch

    try:
        mat_data = loadmat(gt_file_path)
        # Assuming ShanghaiTech Part A annotation structure
        gt_points_full_image = mat_data['image_info'][0,0][0,0][0]
    except Exception as e:
        print(f"Error loading or parsing GT file {gt_file_path}: {e}")
        return gt_points_in_patch

    for p in gt_points_full_image:
        # Original GT points are (col, row) -> (x, y)
        gt_x, gt_y = p[0], p[1]

        # Transform to patch-relative coordinates
        patch_x = gt_x - c_start_patch
        patch_y = gt_y - r_start_patch

        # Check if the point is within the patch boundaries
        if 0 <= patch_x < patch_size_w and 0 <= patch_y < patch_size_h:
            gt_points_in_patch.append((patch_x, patch_y))

    return gt_points_in_patch


def main():
    print(f"--- Iterative Processing (Multi-Peak Selection v2) of Patch P(1,1) ---")

    TARGET_PATCH_ROW = 1
    TARGET_PATCH_COL = 1
    PATCH_SIZE = MODEL_INPUT_SIZE
    NUM_ITERATIONS = 5
    NUM_PEAKS_TO_CONSIDER = 20
    MIN_PEAK_DISTANCE = 1
    CONFIDENCE_THRESHOLD = 0.9 # Threshold for final counting

    image_file_path = r"C:\Users\Mehmet_Postdoc\Desktop\datasets_for_experiments\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\test_data\images\IMG_8.jpg"

    print(f"Using Image path: {image_file_path}")
    image_bgr = cv2.imread(image_file_path)
    if image_bgr is None:
        print(f"Error: Could not load image from {image_file_path}")
        return
    image_rgb_orig = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Original image loaded. Shape: {image_rgb_orig.shape}")

    print(f"Loading model weights from: {BEST_MODEL_PATH}")
    model = VGG19FPNASPP().to(DEVICE)
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model weights not found at {BEST_MODEL_PATH}")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval()
    print(f"Model loaded and in evaluation mode on {DEVICE}.")

    h_orig, w_orig = image_rgb_orig.shape[:2]
    pad_h_bottom = (PATCH_SIZE - h_orig % PATCH_SIZE) % PATCH_SIZE
    pad_w_right = (PATCH_SIZE - w_orig % PATCH_SIZE) % PATCH_SIZE
    padded_image_rgb = cv2.copyMakeBorder(image_rgb_orig, 0, pad_h_bottom, 0, pad_w_right,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])

    r_start = TARGET_PATCH_ROW * PATCH_SIZE
    c_start = TARGET_PATCH_COL * PATCH_SIZE
    patch_p11_np = padded_image_rgb[r_start : r_start + PATCH_SIZE,
                                    c_start : c_start + PATCH_SIZE]

    patch_tensor_np_float = patch_p11_np.astype(np.float32) / 255.0
    patch_tensor_chw = torch.from_numpy(patch_tensor_np_float).permute(2, 0, 1)
    patch_tensor_norm = (patch_tensor_chw - IMG_MEAN_CPU) / IMG_STD_CPU
    image_patch_for_model = patch_tensor_norm.unsqueeze(0).to(DEVICE)
    print(f"Prepared image patch tensor for model.")

    # --- Load Ground Truth for the selected patch ---
    img_filename = os.path.basename(image_file_path)
    gt_filename = "GT_" + img_filename.replace(".jpg", ".mat")
    gt_file_path = os.path.join(os.path.dirname(os.path.dirname(image_file_path)), "ground_truth", gt_filename)
    
    print(f"Attempting to load GT data from: {gt_file_path}")
    gt_points_in_patch = load_and_prepare_gt_points_for_patch(
        gt_file_path, r_start, c_start, PATCH_SIZE, PATCH_SIZE
    )
    if gt_points_in_patch:
        print(f"Loaded {len(gt_points_in_patch)} GT points for the patch.")
    else:
        print(f"No GT points found or error loading GT for the patch (or patch is empty of GT points).")


    current_input_psf_np = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.float32)
    # Store tuples of ((x, y), confidence_score)
    predicted_points_history = []

    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{NUM_ITERATIONS} ---")

        current_input_psf_tensor = torch.from_numpy(current_input_psf_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            predicted_output_psf_tensor, predicted_confidence_logits = model(image_patch_for_model, current_input_psf_tensor)

        output_psf_model_np = predicted_output_psf_tensor.squeeze().cpu().numpy()
        confidence_score = torch.sigmoid(predicted_confidence_logits).item()
        print(f"  Model Confidence: {confidence_score:.4f}")

        psf_min = np.min(output_psf_model_np)
        psf_max = np.max(output_psf_model_np)
        psf_mean = np.mean(output_psf_model_np)
        print(f"  Output PSF Stats: Min={psf_min:.4e}, Max={psf_max:.4e}, Mean={psf_mean:.4e}")

        dynamic_abs_threshold = max(0.1 * psf_max, 1e-5)
        print(f"  Using dynamic absolute threshold for peak finding: {dynamic_abs_threshold:.4e}")

        peak_coordinates_yx = np.array([])
        if psf_max > 1e-6:
            peak_coordinates_yx = peak_local_max(
                output_psf_model_np,
                min_distance=MIN_PEAK_DISTANCE,
                threshold_abs=dynamic_abs_threshold,
                num_peaks=NUM_PEAKS_TO_CONSIDER,
                exclude_border=False
            )

        potential_new_peaks_with_scores = []
        if peak_coordinates_yx.shape[0] > 0:
            for r_idx, c_idx in peak_coordinates_yx:
                peak_value = output_psf_model_np[r_idx, c_idx]
                potential_new_peaks_with_scores.append(((c_idx, r_idx), peak_value))

        potential_new_peaks_with_scores.sort(key=lambda item: item[1], reverse=True)
        print(f"  Found {len(potential_new_peaks_with_scores)} potential new peaks (above dyn_thresh, before history check).")

        current_peak_coord = None
        selected_new_peak_for_this_iteration = False

        # Extract just the coordinates from history for distance checking
        history_coords = [p[0] for p in predicted_points_history]
        for new_peak_coord_xy, new_peak_score in potential_new_peaks_with_scores:
            is_too_close_to_history = False
            for hist_peak_xy in history_coords:
                if euclidean_distance(new_peak_coord_xy, hist_peak_xy) < MIN_PEAK_DISTANCE:
                    is_too_close_to_history = True
                    break

            if not is_too_close_to_history:
                current_peak_coord = new_peak_coord_xy
                print(f"  Selected New Peak: {current_peak_coord} with score {new_peak_score:.4f}")
                # Store the point along with its iteration's confidence score
                predicted_points_history.append((current_peak_coord, confidence_score))
                selected_new_peak_for_this_iteration = True
                break

        if not selected_new_peak_for_this_iteration:
            print("  No new valid peak found in this iteration.")
            if not predicted_points_history:
                current_peak_coord = (MODEL_INPUT_SIZE//2, MODEL_INPUT_SIZE//2)

        # Create the next input PSF from all points found so far
        points_for_next_psf = [p[0] for p in predicted_points_history]
        current_input_psf_np = create_psf_from_points(
            points_list=points_for_next_psf,
            shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            sigma=GT_PSF_SIGMA
        )

        # --- Visualization for the current iteration ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Iteration {iteration + 1} - Confidence: {confidence_score:.4f}", fontsize=14)

        axes[0].imshow(patch_p11_np)
        axes[0].set_title(f"Input Patch P({TARGET_PATCH_ROW},{TARGET_PATCH_COL})")
        ax0_legend_handles = []
        if points_for_next_psf: # Use the list of coords
            hist_points_np = np.array(points_for_next_psf)
            handle_hist, = axes[0].plot(hist_points_np[:,0], hist_points_np[:,1], 'x', c='yellow', markersize=8, label='All Predicted Points')
            ax0_legend_handles.append(handle_hist)
            if selected_new_peak_for_this_iteration and current_peak_coord:
                 handle_curr, = axes[0].plot(current_peak_coord[0], current_peak_coord[1], 'o', c='lime', markersize=10, markeredgecolor='black', label='Current Iteration Point')
                 ax0_legend_handles.append(handle_curr)
        if ax0_legend_handles:
            axes[0].legend(handles=ax0_legend_handles, fontsize='small')
        axes[0].axis('off')

        im_input_psf = axes[1].imshow(current_input_psf_tensor.squeeze().cpu().numpy(), cmap='viridis')
        axes[1].set_title(f"Input PSF to Model (Iter {iteration+1})")
        axes[1].axis('off')
        fig.colorbar(im_input_psf, ax=axes[1], fraction=0.046, pad=0.04)

        im_model_out_psf = axes[2].imshow(output_psf_model_np, cmap='viridis')
        axes[2].set_title(f"Model's Output PSF (Iter {iteration+1})")
        ax2_legend_handles = []
        if peak_coordinates_yx.shape[0] > 0:
            handle_pot, = axes[2].plot(peak_coordinates_yx[:,1], peak_coordinates_yx[:,0], '.', c='cyan', markersize=5, label='Potential Peaks (y,x)')
            ax2_legend_handles.append(handle_pot)
        if selected_new_peak_for_this_iteration and current_peak_coord:
             handle_sel, = axes[2].plot(current_peak_coord[0], current_peak_coord[1], 'x', c='red', markersize=10, label=f'Selected Peak (x,y)')
             ax2_legend_handles.append(handle_sel)
        if ax2_legend_handles:
            axes[2].legend(handles=ax2_legend_handles, fontsize='small')
        axes[2].axis('off')
        fig.colorbar(im_model_out_psf, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.draw()
        plt.pause(0.01)
        plt.close(fig)

        if not selected_new_peak_for_this_iteration and iteration > 0 :
             print("Stopping iterations as no new valid peak was found after the first iteration.")
             break

    # --- Final Filtering and Reporting ---
    print(f"\n--- Iterative processing finished ---")
    print(f"Total points found (with their iteration's confidence score): {predicted_points_history}")
    
    # Filter predictions based on the confidence threshold
    confident_predictions = [p[0] for p in predicted_points_history if p[1] > CONFIDENCE_THRESHOLD]
    
    print(f"\nFinal count (confidence > {CONFIDENCE_THRESHOLD}): {len(confident_predictions)}")
    print(f"Confident point coordinates: {confident_predictions}")
    print(f"Ground truth points in patch ({len(gt_points_in_patch)}): {gt_points_in_patch}")

    # --- Final Visualization with Ground Truth and Confidence Filter ---
    plt.figure(figsize=(10, 10))
    plt.imshow(patch_p11_np)
    title_str = f"Final Predictions vs GT - Patch P({TARGET_PATCH_ROW},{TARGET_PATCH_COL})\n"
    title_str += f"Pred (Conf > {CONFIDENCE_THRESHOLD}): {len(confident_predictions)}, GT: {len(gt_points_in_patch)}"
    plt.title(title_str)

    if confident_predictions:
        pred_points_np = np.array(confident_predictions)
        plt.plot(pred_points_np[:, 0], pred_points_np[:, 1], 'rx', markersize=10, mew=2, label=f'Predicted (Conf > {CONFIDENCE_THRESHOLD}) ({len(confident_predictions)})')

    if gt_points_in_patch:
        gt_points_np = np.array(gt_points_in_patch)
        plt.plot(gt_points_np[:, 0], gt_points_np[:, 1], 'go', markersize=7, mfc='none', mew=2, label=f'Ground Truth ({len(gt_points_in_patch)})')

    if confident_predictions or gt_points_in_patch:
        plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()