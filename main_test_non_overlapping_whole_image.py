import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from tqdm import tqdm

# --- Configuration and Model Imports ---
from config import DEVICE, MODEL_INPUT_SIZE, GT_PSF_SIGMA
from model import VGG19FPNASPP


BEST_MODEL_PATH=r"C:\Users\Mehmet_Postdoc\Desktop\python_set_up_code\iterative_crowd_counting_with_confidence\crowd_counting_outputs\best_model_fpn_100k.pth"
print(f"INFO: Script will use device: {DEVICE}")

# ImageNet Mean/Std for normalization/unnormalization
IMG_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Helper functions (no changes here) ---
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

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def load_full_image_gt_points(gt_file_path):
    if not os.path.exists(gt_file_path):
        print(f"Warning: GT file not found at {gt_file_path}")
        return []
    try:
        mat_data = loadmat(gt_file_path)
        gt_points_full_image = mat_data['image_info'][0,0][0,0][0]
        return [(p[0], p[1]) for p in gt_points_full_image]
    except Exception as e:
        print(f"Error loading or parsing GT file {gt_file_path}: {e}")
        return []

def non_max_suppression_points(points_with_scores, min_distance):
    if not points_with_scores:
        return []
    sorted_points = sorted(points_with_scores, key=lambda item: item[1], reverse=True)
    kept_points = []
    while sorted_points:
        best_point_tuple = sorted_points.pop(0)
        kept_points.append(best_point_tuple)
        best_coord = best_point_tuple[0]
        remaining_points = []
        for point_tuple in sorted_points:
            coord = point_tuple[0]
            if euclidean_distance(best_coord, coord) >= min_distance:
                remaining_points.append(point_tuple)
        sorted_points = remaining_points
    return kept_points

def process_patch_iteratively(
    model,
    patch_p11_np,
    max_iterations, # Renamed for clarity
    confidence_threshold,
    num_peaks_to_consider,
    min_peak_distance,
    progress_bar=None, # To write messages without breaking the bar
    patch_row_idx=0,
    patch_col_idx=0
):
    """
    Processes a single image patch iteratively, with early stopping based on confidence.
    """
    patch_tensor_np_float = patch_p11_np.astype(np.float32) / 255.0
    patch_tensor_chw = torch.from_numpy(patch_tensor_np_float).permute(2, 0, 1)
    patch_tensor_norm = (patch_tensor_chw - IMG_MEAN_CPU) / IMG_STD_CPU
    image_patch_for_model = patch_tensor_norm.unsqueeze(0).to(DEVICE)

    current_input_psf_np = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.float32)
    predicted_points_history = []

    for iteration in range(max_iterations):
        current_input_psf_tensor = torch.from_numpy(current_input_psf_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            predicted_output_psf_tensor, predicted_confidence_logits = model(image_patch_for_model, current_input_psf_tensor)

        output_psf_model_np = predicted_output_psf_tensor.squeeze().cpu().numpy()
        confidence_score = torch.sigmoid(predicted_confidence_logits).item()

        # --- CORE CHANGE: EARLY STOPPING ---
        # If confidence is low, the model thinks it's done with this patch.
        # We only proceed if confidence is HIGH (>= threshold).
        if confidence_score < confidence_threshold and iteration > 0: # Check after 1st iter to find at least one point if possible
            if progress_bar:
                # Use tqdm.write to print messages without disturbing the progress bar
                progress_bar.write(
                    f"INFO: Stopping patch ({patch_row_idx},{patch_col_idx}) at iter {iteration+1}. "
                    f"Confidence ({confidence_score:.3f}) < threshold ({confidence_threshold})."
                )
            break

        psf_max = np.max(output_psf_model_np)
        dynamic_abs_threshold = max(0.1 * psf_max, 1e-5)

        peak_coordinates_yx = np.array([])
        if psf_max > 1e-6:
            peak_coordinates_yx = peak_local_max(
                output_psf_model_np, min_distance=min_peak_distance,
                threshold_abs=dynamic_abs_threshold, num_peaks=num_peaks_to_consider,
                exclude_border=False
            )

        potential_new_peaks_with_scores = []
        if peak_coordinates_yx.shape[0] > 0:
            for r_idx, c_idx in peak_coordinates_yx:
                potential_new_peaks_with_scores.append(((c_idx, r_idx), output_psf_model_np[r_idx, c_idx]))
        potential_new_peaks_with_scores.sort(key=lambda item: item[1], reverse=True)

        selected_new_peak_for_this_iteration = False
        history_coords = [p[0] for p in predicted_points_history]

        for new_peak_coord_xy, _ in potential_new_peaks_with_scores:
            is_too_close_to_history = any(euclidean_distance(new_peak_coord_xy, hist_peak_xy) < min_peak_distance for hist_peak_xy in history_coords)
            if not is_too_close_to_history:
                # We store the point regardless of the confidence that triggered its finding.
                # The confidence is a measure of "are we done yet?", not "how good is this point?".
                predicted_points_history.append((new_peak_coord_xy, confidence_score))
                selected_new_peak_for_this_iteration = True
                break

        # Stop if no new valid peak is found (another stopping condition)
        if not selected_new_peak_for_this_iteration and iteration > 0:
            if progress_bar:
                progress_bar.write(f"INFO: Stopping patch ({patch_row_idx},{patch_col_idx}) at iter {iteration+1}. No new valid peak found.")
            break

        points_for_next_psf = [p[0] for p in predicted_points_history]
        current_input_psf_np = create_psf_from_points(
            points_list=points_for_next_psf,
            shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            sigma=GT_PSF_SIGMA
        )
    
    # We still return all points found during the high-confidence iterations.
    # The confidence score itself is not a per-point score, but a per-iteration score.
    # So we just return the coordinates.
    return [p[0] for p in predicted_points_history]


def main():
    print(f"--- Iterative Processing of Full Image with Early Stopping ---")


    # --- Configuration ---
    PATCH_SIZE = MODEL_INPUT_SIZE
    MAX_ITERATIONS = 500 # This is now a maximum limit per patch
    NUM_PEAKS_TO_CONSIDER = 50
    MIN_PEAK_DISTANCE = 2
    CONFIDENCE_THRESHOLD = 0.9 # THRESHOLD TO STOP PROCESSING A PATCH
    NMS_MIN_DISTANCE = 2
    all_absolute_errors = []
    for image_nume in range(103,104):
        im_num= image_nume + 1
        print(f"\n--- Processing Image {im_num} ---")
        image_file_path = r"C:\Users\Mehmet_Postdoc\Desktop\datasets_for_experiments\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\test_data\images\IMG_%d.jpg"% im_num
        
        # --- Load Image and Model ---
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
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
        model.eval()
        print(f"Model loaded and in evaluation mode on {DEVICE}.")

        # --- Prepare Full Image (Padding) ---
        h_orig, w_orig = image_rgb_orig.shape[:2]
        pad_h_bottom = (PATCH_SIZE - h_orig % PATCH_SIZE) % PATCH_SIZE
        pad_w_right = (PATCH_SIZE - w_orig % PATCH_SIZE) % PATCH_SIZE
        padded_image_rgb = cv2.copyMakeBorder(image_rgb_orig, 0, pad_h_bottom, 0, pad_w_right,
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        h_pad, w_pad = padded_image_rgb.shape[:2]
        num_patches_h = h_pad // PATCH_SIZE
        num_patches_w = w_pad // PATCH_SIZE
        print(f"Image padded to {padded_image_rgb.shape}. Grid size: {num_patches_h}x{num_patches_w} patches.")

        # --- Load Full Ground Truth ---
        img_filename = os.path.basename(image_file_path)
        gt_filename = "GT_" + img_filename.replace(".jpg", ".mat")
        gt_file_path = os.path.join(os.path.dirname(os.path.dirname(image_file_path)), "ground_truth", gt_filename)
        all_gt_points = load_full_image_gt_points(gt_file_path)
        print(f"Loaded {len(all_gt_points)} total ground truth points for the image.")
        
        # --- Process All Patches ---
        all_predictions_before_nms = []
        
        patch_indices = [(r, c) for r in range(num_patches_h) for c in range(num_patches_w)]
        
        with tqdm(total=len(patch_indices), desc="Processing Patches") as pbar:
            for r_idx, c_idx in patch_indices:
                r_start = r_idx * PATCH_SIZE
                c_start = c_idx * PATCH_SIZE
                patch_np = padded_image_rgb[r_start : r_start + PATCH_SIZE,
                                            c_start : c_start + PATCH_SIZE]

                # Returns only coordinates
                points_in_patch = process_patch_iteratively(
                    model=model, patch_p11_np=patch_np, max_iterations=MAX_ITERATIONS,
                    confidence_threshold=CONFIDENCE_THRESHOLD, num_peaks_to_consider=NUM_PEAKS_TO_CONSIDER,
                    min_peak_distance=MIN_PEAK_DISTANCE, progress_bar=pbar,
                    patch_row_idx=r_idx, patch_col_idx=c_idx
                )

                for patch_x, patch_y in points_in_patch:
                    full_image_x = c_start + patch_x
                    full_image_y = r_start + patch_y
                    if full_image_x < w_orig and full_image_y < h_orig:
                        # For NMS, we need a "score". Since we stopped based on a global confidence,
                        # we can assign a dummy score (e.g., 1.0) or reuse the last high confidence.
                        # Let's just use the coordinates, as NMS for points often doesn't need scores.
                        # Or, let's just make one up for the NMS function to work.
                        all_predictions_before_nms.append(((full_image_x, full_image_y), 1.0))
                pbar.update(1)


        # --- Apply NMS to all collected points ---
        print(f"\n--- Applying Non-Maximum Suppression ---")
        print(f"Found {len(all_predictions_before_nms)} points before NMS.")
        
        final_points_after_nms = non_max_suppression_points(
            all_predictions_before_nms, 
            min_distance=NMS_MIN_DISTANCE
        )
        
        # --- Final Reporting ---
        print(f"--- Full Image Processing Finished ---")
        predicted_count = len(final_points_after_nms)
        gt_count = len(all_gt_points)
        print(f"Total Predicted Count (after NMS): {predicted_count}")
        print(f"Total Ground Truth Count: {gt_count}")
        print(f"Absolute Error: {abs(predicted_count - gt_count)}")
        
        all_absolute_errors.append(abs(predicted_count - gt_count))

        # --- Final Visualization for the Whole Image ---
        plt.figure(figsize=(16, 12))
        plt.imshow(image_rgb_orig)
        title_str = f"Final Predictions vs GT for {img_filename} (After NMS)\n"
        title_str += f"Predicted: {predicted_count} | GT: {gt_count}"
        plt.title(title_str, fontsize=14)

        if final_points_after_nms:
            pred_coords_only = [p[0] for p in final_points_after_nms]
            pred_points_np = np.array(pred_coords_only)
            plt.plot(pred_points_np[:, 0], pred_points_np[:, 1], 'rx', markersize=8, mew=1.5, label=f'Predicted ({predicted_count})')

        if all_gt_points:
            gt_points_np = np.array(all_gt_points)
            plt.plot(gt_points_np[:, 0], gt_points_np[:, 1], 'go', markersize=6, mfc='none', mew=1.5, label=f'Ground Truth ({gt_count})')

        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    mae= np.mean(all_absolute_errors)
    print(f"Mean Absolute Error across all images: {mae}")


if __name__ == "__main__":
    main()