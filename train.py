#train.py
"""
Iterative Crowd Counting Model Training Script
"""
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

from config import (
    DEVICE, SEED, TOTAL_ITERATIONS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    VALIDATION_INTERVAL, VALIDATION_BATCHES,
    IMAGE_DIR_TRAIN_VAL, GT_DIR_TRAIN_VAL, OUTPUT_DIR, LOG_FILE_PATH, BEST_MODEL_PATH,
    AUGMENTATION_SIZE, MODEL_INPUT_SIZE, GT_PSF_SIGMA
)
from utils import set_seed, find_and_sort_paths, split_train_val
from dataset import generate_batch, generate_train_sample # generate_train_sample used by generate_batch
from model import VGG19FPNASPP
from losses import combined_loss 

KL_LOSS_WEIGHT = 1.0
BCE_LOSS_WEIGHT = 1.0 
NEGATIVE_SAMPLE_PROB = 0.1 

def train():
    print("Setting up training...")
    set_seed(SEED)

    sorted_image_paths_train_val = find_and_sort_paths(IMAGE_DIR_TRAIN_VAL, '*.jpg')
    sorted_gt_paths_train_val = find_and_sort_paths(GT_DIR_TRAIN_VAL, '*.mat')
    if not sorted_image_paths_train_val or not sorted_gt_paths_train_val:
        raise FileNotFoundError("Training/Validation images or GT files not found.")

    train_image_paths, train_gt_paths, val_image_paths, val_gt_paths = split_train_val(
        sorted_image_paths_train_val, sorted_gt_paths_train_val, val_ratio=0.1, seed=SEED
    )
    if not train_image_paths or not val_image_paths:
        raise ValueError("Train or validation set is empty after splitting.")

    model = VGG19FPNASPP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_ITERATIONS, eta_min=1e-6)

    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp: print("Using Automatic Mixed Precision (AMP).")

    best_val_loss = float('inf')
    iterations_log = []
    train_total_loss_log = []
    val_total_loss_log = []
    val_kl_loss_log = []
    val_bce_loss_log = []

    if os.path.exists(LOG_FILE_PATH):
        try: os.remove(LOG_FILE_PATH)
        except OSError as e: print(f"Warning: Could not remove existing log file: {e}")

    print("Starting training...")
    pbar = tqdm(range(1, TOTAL_ITERATIONS + 1), desc=f"Iteration 1/{TOTAL_ITERATIONS}", unit="iter")

    train_total_loss_accum = 0.0
    train_kl_loss_accum = 0.0
    train_bce_loss_accum = 0.0
    samples_in_accum = 0

    for iteration in pbar:
        model.train()

        img_batch, in_psf_batch, tgt_psf_batch, confidence_target_batch = generate_batch(
            train_image_paths, train_gt_paths, BATCH_SIZE,
            generation_fn=generate_train_sample,
            augment_size=AUGMENTATION_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            psf_sigma=GT_PSF_SIGMA,
            negative_prob=NEGATIVE_SAMPLE_PROB,
            skew_target_selection_to_top=False, # Skew target point to top within patch
                        skew_patch_to_top=False # Skew patch selection to top of image 
        )

        if img_batch is None:
            print(f"Warning: Failed to generate training batch at iteration {iteration}. Skipping.")
            pbar.set_postfix_str("Batch gen failed, skipping")
            if iteration % VALIDATION_INTERVAL == 0 and samples_in_accum == 0 : 
                 print("Skipping validation due to prior training batch failure & no accumulated loss.")
            continue

        img_batch = img_batch.to(DEVICE)
        in_psf_batch = in_psf_batch.to(DEVICE)
        tgt_psf_batch = tgt_psf_batch.to(DEVICE)
        confidence_target_batch = confidence_target_batch.to(DEVICE).unsqueeze(1) 

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            predicted_psf, predicted_confidence_logits = model(img_batch, in_psf_batch) 
            loss, kl_loss_val, bce_loss_val = combined_loss(
                predicted_psf, predicted_confidence_logits, 
                tgt_psf_batch, confidence_target_batch,
                kl_weight=KL_LOSS_WEIGHT, bce_weight=BCE_LOSS_WEIGHT
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_total_loss_accum += loss.item() * img_batch.size(0)
        train_kl_loss_accum += kl_loss_val.item() * img_batch.size(0)
        train_bce_loss_accum += bce_loss_val.item() * img_batch.size(0)
        samples_in_accum += img_batch.size(0)
        
        pbar.set_description(f"Iter {iteration}/{TOTAL_ITERATIONS} | Batch Total Loss: {loss.item():.4f}")

        if iteration % VALIDATION_INTERVAL == 0:
            avg_train_total_loss = train_total_loss_accum / samples_in_accum if samples_in_accum > 0 else 0.0
            avg_train_kl_loss = train_kl_loss_accum / samples_in_accum if samples_in_accum > 0 else 0.0
            avg_train_bce_loss = train_bce_loss_accum / samples_in_accum if samples_in_accum > 0 else 0.0
            
            train_total_loss_log.append(avg_train_total_loss) 

            rng_state = {'random': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                         'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

            model.eval()
            val_total_loss_epoch = 0.0
            val_kl_loss_epoch = 0.0
            val_bce_loss_epoch = 0.0
            total_val_samples = 0
            with torch.no_grad():
                for i in range(VALIDATION_BATCHES):
                    val_seed = SEED + iteration + i; set_seed(val_seed)
                    val_img, val_in_psf, val_tgt_psf, val_conf_tgt = generate_batch(
                        val_image_paths, val_gt_paths, BATCH_SIZE,
                        generation_fn=generate_train_sample,
                        augment_size=AUGMENTATION_SIZE, model_input_size=MODEL_INPUT_SIZE,
                        psf_sigma=GT_PSF_SIGMA, negative_prob=NEGATIVE_SAMPLE_PROB,
                        skew_target_selection_to_top=False, # Skew target point to top within patch
                        skew_patch_to_top=False # Skew patch selection to top of image
                    )
                    if val_img is None: continue

                    val_img = val_img.to(DEVICE)
                    val_in_psf = val_in_psf.to(DEVICE)
                    val_tgt_psf = val_tgt_psf.to(DEVICE)
                    val_conf_tgt = val_conf_tgt.to(DEVICE).unsqueeze(1)

                    with autocast(enabled=use_amp):
                        val_pred_psf, val_pred_conf_logits = model(val_img, val_in_psf) 
                        v_loss, v_kl, v_bce = combined_loss(
                            val_pred_psf, val_pred_conf_logits, val_tgt_psf, val_conf_tgt, 
                            kl_weight=KL_LOSS_WEIGHT, bce_weight=BCE_LOSS_WEIGHT
                        )
                    val_total_loss_epoch += v_loss.item() * val_img.size(0)
                    val_kl_loss_epoch += v_kl.item() * val_img.size(0)
                    val_bce_loss_epoch += v_bce.item() * val_img.size(0)
                    total_val_samples += val_img.size(0)

            random.setstate(rng_state['random']); np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if rng_state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng_state['cuda'])
            set_seed(SEED + iteration + VALIDATION_BATCHES + 1)

            avg_val_total_loss = val_total_loss_epoch / total_val_samples if total_val_samples > 0 else float('inf')
            avg_val_kl_loss = val_kl_loss_epoch / total_val_samples if total_val_samples > 0 else float('inf')
            avg_val_bce_loss = val_bce_loss_epoch / total_val_samples if total_val_samples > 0 else float('inf')

            iterations_log.append(iteration)
            val_total_loss_log.append(avg_val_total_loss)
            val_kl_loss_log.append(avg_val_kl_loss)
            val_bce_loss_log.append(avg_val_bce_loss)

            log_message = (f"Iter [{iteration}/{TOTAL_ITERATIONS}] LR: {optimizer.param_groups[0]['lr']:.2e}\n"
                           f"  Train Total: {avg_train_total_loss:.4f} (KL: {avg_train_kl_loss:.4f}, BCE: {avg_train_bce_loss:.4f})\n"
                           f"  Val   Total: {avg_val_total_loss:.4f} (KL: {avg_val_kl_loss:.4f}, BCE: {avg_val_bce_loss:.4f})")
            print(f"\n{log_message}")
            with open(LOG_FILE_PATH, "a") as log_file: log_file.write(log_message + "\n\n")

            if avg_val_total_loss < best_val_loss:
                best_val_loss = avg_val_total_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"    -> New best model saved with Val Total Loss: {best_val_loss:.4f}")

            train_total_loss_accum = 0.0; train_kl_loss_accum = 0.0; train_bce_loss_accum = 0.0
            samples_in_accum = 0
    
    print("Training complete!")
    pbar.close()

    print("Generating training plots...")
    plt.figure(figsize=(12, 8))
    plt.plot(iterations_log, train_total_loss_log, label='Train Total Loss', alpha=0.7)
    plt.plot(iterations_log, val_total_loss_log, label='Val Total Loss', linewidth=2)
    plt.plot(iterations_log, val_kl_loss_log, label='Val KL Loss', linestyle=':', alpha=0.8)
    plt.plot(iterations_log, val_bce_loss_log, label='Val BCE Loss', linestyle='--', alpha=0.8)
    plt.title("Training and Validation Losses over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_losses_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"Log file saved to: {LOG_FILE_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH} (Val Total Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()