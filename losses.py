#losses.py
"""
Loss functions for model training.
"""
import torch
import torch.nn.functional as F # Added for F.binary_cross_entropy

def kl_divergence_loss(predicted_psf, target_psf, epsilon=1e-7):
    """
    Computes KL Divergence loss: sum(target * log(target / predicted)).
    Assumes inputs are probability distributions (sum to 1 spatially).

    Args:
        predicted_psf (torch.Tensor): Predicted map (B, 1, H, W). Output of Softmax.
        target_psf (torch.Tensor): Target map (B, 1, H, W). Should sum to 1 spatially for positive, 0 for negative.
        epsilon (float): Small value for numerical stability.

    Returns:
        torch.Tensor: Scalar KL divergence loss averaged over the batch.
    """
    pred_clamped = torch.clamp(predicted_psf, min=epsilon)
    # For negative samples, target_psf is all zeros. KL div should be 0.
    # For positive samples, target_psf sums to 1.
    
    # Only compute KL divergence where target_psf > epsilon (i.e., for positive samples' target points)
    # Where target_psf is 0 (negative samples, or non-target areas of positive samples), KL component is 0.
    kl_div_elements = target_psf * (torch.log(torch.clamp(target_psf, min=epsilon)) - torch.log(pred_clamped))
    
    # Sum over spatial dimensions (H, W)
    kl_div_sample = kl_div_elements.sum(dim=(2, 3)) # Shape (B, 1)

    # Average over batch.
    # For negative samples, kl_div_sample will be 0.
    # We only want to average KL loss over positive samples for the KL term,
    # but the combined loss will handle weighting.
    # If a batch is all negative samples, kl_loss will be 0, which is correct.
    kl_loss = kl_div_sample.mean()
    
    return kl_loss


def combined_loss(predicted_psf, predicted_confidence_logits,  # Renamed for clarity
                  target_psf, target_confidence, 
                  kl_weight=1.0, bce_weight=1.0, epsilon_kl=1e-7):
    """
    Combined KL divergence loss for PSF and BCEWithLogits loss for confidence.
    
    Args:
        predicted_psf (torch.Tensor): Predicted PSF map (B, 1, H, W)
        predicted_confidence_logits (torch.Tensor): Predicted confidence LOGITS (B, 1)
        target_psf (torch.Tensor): Target PSF map (B, 1, H, W)
        target_confidence (torch.Tensor): Target confidence (B, 1) values 0 or 1
        kl_weight (float): Weight for KL loss
        bce_weight (float): Weight for BCE loss
        epsilon_kl (float): Epsilon for KL divergence stability
    Returns:
        tuple: (total_loss, kl_loss_val, bce_loss_val)
    """
    kl_loss_val = kl_divergence_loss(predicted_psf, target_psf, epsilon=epsilon_kl)
    
    # Binary Cross Entropy Loss for Confidence using logits
    # predicted_confidence_logits is (B, 1), target_confidence is (B, 1)
    bce_loss_val = F.binary_cross_entropy_with_logits(predicted_confidence_logits, target_confidence) # CHANGED HERE
    
    total_loss = (kl_weight * kl_loss_val) + (bce_weight * bce_loss_val)
    
    return total_loss, kl_loss_val, bce_loss_val