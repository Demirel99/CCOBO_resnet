#model.py
"""
Model definitions including ResNet Encoder, FPN Decoder, ASPP, and PSF Head.
This version uses a ResNet backbone and Cross-Attention fusion for improved performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Import from config
from config import PSF_HEAD_TEMP, MODEL_INPUT_SIZE

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module."""
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        # 1x1 conv
        self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        # Atrous convs
        for rate in rates[1:]:
             self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                        padding=rate, dilation=rate, bias=False))

        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True) # ReLU after GAP conv
        )

        # Batch norm for each branch
        self.bn_ops = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(len(self.convs) + 1)]) # +1 for GAP

        # Final 1x1 conv and dropout
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(self.convs) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), # BN after final projection
            nn.ReLU(inplace=True),
            nn.Dropout(0.2) # Consider placing dropout after ReLU
        )

    def forward(self, x):
        size = x.shape[2:]
        features = []
        # Parallel convolutions
        for i, conv in enumerate(self.convs):
            features.append(F.relu(self.bn_ops[i](conv(x)))) # ReLU after BN
        # Global pooling
        gap_feat = self.global_pool(x)
        gap_feat = F.interpolate(gap_feat, size=size, mode='bilinear', align_corners=False)
        features.append(self.bn_ops[-1](gap_feat)) # BN for GAP feature

        # Concatenate and project
        x = torch.cat(features, dim=1)
        x = self.project(x)
        return x

class ResNetEncoder(nn.Module):
    """Encodes an image using ResNet50 features at multiple scales."""
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        # Load a pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # We'll capture features from different stages
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # C1 (after maxpool)
        self.layer1 = resnet.layer1 # C2
        self.layer2 = resnet.layer2 # C3
        self.layer3 = resnet.layer3 # C4
        self.layer4 = resnet.layer4 # C5

    def forward(self, x):
        c1 = self.layer0(x)     # out: 64 channels, stride 4
        c2 = self.layer1(c1)    # out: 256 channels, stride 4
        c3 = self.layer2(c2)    # out: 512 channels, stride 8
        c4 = self.layer3(c3)    # out: 1024 channels, stride 16
        c5 = self.layer4(c4)    # out: 2048 channels, stride 32
        
        return [c1, c2, c3, c4, c5]

class CrossAttentionFusion(nn.Module):
    """
    Fuses image features and mask features using a cross-attention mechanism.
    The mask features (query) attend to the image features (key, value).
    """
    def __init__(self, image_channels, mask_channels, output_channels):
        super().__init__()
        # The residual connection requires image_channels == output_channels
        assert image_channels == output_channels, \
            "CrossAttentionFusion with residual connection requires image_channels to be equal to output_channels."

        self.q_conv = nn.Conv2d(mask_channels, output_channels, 1)
        self.k_conv = nn.Conv2d(image_channels, output_channels, 1)
        self.v_conv = nn.Conv2d(image_channels, output_channels, 1)
        self.scale = output_channels ** -0.5
        self.output_conv = nn.Conv2d(output_channels, output_channels, 1)

    def forward(self, image_feat, mask_feat):
        # image_feat: (B, C_img, H, W) -> e.g., C5 from ResNet
        # mask_feat: (B, C_mask, H, W) -> from SmallPSFEncoder
        
        B, _, H, W = image_feat.shape
        
        # Project to Q, K, V
        q = self.q_conv(mask_feat).view(B, -1, H * W)  # Query from mask
        k = self.k_conv(image_feat).view(B, -1, H * W)  # Key from image
        v = self.v_conv(image_feat).view(B, -1, H * W)  # Value from image
        
        # Attention score calculation (b, hw, c) x (b, c, hw) -> (b, hw, hw)
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to value (b, c, hw) x (b, hw, hw) -> (b, c, hw)
        attended_value = torch.bmm(v, attn.transpose(1, 2))
        
        # Reshape and process through output conv
        attended_value = attended_value.view(B, -1, H, W)
        
        # Residual connection with the original image feature
        output = image_feat + self.output_conv(attended_value)
        return output

class SmallPSFEncoder(nn.Module):
    """Encodes the 1-channel input PSF mask."""
    def __init__(self):
        super(SmallPSFEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)

class FPNDecoder(nn.Module):
    """Feature Pyramid Network (FPN) decoder."""
    def __init__(self, encoder_channels=[64, 128, 256, 512, 512], fpn_channels=256, out_channels=64):
        super(FPNDecoder, self).__init__()
        assert len(encoder_channels) == 5, "Expected 5 encoder channel numbers for C1 to C5_effective."
        self.lateral_convs = nn.ModuleList()
        for enc_ch in reversed(encoder_channels):
            self.lateral_convs.append(nn.Conv2d(enc_ch, fpn_channels, kernel_size=1))
        self.smooth_convs = nn.ModuleList()
        for _ in range(len(encoder_channels)):
             self.smooth_convs.append(nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1))
        self.final_conv = nn.Conv2d(fpn_channels, out_channels, kernel_size=3, padding=1)

    def _upsample_add(self, top_down_feat, lateral_feat):
        _, _, H, W = lateral_feat.shape
        upsampled_feat = F.interpolate(top_down_feat, size=(H, W), mode='bilinear', align_corners=False)
        return upsampled_feat + lateral_feat

    def forward(self, x_top, encoder_features_c1_c4):
        C1, C2, C3, C4 = encoder_features_c1_c4
        all_features = [C1, C2, C3, C4, x_top]
        pyramid_features = []
        p = self.lateral_convs[0](all_features[-1])
        p = self.smooth_convs[0](p)
        pyramid_features.append(p)
        for i in range(1, len(self.lateral_convs)):
            lateral_idx = len(all_features) - 1 - i
            lateral_feat = self.lateral_convs[i](all_features[lateral_idx])
            p_prev = pyramid_features[-1]
            top_down_feat = self._upsample_add(p_prev, lateral_feat)
            p = self.smooth_convs[i](top_down_feat)
            pyramid_features.append(p)
        p1_output = pyramid_features[-1]
        out = F.relu(self.final_conv(p1_output))
        return out

class PSFHead(nn.Module):
    """Predicts the PSF map and confidence score from the final decoder features."""
    def __init__(self, in_channels, temperature=PSF_HEAD_TEMP):
        super(PSFHead, self).__init__()
        # PSF map branch
        self.psf_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)  # Output 1 channel (logits for PSF)
        )
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)  # Softmax over spatial dimensions (H*W)
        
        # Confidence branch
        self.confidence_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # Global average pooling of input features 'x'
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 2), # Linear layer on pooled features
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, 1) # Output logits for confidence (NO SIGMOID HERE)
        )

    def forward(self, x):
        # PSF Map Prediction
        psf_logits = self.psf_branch(x) # Shape: (B, 1, H, W)
        b, c, h, w = psf_logits.shape
        
        reshaped_logits = psf_logits.view(b, c, -1) 
        if self.temperature > 1e-6: 
            reshaped_logits = reshaped_logits / self.temperature
        
        psf_distribution = self.softmax(reshaped_logits) 
        output_psf_map = psf_distribution.view(b, c, h, w) 
        
        # Confidence Score Prediction
        confidence_logits = self.confidence_branch(x) # Shape: (B, 1)
        
        return output_psf_map, confidence_logits


class ResNetFPNASPP(nn.Module):
    """
    The main model combining a ResNet-50 Encoder, Cross-Attention Fusion, 
    ASPP, FPN, and a dual-output PSF Head.
    """
    def __init__(self):
        super(ResNetFPNASPP, self).__init__()
        self.image_encoder = ResNetEncoder()
        self.mask_encoder = SmallPSFEncoder()

        # Channel dimensions from ResNet-50 Encoder
        resnet_c1_channels = 64
        resnet_c2_channels = 256
        resnet_c3_channels = 512
        resnet_c4_channels = 1024
        resnet_c5_channels = 2048
        mask_features_channels = 64

        # Fusion module combines C5 image features and mask features
        self.fusion_c5 = CrossAttentionFusion(
            image_channels=resnet_c5_channels,
            mask_channels=mask_features_channels,
            output_channels=resnet_c5_channels # Required for the residual connection
        )

        # ASPP processes the fused features and reduces channel dimension
        aspp_out_channels = 512
        self.aspp_c5 = ASPP(in_channels=resnet_c5_channels, out_channels=aspp_out_channels)
        
        # FPN decoder gets features from ResNet and the processed C5 from ASPP
        fpn_encoder_channels = [
            resnet_c1_channels, resnet_c2_channels, resnet_c3_channels, 
            resnet_c4_channels, aspp_out_channels
        ]
        self.fpn_decoder = FPNDecoder(
             encoder_channels=fpn_encoder_channels,
             fpn_channels=256,
             out_channels=64
         )
        self.psf_head = PSFHead(in_channels=64, temperature=PSF_HEAD_TEMP)

    def forward(self, image, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        encoder_features = self.image_encoder(image)
        C1, C2, C3, C4, C5 = encoder_features
        mask_features = self.mask_encoder(mask)

        # Fuse C5 and mask features using cross-attention
        fused_c5 = self.fusion_c5(C5, mask_features)
        
        # Process fused features through ASPP
        aspp_output = self.aspp_c5(fused_c5)
        
        # Decode using FPN
        decoder_output = self.fpn_decoder(aspp_output, [C1, C2, C3, C4])
        
        # Predict final map and confidence
        predicted_psf_map, confidence_score = self.psf_head(decoder_output)

        return predicted_psf_map, confidence_score
