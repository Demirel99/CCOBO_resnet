#model.py
"""
Model definitions including VGG19 Encoder, FPN Decoder, ASPP, and PSF Head.
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

class VGG19Encoder(nn.Module):
    """Encodes an image using VGG19 features at multiple scales."""
    def __init__(self):
        super(VGG19Encoder, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = list(vgg19.features)
        self.feature_layers = nn.ModuleList(features)
        self.capture_indices = {3, 8, 17, 26, 35}

    def forward(self, x):
        results = {}
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i in self.capture_indices:
                 if i == 3: results['C1'] = x
                 elif i == 8: results['C2'] = x
                 elif i == 17: results['C3'] = x
                 elif i == 26: results['C4'] = x
                 elif i == 35: results['C5'] = x
        return [results['C1'], results['C2'], results['C3'], results['C4'], results['C5']]

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
            # nn.Sigmoid()  # REMOVE THIS LINE
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
        # The confidence branch now outputs logits
        confidence_logits = self.confidence_branch(x) # Shape: (B, 1)
        
        return output_psf_map, confidence_logits # Return logits for confidence


class VGG19FPNASPP(nn.Module):
    """The main model combining VGG19, ASPP, FPN, and PSF Head."""
    def __init__(self):
        super(VGG19FPNASPP, self).__init__()
        self.image_encoder = VGG19Encoder()
        self.mask_encoder = SmallPSFEncoder()

        vgg_c1_channels = 64
        vgg_c2_channels = 128
        vgg_c3_channels = 256
        vgg_c4_channels = 512
        vgg_c5_channels = 512
        mask_features_channels = 64

        fusion_in_channels_c5 = vgg_c5_channels + mask_features_channels
        fusion_out_channels_c5 = 512
        self.fusion_conv_c5 = nn.Sequential(
            nn.Conv2d(fusion_in_channels_c5, fusion_out_channels_c5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_out_channels_c5),
            nn.ReLU(inplace=True)
        )
        self.aspp_c5 = ASPP(in_channels=fusion_out_channels_c5, out_channels=fusion_out_channels_c5)
        fpn_encoder_channels = [vgg_c1_channels, vgg_c2_channels, vgg_c3_channels, vgg_c4_channels, fusion_out_channels_c5]
        self.fpn_decoder = FPNDecoder(
             encoder_channels=fpn_encoder_channels,
             fpn_channels=256,
             out_channels=64
         )
        self.psf_head = PSFHead(in_channels=64, temperature=PSF_HEAD_TEMP) # PSFHead now has two branches

    def forward(self, image, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        encoder_features = self.image_encoder(image)
        C1, C2, C3, C4, C5 = encoder_features
        mask_features = self.mask_encoder(mask)

        fused_features = torch.cat([C5, mask_features], dim=1)
        fused_c5 = self.fusion_conv_c5(fused_features)
        aspp_output = self.aspp_c5(fused_c5)
        decoder_output = self.fpn_decoder(aspp_output, [C1, C2, C3, C4])
        
        # PSFHead now returns psf_map and confidence_score
        predicted_psf_map, confidence_score = self.psf_head(decoder_output)

        return predicted_psf_map, confidence_score