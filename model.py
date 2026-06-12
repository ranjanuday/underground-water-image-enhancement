import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
#  UTILITIES & BASIC BLOCKS
# =========================================================
class ConvBlock(nn.Module):
    """
    Standard feature extractor: Convolution -> Instance Normalization -> LeakyReLU.
    Bias is False because the subsequent Normalization layer cancels it out.
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # input: [in_c, H, W]
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False), # [out_c, H/stride, W/stride]
            nn.InstanceNorm2d(out_c, affine=True),            # [out_c, H/stride, W/stride]
            nn.LeakyReLU(0.2, inplace=True)                   # [out_c, H/stride, W/stride]
        )
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    """
    2-layer Residual Block to prevent vanishing gradients. 
    Learns the difference (residual) rather than the direct mapping.
    """
    def __init__(self, c):
        super().__init__()
        # input: [c, H, W]
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)     # [c, H, W]
        self.in1 = nn.InstanceNorm2d(c, affine=True)          
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)     # [c, H, W]
        self.in2 = nn.InstanceNorm2d(c, affine=True)          
        
    def forward(self, x):
        r = F.leaky_relu(self.in1(self.conv1(x)), 0.2, inplace=True)
        r = self.in2(self.conv2(r))
        return F.leaky_relu(x + r, 0.2, inplace=True)         # x + r forms the skip connection

# =========================================================
#  BRANCH 1: MULTI-SCALE MAIN BRANCH (Massive PSNR Boost)
# =========================================================
class UNetBranch(nn.Module):
    """
    A U-Net topology that captures local details (texture) and global context 
    (illumination) by downsampling and upsampling. Predicts a residual map.
    """
    def __init__(self, nf=32):
        super().__init__()
        # --- Encoder (Downsampling) ---
        self.enc1 = ConvBlock(3, nf)                          # Out: [nf, H, W]
        self.enc2 = ConvBlock(nf, nf * 2, stride=2)           # Out: [nf*2, H/2, W/2]
        self.enc3 = ConvBlock(nf * 2, nf * 4, stride=2)       # Out: [nf*4, H/4, W/4]
        
        # --- Bottleneck ---
        self.bottle = nn.Sequential(ResBlock(nf * 4), ResBlock(nf * 4)) # Out: [nf*4, H/4, W/4]
        
        # --- Decoder (Upsampling + Skip Connections) ---
        self.up2 = nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1)# Upsample to 1/2 scale
        self.dec2 = ConvBlock(nf * 4, nf * 2)                 # in_c is nf*4 due to concat with enc2
        
        self.up1 = nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1)    # Upsample to 1x scale
        self.dec1 = ConvBlock(nf * 2, nf)                     # in_c is nf*2 due to concat with enc1
        
        self.out_conv = nn.Conv2d(nf, 3, 3, 1, 1)             # Back to 3 RGB channels

    def forward(self, x):
        # Downward path
        e1 = self.enc1(x)                                     # [B, 32, 256, 256]
        e2 = self.enc2(e1)                                    # [B, 64, 128, 128]
        e3 = self.enc3(e2)                                    # [B, 128, 64, 64]
        
        # Bottleneck
        b = self.bottle(e3)                                   # [B, 128, 64, 64]
        
        # Upward path with skip connections
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))   # [B, 64, 128, 128]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 32, 256, 256]
        
        # Residual learning: Predict the difference, limit to [-1, 1] using tanh
        r = torch.tanh(self.out_conv(d1))                     # [B, 3, 256, 256]
        return torch.clamp(x + r, 0, 1)                       # Final UNet Out: [B, 3, 256, 256]

# =========================================================
#  BRANCH 2: STABILIZED DEHAZE BRANCH
# =========================================================
class DehazeBranch(nn.Module):
    """
    Solves the atmospheric scattering model: J(x) = (I(x) - A) / t(x) + A
    Predicts Transmission (t) and Ambient Light (A) directly via convolutions.
    """
    def __init__(self, nf=32):
        super().__init__()
        # Extracts features for global ambient light estimation
        self.net = nn.Sequential(
            ConvBlock(3, nf),
            ResBlock(nf),
            nn.AdaptiveAvgPool2d(1)                           # Collapse to [nf, 1, 1] for global context
        )
        
        # Predicts the 1-channel spatial Transmission Map (t)
        self.t_conv = nn.Sequential(
            nn.Conv2d(3, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, 1, 3, 1, 1),
            nn.Sigmoid()                                      # Bound to [0, 1]
        )
        
        # Fully connected layers to predict 3-channel global Ambient Light (A)
        self.A_fc = nn.Sequential(
            nn.Linear(nf, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
            nn.Sigmoid()                                      # Bound to [0, 1]
        )

    def forward(self, x):
        # 1. Predict Transmission Map (t)
        t = self.t_conv(x)                                    # [B, 1, 256, 256]
        t = torch.clamp(t, min=0.1, max=0.95)                 # Prevent division by zero
        
        # 2. Predict Ambient Light (A)
        feat = self.net(x).view(x.size(0), -1)                # [B, 32] (flattened)
        A = self.A_fc(feat).view(x.size(0), 3, 1, 1)          # [B, 3, 1, 1] (reshaped for broadcasting)
        
        # 3. Apply physical dehazing equation with safety epsilon
        J = (x - A) / (t + 1e-6) + A                          # [B, 3, 256, 256]
        
        return torch.clamp(J, 0, 1), t, A                     # Final Dehaze Out: [B, 3, 256, 256]

# =========================================================
#  BRANCH 3: COLOR CURVE BRANCH
# =========================================================
class ColorCurveBranch(nn.Module):
    """
    Predicts parameter maps for a pixel-wise quadratic curve: f(x) = x + alpha * x * (1 - x)
    Highly stable, differentiable alternative to complex color space conversions.
    """
    def __init__(self, nf=32):
        super().__init__()
        self.curve_net = nn.Sequential(
            nn.Conv2d(3, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, 3, 3, 1, 1),                        # Predicts 3 curve parameters per pixel
            nn.Tanh()                                         # Alpha bounded between [-1, 1]
        )

    def forward(self, x):
        alpha = self.curve_net(x)                             # [B, 3, 256, 256]
        out = x + alpha * x * (1 - x)                         # [B, 3, 256, 256]
        return torch.clamp(out, 0, 1)                         # Final Color Out: [B, 3, 256, 256]

# =========================================================
#  SPATIAL FUSION
# =========================================================
class SpatialFusion(nn.Module):
    """
    Attention mechanism that predicts per-pixel blending weights for the 3 branches,
    dynamically favoring the best branch for specific image regions.
    """
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(3 * 3, 32, 3, 1, 1),                    # 9 input channels (3 branches * 3 RGB)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, 3, 1, 1)                         # 3 output channels (1 weight map per branch)
        )

    def forward(self, branches):
        # Concatenate outputs from UNet, Dehaze, and ColorCurve
        concat = torch.cat(branches, dim=1)                   # [B, 9, 256, 256]
        
        # Softmax forces weights of the 3 branches to sum to 1.0 at every single pixel
        weights = F.softmax(self.attention(concat), dim=1)    # [B, 3, 256, 256]
        
        # Perform element-wise weighted sum of the 3 branch outputs
        out = (weights[:, 0:1] * branches[0] +                # [B, 1, H, W] * [B, 3, H, W]
               weights[:, 1:2] * branches[1] + 
               weights[:, 2:3] * branches[2])
        
        return torch.clamp(out, 0, 1)                         # Final Fused Out: [B, 3, 256, 256]

# =========================================================
#  FINAL MODEL
# =========================================================
class UIE_Net(nn.Module):
    """
    Coordinates the parallel execution of the 3 specialized branches and fuses them.
    """
    def __init__(self):
        super().__init__()
        self.main_unet = UNetBranch(nf=32)
        self.dehaze = DehazeBranch(nf=32)
        self.color = ColorCurveBranch(nf=32)
        self.fuse = SpatialFusion()

    def forward(self, x):
        unet_out = self.main_unet(x)
        dehaze_out, _, _ = self.dehaze(x)
        color_out = self.color(x)

        return self.fuse([unet_out, dehaze_out, color_out])

    def forward_with_extras(self, x):
        # Same as forward, but exposes t_map and A for loss/regularization functions
        unet_out = self.main_unet(x)
        dehaze_out, t_map, A = self.dehaze(x)
        color_out = self.color(x)

        out = self.fuse([unet_out, dehaze_out, color_out])
        return out, t_map, A

# =========================================================
#  SANITY CHECK
# =========================================================
if __name__ == "__main__":
    model = UIE_Net()
    x = torch.rand(2, 3, 256, 256)

    with torch.no_grad():
        y, t, b = model.forward_with_extras(x)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded successfully!")
    print(f"Parameters: {n_params:,}")
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")