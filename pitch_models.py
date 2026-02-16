"""
PyTorch Model Architectures for Pitch Anomaly Detection

Contains 4 autoencoder variants for time-series reconstruction:
- CAE1D: Baseline convolutional autoencoder
- TCNAE: Temporal Convolutional Network Autoencoder
- UNetAE: U-Net with skip connections
- TCNUNet: Combined TCN + U-Net architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualBlock(nn.Module):
    """
    A single dilated convolutional block with residual connection.

    Dilated Convolution Explanation:
    - Standard conv (d=1): kernel looks at consecutive elements [x1, x2, x3]
    - Dilated conv (d=2): kernel skips elements [x1, _, x3, _, x5]
    - This expands receptive field without adding parameters

    Receptive Field = 1 + (kernel_size - 1) * dilation
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection (1x1 conv if channel mismatch)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        # Align lengths if needed (due to dilation edge effects)
        if out.size(-1) != residual.size(-1):
            min_len = min(out.size(-1), residual.size(-1))
            out = out[..., :min_len]
            residual = residual[..., :min_len]

        out = self.relu(out + residual)
        return out


class CAE1D(nn.Module):
    """
    Baseline 1D Convolutional Autoencoder.

    Architecture: Simple encoder-decoder with pooling/upsampling.
    Receptive Field: ~30 timesteps
    """
    def __init__(self, in_channels: int, latent_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        def conv_block(cin, cout, k):
            pad = (k - 1) // 2
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, stride=1, padding=pad),
                nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
                nn.ReLU(inplace=True),
            )

        def up_block(cin, cout, k):
            pad = (k - 1) // 2
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(cin, cout, kernel_size=k, stride=1, padding=pad),
                nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 32, k=7)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.enc2 = conv_block(32, 64, k=5)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.enc3 = conv_block(64, 128, k=5)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.enc4 = conv_block(128, latent_dim, k=3)
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Decoder
        self.dec1 = up_block(latent_dim, 128, k=3)
        self.dec2 = up_block(128, 64, k=5)
        self.dec3 = up_block(64, 32, k=5)
        self.dec4 = up_block(32, in_channels, k=7)
        self.out_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def encoder(self, x):
        x = self.pool1(self.enc1(x))
        x = self.dropout(x)
        x = self.pool2(self.enc2(x))
        x = self.dropout(x)
        x = self.pool3(self.enc3(x))
        x = self.pool4(self.enc4(x))
        return x

    def decoder(self, z, target_T: int):
        x = self.dec1(z)
        x = self.dropout(x)
        x = self.dec2(x)
        x = self.dropout(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.out_conv(x)
        T = x.size(-1)
        if T > target_T:
            x = x[..., :target_T]
        elif T < target_T:
            x = F.pad(x, (0, target_T - T))
        return x

    def forward(self, x):
        T_target = x.size(-1)
        z = self.encoder(x)
        x_hat = self.decoder(z, target_T=T_target)
        return x_hat, z


class TCNAE(nn.Module):
    """
    Temporal Convolutional Network Autoencoder.

    Architecture:
    - Encoder: 4 dilated residual blocks with dilation rates [1, 2, 4, 8]
    - Bottleneck: Average pooling + 1x1 conv compression
    - Decoder: Mirror structure with upsampling

    Receptive Field: ~91 timesteps
    """
    def __init__(self, in_channels: int, latent_dim: int = 64, dropout: float = 0.2):
        super().__init__()

        # Encoder: dilated residual blocks
        self.enc1 = DilatedResidualBlock(in_channels, 32, kernel_size=7, dilation=1, dropout=dropout)
        self.enc2 = DilatedResidualBlock(32, 64, kernel_size=7, dilation=2, dropout=dropout)
        self.enc3 = DilatedResidualBlock(64, 128, kernel_size=7, dilation=4, dropout=dropout)
        self.enc4 = DilatedResidualBlock(128, 128, kernel_size=7, dilation=8, dropout=dropout)

        # Bottleneck
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.bottleneck_down = nn.Conv1d(128, latent_dim, 1)
        self.bottleneck_up = nn.Conv1d(latent_dim, 128, 1)

        # Decoder: mirror dilated residual blocks
        self.dec4 = DilatedResidualBlock(128, 128, kernel_size=7, dilation=8, dropout=dropout)
        self.dec3 = DilatedResidualBlock(128, 64, kernel_size=7, dilation=4, dropout=dropout)
        self.dec2 = DilatedResidualBlock(64, 32, kernel_size=7, dilation=2, dropout=dropout)
        self.dec1 = DilatedResidualBlock(32, in_channels, kernel_size=7, dilation=1, dropout=dropout)

        self.out_conv = nn.Conv1d(in_channels, in_channels, 1)

    def encoder(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.pool(x)
        x = self.bottleneck_down(x)
        return x

    def decoder(self, z, target_T: int):
        x = self.bottleneck_up(z)
        x = F.interpolate(x, size=target_T, mode='nearest')
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.out_conv(x)

        T = x.size(-1)
        if T > target_T:
            x = x[..., :target_T]
        elif T < target_T:
            x = F.pad(x, (0, target_T - T))
        return x

    def forward(self, x):
        T_target = x.size(-1)
        z = self.encoder(x)
        x_hat = self.decoder(z, target_T=T_target)
        return x_hat, z


class UNetAE(nn.Module):
    """
    U-Net style Autoencoder with Skip Connections.

    Skip connections provide "shortcuts" for high-frequency information,
    bypassing the bottleneck to preserve fine-grained details.

    Receptive Field: ~60 timesteps
    """
    def __init__(self, in_channels: int, latent_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        def conv_block(cin, cout, k):
            pad = (k - 1) // 2
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=pad),
                nn.GroupNorm(min(8, cout), cout),
                nn.ReLU(inplace=True),
                nn.Conv1d(cout, cout, kernel_size=k, padding=pad),
                nn.GroupNorm(min(8, cout), cout),
                nn.ReLU(inplace=True),
            )

        # Encoder path
        self.enc1 = conv_block(in_channels, 32, k=7)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = conv_block(32, 64, k=5)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = conv_block(64, 128, k=5)
        self.pool3 = nn.MaxPool1d(2)
        self.enc4 = conv_block(128, 256, k=3)
        self.pool4 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(256, latent_dim, 1),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_up = nn.Conv1d(latent_dim, 256, 1)

        # Decoder path (input channels = encoder channels + skip channels)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec4 = conv_block(256 + 256, 128, k=3)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = conv_block(128 + 128, 64, k=5)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = conv_block(64 + 64, 32, k=5)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = conv_block(32 + 32, 32, k=7)

        self.out_conv = nn.Conv1d(32, in_channels, 1)

    def encoder(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        z = self.bottleneck(p4)

        return z, (e1, e2, e3, e4)

    def decoder(self, z, skips, target_T: int):
        e1, e2, e3, e4 = skips

        x = self.bottleneck_up(z)

        x = self.up4(x)
        x = self._match_and_concat(x, e4)
        x = self.dec4(x)
        x = self.dropout(x)

        x = self.up3(x)
        x = self._match_and_concat(x, e3)
        x = self.dec3(x)
        x = self.dropout(x)

        x = self.up2(x)
        x = self._match_and_concat(x, e2)
        x = self.dec2(x)

        x = self.up1(x)
        x = self._match_and_concat(x, e1)
        x = self.dec1(x)

        x = self.out_conv(x)

        T = x.size(-1)
        if T > target_T:
            x = x[..., :target_T]
        elif T < target_T:
            x = F.pad(x, (0, target_T - T))
        return x

    def _match_and_concat(self, x, skip):
        """Match temporal dimensions and concatenate along channel axis."""
        if x.size(-1) != skip.size(-1):
            min_len = min(x.size(-1), skip.size(-1))
            x = x[..., :min_len]
            skip = skip[..., :min_len]
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        T_target = x.size(-1)
        z, skips = self.encoder(x)
        x_hat = self.decoder(z, skips, target_T=T_target)
        return x_hat, z


class TCNUNet(nn.Module):
    """
    Combined TCN + U-Net Autoencoder.

    Combines:
    1. TCN's dilated convolutions: Large receptive field (~91 timesteps)
    2. U-Net's skip connections: Preserve fine-grained reconstruction details
    """
    def __init__(self, in_channels: int, latent_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Encoder: TCN blocks with increasing dilation
        self.enc1 = DilatedResidualBlock(in_channels, 32, kernel_size=7, dilation=1, dropout=dropout)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = DilatedResidualBlock(32, 64, kernel_size=5, dilation=2, dropout=dropout)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = DilatedResidualBlock(64, 128, kernel_size=5, dilation=4, dropout=dropout)
        self.pool3 = nn.MaxPool1d(2)

        self.enc4 = DilatedResidualBlock(128, 256, kernel_size=3, dilation=8, dropout=dropout)
        self.pool4 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck_down = nn.Sequential(
            nn.Conv1d(256, latent_dim, 1),
            nn.GroupNorm(min(8, latent_dim), latent_dim),
            nn.ReLU(inplace=True),
        )
        self.bottleneck_up = nn.Sequential(
            nn.Conv1d(latent_dim, 256, 1),
            nn.GroupNorm(min(8, 256), 256),
            nn.ReLU(inplace=True),
        )

        # Decoder: TCN blocks with skip concatenation
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec4 = DilatedResidualBlock(256 + 256, 128, kernel_size=3, dilation=8, dropout=dropout)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = DilatedResidualBlock(128 + 128, 64, kernel_size=5, dilation=4, dropout=dropout)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = DilatedResidualBlock(64 + 64, 32, kernel_size=5, dilation=2, dropout=dropout)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = DilatedResidualBlock(32 + 32, 32, kernel_size=7, dilation=1, dropout=dropout)

        self.out_conv = nn.Conv1d(32, in_channels, 1)

    def encoder(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        z = self.bottleneck_down(p4)

        return z, (e1, e2, e3, e4)

    def decoder(self, z, skips, target_T: int):
        e1, e2, e3, e4 = skips

        x = self.bottleneck_up(z)

        x = self.up4(x)
        x = self._match_and_concat(x, e4)
        x = self.dec4(x)

        x = self.up3(x)
        x = self._match_and_concat(x, e3)
        x = self.dec3(x)

        x = self.up2(x)
        x = self._match_and_concat(x, e2)
        x = self.dec2(x)

        x = self.up1(x)
        x = self._match_and_concat(x, e1)
        x = self.dec1(x)

        x = self.out_conv(x)

        T = x.size(-1)
        if T > target_T:
            x = x[..., :target_T]
        elif T < target_T:
            x = F.pad(x, (0, target_T - T))
        return x

    def _match_and_concat(self, x, skip):
        """Match temporal dimensions and concatenate along channel axis."""
        if x.size(-1) != skip.size(-1):
            min_len = min(x.size(-1), skip.size(-1))
            x = x[..., :min_len]
            skip = skip[..., :min_len]
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        T_target = x.size(-1)
        z, skips = self.encoder(x)
        x_hat = self.decoder(z, skips, target_T=T_target)
        return x_hat, z
