import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock1D(in_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class Up1D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.reduce = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv = ConvBlock1D(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SinusoidalPositionalEncoding1D(nn.Module):
    def __init__(self, dim, max_length=4096):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_length, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        length = x.shape[1]
        if length > self.pe.shape[1]:
            raise ValueError(f"Token length {length} exceeds positional encoding length {self.pe.shape[1]}")
        return x + self.pe[:, :length, :]


class SpectrumUNetTransformer1D(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_length=2501,
        base_channels=32,
        trans_heads=4,
        trans_layers=2,
        dropout=0.1,
        output_activation="none",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_length = int(out_length)
        self.output_activation = str(output_activation).lower()

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8

        self.down1 = Down1D(self.in_channels, c1, dropout=0.0)
        self.down2 = Down1D(c1, c2, dropout=0.0)
        self.down3 = Down1D(c2, c3, dropout=0.0)

        self.bottleneck = ConvBlock1D(c3, c4, dropout=dropout)
        self.positional_encoding = SinusoidalPositionalEncoding1D(c4, max_length=4096)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c4,
            nhead=int(trans_heads),
            dim_feedforward=4 * c4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(trans_layers))

        self.up3 = Up1D(c4, c3, c3, dropout=0.0)
        self.up2 = Up1D(c3, c2, c2, dropout=0.0)
        self.up1 = Up1D(c2, c1, c1, dropout=0.0)
        self.head = nn.Conv1d(c1, 1, kernel_size=1)

    def _apply_output_activation(self, x):
        if self.output_activation in ("none", "", "null"):
            return x
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "relu":
            return F.relu(x)
        raise ValueError(f"Unsupported output_activation: {self.output_activation}")

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, C, L], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")

        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)

        x = self.bottleneck(x)
        tokens = x.transpose(1, 2)
        tokens = self.positional_encoding(tokens)
        tokens = self.transformer(tokens)
        x = tokens.transpose(1, 2)

        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        x = F.interpolate(x, size=self.out_length, mode="linear", align_corners=False)
        x = self.head(x)
        x = self._apply_output_activation(x)
        return x.squeeze(1)


if __name__ == "__main__":
    model = SpectrumUNetTransformer1D()
    sample = torch.randn(2, 4, 2501)
    output = model(sample)
    print("input_shape:", tuple(sample.shape))
    print("output_shape:", tuple(output.shape))
