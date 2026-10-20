#!/usr/bin/env python3
"""
Proposed Spatio-Temporal Model: ConvLSTM + Multi-Head Attention
Capstone Project: Deep Learning for Traffic Forecasting
Workspace: DL_PROJECT

Architecture Overview:
1. Spatial Feature Extractor (1D Convolution):
   Extracts local spatial dependencies across the 307 road sensor network.
2. Spatio-Temporal Encoder (ConvLSTM):
   Performs recurrent temporal modeling directly over spatial feature maps using
   convolutional gating mechanisms.
3. Multi-Head Attention (MHA) Mechanism:
   Dynamically attends over the historical sequence to weight critical time steps
   (e.g., sudden traffic surges or congestion onset).
4. Fully Connected Decoder:
   Maps the attended spatio-temporal context representation into multi-step
   future traffic flow forecasts [Batch_Size, 12, 307, 1].
"""

import sys
from typing import Tuple, Optional

import torch
import torch.nn as nn

# Ensure safe console stdout encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class SpatialFeatureExtractor(nn.Module):
    """
    Stage 1: Spatial Feature Extraction.
    Applies 1D spatial convolutions over the sensor network dimension to capture
    local structural road interactions and correlations across traffic channels.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [Batch_Size, Time_Steps, Num_Sensors, In_Channels]
        Returns:
            Tensor of shape [Batch_Size, Time_Steps, Out_Channels, Num_Sensors]
        """
        B, T, N, C = x.shape
        # Permute to [B * T, C, N] for 1D convolution across sensors
        x_flat = x.permute(0, 1, 3, 2).reshape(B * T, C, N)
        spatial_feats = self.conv(x_flat)  # [B * T, Out_Channels, N]
        return spatial_feats.view(B, T, -1, N)


class ConvLSTM1D(nn.Module):
    """
    Stage 2: 1D Convolutional Long Short-Term Memory (ConvLSTM) Encoder.
    Extends standard LSTM by replacing matrix multiplications with spatial convolutions,
    preserving topological alignment and temporal memory across road sensors.
    """
    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 16,
        kernel_size: int = 3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Combined convolution for all 4 gates (input, forget, candidate, output)
        self.conv_x = nn.Conv1d(
            in_channels=in_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self.conv_h = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

    def forward(
        self,
        seq_in: torch.Tensor,
        init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            seq_in: Tensor of shape [Batch_Size, Time_Steps, In_Channels, Num_Sensors]
            init_states: Optional initial (h_0, c_0)
        Returns:
            Sequence of hidden states of shape [Batch_Size, Time_Steps, Hidden_Channels, Num_Sensors]
        """
        B, T, C, N = seq_in.shape
        device = seq_in.device

        # Precompute input gate convolutions across all timesteps simultaneously
        x_all = seq_in.view(B * T, C, N)
        wx_all = self.conv_x(x_all).view(B, T, 4 * self.hidden_channels, N)

        # Initialize hidden and cell states
        if init_states is None:
            h = torch.zeros(B, self.hidden_channels, N, device=device)
            c = torch.zeros(B, self.hidden_channels, N, device=device)
        else:
            h, c = init_states

        outputs = []
        for t in range(T):
            gates = wx_all[:, t] + self.conv_h(h)
            i, f, g, o = torch.chunk(gates, 4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h)

        return torch.stack(outputs, dim=1)  # [B, T, Hidden_Channels, N]


class MultiHeadTemporalAttention(nn.Module):
    """
    Stage 3: Multi-Head Self-Attention over the Temporal Horizon.
    Enables the network to dynamically assign query-key attention weights
    across historical timesteps, identifying peak transitions and anomalies.
    """
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Temporal sequence embeddings of shape [Batch_Size, Time_Steps, d_model]
        Returns:
            Context representations [Batch_Size, Time_Steps, d_model] and attention weights
        """
        attn_out, attn_weights = self.mha(query=x, key=x, value=x)
        # Residual connection + Layer Normalization
        out = self.norm(x + self.dropout(attn_out))
        return out, attn_weights


class ConvLSTMAttention(nn.Module):
    """
    End-to-End Spatio-Temporal Architecture:
    Spatial Feature Extractor -> ConvLSTM Encoder -> Multi-Head Attention -> Linear Decoder.
    """
    def __init__(
        self,
        num_sensors: int = 307,
        in_channels: int = 3,
        spatial_channels: int = 16,
        lstm_hidden: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        out_steps: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.out_steps = out_steps
        self.d_model = d_model

        # 1. Spatial feature extraction (1D Convolution)
        self.spatial = SpatialFeatureExtractor(
            in_channels=in_channels,
            out_channels=spatial_channels,
            kernel_size=3,
            dropout=dropout
        )

        # 2. Spatio-temporal recurrent memory (ConvLSTM)
        self.conv_lstm = ConvLSTM1D(
            in_channels=spatial_channels,
            hidden_channels=lstm_hidden,
            kernel_size=3
        )

        # 3. Projection to attention embedding space
        self.temporal_proj = nn.Linear(lstm_hidden * num_sensors, d_model)

        # 4. Multi-head temporal attention
        self.attention = MultiHeadTemporalAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # 5. Multi-step forecasting decoder head
        self.decoder = nn.Sequential(
            nn.Linear(12 * d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_steps * num_sensors * 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [Batch_Size, 12, 307, 3]
        Returns:
            Forecast tensor [Batch_Size, 12, 307, 1]
        """
        B = x.size(0)

        # Stage 1: Spatial convolution over sensors [B, 12, spatial_channels, 307]
        spatial_feats = self.spatial(x)

        # Stage 2: ConvLSTM temporal recurrence [B, 12, lstm_hidden, 307]
        lstm_seq = self.conv_lstm(spatial_feats)

        # Stage 3: Project and attend over time steps
        flat_seq = lstm_seq.reshape(B, 12, -1)  # [B, 12, lstm_hidden * num_sensors]
        emb = self.temporal_proj(flat_seq)      # [B, 12, d_model]
        attn_out, _ = self.attention(emb)       # [B, 12, d_model]

        # Stage 4: Decode attention context vector into multi-horizon forecasts
        context = attn_out.reshape(B, -1)       # [B, 12 * d_model]
        pred = self.decoder(context)            # [B, out_steps * num_sensors * 1]

        # Final reshape to target shape [Batch_Size, 12, 307, 1]
        return pred.view(B, self.out_steps, self.num_sensors, 1)


if __name__ == "__main__":
    print("=" * 80)
    print("ConvLSTM + Multi-Head Attention Architecture Smoke Test")
    print("=" * 80)

    model = ConvLSTMAttention(
        num_sensors=307,
        in_channels=3,
        spatial_channels=16,
        lstm_hidden=16,
        d_model=64,
        num_heads=4,
        out_steps=12
    )
    print(f"Model Architecture:\n{model}\n")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total Trainable Parameters: {total_params:,}")

    # Forward pass test with batch of shape [64, 12, 307, 3]
    dummy_input = torch.randn(64, 12, 307, 3)
    output = model(dummy_input)
    print(f"[+] Input Shape:  {dummy_input.shape}")
    print(f"[+] Output Shape: {output.shape}")

    assert output.shape == torch.Size([64, 12, 307, 1]), "Output shape mismatch!"
    print("\n[✓] All architecture assertions PASSED successfully!")
