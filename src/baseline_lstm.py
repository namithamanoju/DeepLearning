#!/usr/bin/env python3
"""
Baseline Vanilla LSTM for Spatio-Temporal Traffic Forecasting
Capstone Project: Deep Learning for Traffic Forecasting
Workspace: DL_PROJECT

This module defines the VanillaLSTM architecture:
- Multi-layer recurrent neural network (nn.LSTM) capturing temporal traffic dynamics.
- Linear projection layer mapping encoded hidden states to multi-horizon future flow predictions.
- Accepts input sequences:
    [Batch_Size, Time_Steps, Channels] (e.g., [64, 12, 921])
    or [Batch_Size, Time_Steps, Num_Sensors, Features] (e.g., [64, 12, 307, 3])
- Output shape: [Batch_Size, 12, 307, 1] (Target: Flow for 12 future timesteps)
"""

import sys
import torch
import torch.nn as nn

# Ensure safe console stdout encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class VanillaLSTM(nn.Module):
    """
    Standard Baseline LSTM for Multi-Step Ahead Traffic Flow Forecasting.

    Architecture:
      1. Sequence Input: [Batch_Size, Time_Steps, Channels]
      2. 2-Layer Recurrent Unit: nn.LSTM(hidden_size=64, num_layers=2)
      3. Fully Connected Projection: nn.Linear(hidden_size=64, out_steps * num_sensors * 1)
      4. Output Reshape: [Batch_Size, out_steps, num_sensors, 1]
    """
    def __init__(
        self,
        input_dim: int = 307 * 3,   # Channels = num_sensors (307) * features (3) = 921
        hidden_size: int = 64,
        num_layers: int = 2,
        out_steps: int = 12,
        num_sensors: int = 307,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_steps = out_steps
        self.num_sensors = num_sensors

        # 2-layer LSTM with batch_first=True -> input shape: [Batch, Time_Steps, Channels]
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Linear projection to forecast future Flow for all 307 sensors across 12 timesteps
        self.fc = nn.Linear(hidden_size, out_steps * num_sensors * 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [Batch_Size, Time_Steps, Channels]
               or [Batch_Size, Time_Steps, Num_Sensors, Features].

        Returns:
            Tensor of shape [Batch_Size, 12, 307, 1] containing Flow predictions.
        """
        # If input has 4 dimensions [Batch_Size, Time_Steps, Num_Sensors, Features],
        # flatten the spatial and feature dimensions into Channels: [Batch_Size, Time_Steps, Channels]
        if x.dim() == 4:
            batch_size, time_steps = x.size(0), x.size(1)
            x = x.reshape(batch_size, time_steps, -1)
        elif x.dim() == 3:
            batch_size = x.size(0)
        else:
            raise ValueError(f"Expected 3D or 4D input tensor, got shape: {x.shape}")

        # LSTM forward pass -> lstm_out: [Batch_Size, Time_Steps, hidden_size]
        lstm_out, _ = self.lstm(x)

        # Extract context representation from the last historical timestep (t=12)
        last_hidden = lstm_out[:, -1, :]  # [Batch_Size, hidden_size]

        # Project to future multi-step traffic flow horizon
        out = self.fc(last_hidden)  # [Batch_Size, out_steps * num_sensors * 1]

        # Reshape to target shape [Batch_Size, 12, 307, 1]
        out = out.view(batch_size, self.out_steps, self.num_sensors, 1)
        return out


if __name__ == "__main__":
    print("=" * 80)
    print("VanillaLSTM Module Smoke Test & Shape Verification")
    print("=" * 80)

    model = VanillaLSTM(
        input_dim=307 * 3,
        hidden_size=64,
        num_layers=2,
        out_steps=12,
        num_sensors=307
    )
    print(f"Model Summary:\n{model}\n")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total Trainable Parameters: {total_params:,}")

    # Test 1: 3D input tensor [Batch_Size, Time_Steps, Channels]
    batch_size = 32
    time_steps = 12
    channels = 307 * 3
    dummy_3d = torch.randn(batch_size, time_steps, channels)
    out_3d = model(dummy_3d)
    print(f"[+] Test 1 (3D Input [32, 12, 921]): Output Shape = {out_3d.shape}")
    assert out_3d.shape == torch.Size([batch_size, 12, 307, 1]), "Shape mismatch for 3D input!"

    # Test 2: 4D input tensor [Batch_Size, Time_Steps, Sensors, Features]
    dummy_4d = torch.randn(batch_size, time_steps, 307, 3)
    out_4d = model(dummy_4d)
    print(f"[+] Test 2 (4D Input [32, 12, 307, 3]): Output Shape = {out_4d.shape}")
    assert out_4d.shape == torch.Size([batch_size, 12, 307, 1]), "Shape mismatch for 4D input!"

    print("\n[✓] All VanillaLSTM model shape assertions PASSED successfully!")
