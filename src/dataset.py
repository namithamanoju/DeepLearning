#!/usr/bin/env python3
"""
PeMS04 Spatio-Temporal Traffic Dataset and DataLoader
Capstone Project: Deep Learning for Traffic Forecasting
Workspace: DL_PROJECT

This module implements:
1. Chronological Train (70%) / Validation (10%) / Test (20%) splitting without data leakage.
2. Z-Score Standardization (StandardScaler) fitted strictly on the Training set.
3. Sliding window sequence construction:
   - X: Past 12 timesteps (1 hour lookback), shape [12, 307, 3] (Flow, Occupancy, Speed)
   - Y: Future 12 timesteps (1 hour horizon), shape [12, 307, 1] (Target: Flow only)
4. PyTorch Dataset and DataLoaders factory function `get_dataloaders(batch_size=64)`.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure safe console stdout encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class StandardScaler:
    """
    Z-Score Standardization: (x - mean) / std.
    Fitted exclusively on training data to prevent lookahead/snooping leakage.
    Supports multi-channel broadcasting and inverse transformation.
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        # Avoid zero-division if a sensor has constant reading
        self.std = np.where(std == 0.0, 1.0, std).astype(np.float32)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Standardize raw data."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Convert standardized data back to original physical scale."""
        return (data * self.std) + self.mean

    def inverse_transform_target(self, target_data: np.ndarray, channel_idx: int = 0) -> np.ndarray:
        """
        Inverse-transform the target variable (Flow channel at channel_idx=0).
        Accepts arrays/tensors of shape [..., 1].
        """
        target_mean = self.mean[..., channel_idx:channel_idx + 1]
        target_std = self.std[..., channel_idx:channel_idx + 1]
        return (target_data * target_std) + target_mean


class PeMSTrafficDataset(Dataset):
    """
    Sliding Window Spatio-Temporal Dataset for PyTorch.

    Each sample consists of:
      - X: Past `in_steps` observations for all sensors and all feature channels.
           Shape: [in_steps, num_sensors, num_features] -> [12, 307, 3]
      - Y: Future `out_steps` ground-truth for all sensors for the target channel (Flow).
           Shape: [out_steps, num_sensors, 1] -> [12, 307, 1]
    """
    def __init__(
        self,
        data: np.ndarray,
        in_steps: int = 12,
        out_steps: int = 12,
        target_channel: int = 0
    ):
        """
        Args:
            data: Standardized array of shape [T, num_sensors, num_features].
            in_steps: Historical window length (default: 12 steps = 1 hr).
            out_steps: Forecasting horizon length (default: 12 steps = 1 hr).
            target_channel: Channel index to predict (0 = Flow).
        """
        super().__init__()
        self.data = data.astype(np.float32)
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.target_channel = target_channel
        self.total_timesteps = len(self.data)
        self.seq_len = in_steps + out_steps

        if self.total_timesteps < self.seq_len:
            raise ValueError(
                f"Data length ({self.total_timesteps}) is shorter than sequence length ({self.seq_len})."
            )

        # Total valid sliding window slices
        self.num_samples = self.total_timesteps - self.seq_len + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: Past `in_steps` timesteps, all sensors, all features [12, 307, 3]
        x_slice = self.data[idx : idx + self.in_steps]

        # Target: Future `out_steps` timesteps, all sensors, Flow channel only [12, 307, 1]
        target_start = idx + self.in_steps
        target_end = target_start + self.out_steps
        y_slice = self.data[target_start:target_end, :, self.target_channel : self.target_channel + 1]

        x_tensor = torch.from_numpy(x_slice).float()
        y_tensor = torch.from_numpy(y_slice).float()

        return x_tensor, y_tensor


def load_and_preprocess_pems04(
    data_path: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    in_steps: int = 12,
    out_steps: int = 12,
    target_channel: int = 0
) -> Tuple[PeMSTrafficDataset, PeMSTrafficDataset, PeMSTrafficDataset, StandardScaler]:
    """
    Loads raw PEMS04.npz, applies strict chronological split, computes Z-Score
    normalization parameters on training data, and returns train/val/test Datasets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Default file path resolution
    if data_path is None:
        project_root = Path(__file__).resolve().parent.parent
        data_path = project_root / "data" / "PEMS04.npz"
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"PEMS04 dataset file not found at: {data_path.resolve()}")

    # 1. Load raw tensor: expected shape [16992, 307, 3]
    raw_npz = np.load(data_path)
    data = raw_npz["data"]  # [Time Steps, Sensors, Features]
    total_timesteps, num_sensors, num_features = data.shape

    # 2. Chronological Split (STRICT NO-SHUFFLE)
    train_end = int(total_timesteps * train_ratio)
    val_end = train_end + int(total_timesteps * val_ratio)

    train_raw = data[:train_end]
    val_raw = data[train_end:val_end]
    test_raw = data[val_end:]

    # 3. Z-Score Normalization (Fitted strictly on Training set)
    # Compute mean & std per feature channel across all (timesteps, sensors)
    # Mean and Std shape: [1, 1, num_features]
    train_mean = np.mean(train_raw, axis=(0, 1), keepdims=True)
    train_std = np.std(train_raw, axis=(0, 1), keepdims=True)

    scaler = StandardScaler(mean=train_mean, std=train_std)

    # Standardize all splits using training statistics
    train_norm = scaler.transform(train_raw)
    val_norm = scaler.transform(val_raw)
    test_norm = scaler.transform(test_raw)

    # 4. Construct Datasets
    train_dataset = PeMSTrafficDataset(
        data=train_norm,
        in_steps=in_steps,
        out_steps=out_steps,
        target_channel=target_channel
    )
    val_dataset = PeMSTrafficDataset(
        data=val_norm,
        in_steps=in_steps,
        out_steps=out_steps,
        target_channel=target_channel
    )
    test_dataset = PeMSTrafficDataset(
        data=test_norm,
        in_steps=in_steps,
        out_steps=out_steps,
        target_channel=target_channel
    )

    return train_dataset, val_dataset, test_dataset, scaler


def get_dataloaders(
    batch_size: int = 64,
    data_path: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    in_steps: int = 12,
    out_steps: int = 12,
    target_channel: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Creates and returns PyTorch DataLoaders for Train, Validation, and Test sets,
    along with the fitted StandardScaler.

    Args:
        batch_size: Mini-batch size (default: 64).
        data_path: Optional custom path to PEMS04.npz.
        train_ratio: Fraction for training (default: 0.70).
        val_ratio: Fraction for validation (default: 0.10).
        test_ratio: Fraction for testing (default: 0.20).
        in_steps: Historical window length (default: 12 = 1 hr).
        out_steps: Prediction horizon length (default: 12 = 1 hr).
        target_channel: Target feature index (0 = Flow).
        num_workers: Background worker subprocesses (default: 0 for Windows safety).
        pin_memory: Whether to pin memory for CUDA transfers (default: False).
        shuffle_train: Whether to shuffle training mini-batches (default: True).

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    train_set, val_set, test_set, scaler = load_and_preprocess_pems04(
        data_path=data_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        in_steps=in_steps,
        out_steps=out_steps,
        target_channel=target_channel
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    print("=" * 80)
    print("PeMS04 PyTorch Dataset & DataLoader Pipeline Verification")
    print("=" * 80)

    # 1. Instantiate DataLoaders
    batch_size = 64
    print(f"\n[*] Generating DataLoaders (batch_size={batch_size})...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(batch_size=batch_size)

    # 2. Print split diagnostics
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset)
    total_samples = train_samples + val_samples + test_samples

    print("\n[+] Chronological Split Diagnostics:")
    print(f"  - Training Sequences:   {train_samples:,} batches: {len(train_loader):,}")
    print(f"  - Validation Sequences: {val_samples:,} batches: {len(val_loader):,}")
    print(f"  - Testing Sequences:    {test_samples:,} batches: {len(test_loader):,}")
    print(f"  - Total Sequences:      {total_samples:,}")

    # 3. Print Normalization Parameters (computed strictly on train split)
    flow_mean = float(scaler.mean[0, 0, 0])
    flow_std = float(scaler.std[0, 0, 0])
    occ_mean = float(scaler.mean[0, 0, 1])
    occ_std = float(scaler.std[0, 0, 1])
    spd_mean = float(scaler.mean[0, 0, 2])
    spd_std = float(scaler.std[0, 0, 2])

    print("\n[+] Fitted StandardScaler Parameters (Training Set Only):")
    print(f"  - Feature 0 (Flow):      Mean = {flow_mean:8.2f} | Std = {flow_std:8.2f}")
    print(f"  - Feature 1 (Occupancy): Mean = {occ_mean:8.4f} | Std = {occ_std:8.4f}")
    print(f"  - Feature 2 (Speed):     Mean = {spd_mean:8.2f} | Std = {spd_std:8.2f}")

    # 4. Extract one single batch and inspect exact tensor shapes
    print("\n[*] Fetching one single batch from train_loader...")
    batch_x, batch_y = next(iter(train_loader))

    print("\n" + "=" * 80)
    print("EXACT TENSOR SHAPE VERIFICATION")
    print("=" * 80)
    print(f"• Input  Batch (X) Shape:  {batch_x.shape}")
    print(f"• Target Batch (Y) Shape:  {batch_y.shape}")
    print(f"• Single Sample (X) Shape: {batch_x[0].shape}  [12 timesteps, 307 sensors, 3 features]")
    print(f"• Single Target (Y) Shape: {batch_y[0].shape}  [12 timesteps, 307 sensors, 1 target (Flow)]")
    print(f"• X Data Type:             {batch_x.dtype}")
    print(f"• Y Data Type:             {batch_y.dtype}")
    print(f"• X Value Range:           [{batch_x.min():.4f}, {batch_x.max():.4f}] (Standardized)")
    print(f"• Y Value Range:           [{batch_y.min():.4f}, {batch_y.max():.4f}] (Standardized)")
    print("=" * 80)

    # Sanity assertion check
    assert batch_x.shape == torch.Size([batch_size, 12, 307, 3]), f"Unexpected X shape: {batch_x.shape}"
    assert batch_y.shape == torch.Size([batch_size, 12, 307, 1]), f"Unexpected Y shape: {batch_y.shape}"
    print("\n[✓] All shape and data integrity assertions PASSED successfully!")
