#!/usr/bin/env python3
"""
Proposed ConvLSTM + Multi-Head Attention Training & Evaluation Pipeline
Capstone Project: Deep Learning for Traffic Forecasting
Workspace: DL_PROJECT

This script:
1. Loads the preprocessed PeMS04 DataLoaders and StandardScaler.
2. Initializes the ConvLSTMAttention neural network.
3. Configures hardware acceleration (CUDA if available, with multi-threaded CPU fallback).
4. Trains using MSE loss and Adam optimizer (lr=1e-3) for up to 50 epochs.
5. Monitors validation loss and applies Early Stopping (patience=5).
6. Checkpoints best weights to 'models/proposed_conv_lstm_attention.pth'.
7. Evaluates on the Test set in physical vehicle flow units (veh/5min) and logs
   comprehensive metrics to 'models/proposed_metrics.json'.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Add project root and src to sys.path for robust imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Ensure safe console stdout encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optimize CPU parallelism if running without GPU
if not torch.cuda.is_available():
    cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(min(cpu_cores, 8))

# Import project modules
try:
    from dataset import get_dataloaders, StandardScaler
    from conv_lstm_attention import ConvLSTMAttention
except ImportError:
    from src.dataset import get_dataloaders, StandardScaler
    from src.conv_lstm_attention import ConvLSTMAttention


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    scaler: StandardScaler,
    mape_threshold: float = 5.0
) -> Dict[str, Any]:
    """
    Computes regression metrics in both normalized and unscaled physical domain (veh/5min).
    """
    # Normalized metrics
    norm_mse = float(np.mean((preds - targets) ** 2))
    norm_rmse = float(np.sqrt(norm_mse))
    norm_mae = float(np.mean(np.abs(preds - targets)))

    # Real-world physical scale (vehicles / 5 minutes)
    real_preds = scaler.inverse_transform_target(preds)
    real_targets = scaler.inverse_transform_target(targets)

    diff = real_preds - real_targets
    real_mse = float(np.mean(diff ** 2))
    real_rmse = float(np.sqrt(real_mse))
    real_mae = float(np.mean(np.abs(diff)))

    # Masked MAPE to avoid zero/near-zero denominator inflation
    mask = real_targets > mape_threshold
    real_mape = float(np.mean(np.abs(diff[mask] / real_targets[mask]))) * 100.0 if np.sum(mask) > 0 else 0.0

    # Horizon-specific metrics (15 min = step 3, 30 min = step 6, 60 min = step 12)
    horizon_metrics = {}
    steps = [2, 5, 11]  # 0-indexed indices for steps 3, 6, 12
    step_names = ["15_min_step3", "30_min_step6", "60_min_step12"]
    for idx, name in zip(steps, step_names):
        if preds.shape[1] > idx:
            h_diff = real_preds[:, idx] - real_targets[:, idx]
            h_mae = float(np.mean(np.abs(h_diff)))
            h_rmse = float(np.sqrt(np.mean(h_diff ** 2)))
            horizon_metrics[name] = {
                "mae": round(h_mae, 4),
                "rmse": round(h_rmse, 4)
            }

    return {
        "mae": round(real_mae, 4),
        "rmse": round(real_rmse, 4),
        "mape_percent": round(real_mape, 4),
        "normalized_mae": round(norm_mae, 6),
        "normalized_rmse": round(norm_rmse, 6),
        "normalized_mse": round(norm_mse, 6),
        "horizons": horizon_metrics
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Evaluate model on a DataLoader and return the average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            total_loss += loss.item()

    return total_loss / num_batches if num_batches > 0 else 0.0


def run_test_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device
) -> Dict[str, Any]:
    """Inference over entire test set to compute final benchmark metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())

    all_preds_arr = np.concatenate(all_preds, axis=0)
    all_targets_arr = np.concatenate(all_targets, axis=0)

    return compute_metrics(all_preds_arr, all_targets_arr, scaler)


def train_proposed_model(
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 5,
    model_save_path: Path = None,
    metrics_save_path: Path = None
):
    """Main training routine with early stopping and test evaluation."""
    if model_save_path is None:
        model_save_path = PROJECT_ROOT / "models" / "proposed_conv_lstm_attention.pth"
    if metrics_save_path is None:
        metrics_save_path = PROJECT_ROOT / "models" / "proposed_metrics.json"

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Hardware acceleration check (CUDA vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("PROPOSED CONVLSTM + MULTI-HEAD ATTENTION TRAINING PIPELINE")
    print("=" * 80)
    print(f"• Hardware Device:       {device} {f'({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else f'({torch.get_num_threads()} CPU threads)'}")
    print(f"• Maximum Epochs:        {epochs}")
    print(f"• Mini-Batch Size:       {batch_size}")
    print(f"• Initial Learning Rate: {learning_rate}")
    print(f"• Early Stopping:        Patience = {early_stopping_patience} epochs")
    print(f"• Model Weights Path:    {model_save_path.resolve()}")
    print(f"• Final Metrics Path:    {metrics_save_path.resolve()}")
    print("=" * 80)

    # 1. Load DataLoaders
    print("\n[*] Loading dataset and creating data loaders...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        batch_size=batch_size,
        data_path=str(PROJECT_ROOT / "data" / "PEMS04.npz"),
        shuffle_train=True
    )
    print(f"[+] Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    print(f"[+] Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")
    print(f"[+] Test:  {len(test_loader.dataset):,} samples ({len(test_loader)} batches)")

    # 2. Initialize Model Architecture
    model = ConvLSTMAttention(
        num_sensors=307,
        in_channels=3,
        spatial_channels=16,
        lstm_hidden=16,
        d_model=64,
        num_heads=4,
        out_steps=12,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Model initialized with {total_params:,} trainable parameters.")

    # 3. Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)

    # 4. Training Loop with Early Stopping
    print("\n[*] Starting training loop...")
    header = f"{'Epoch':<7} | {'Train MSE':<12} | {'Val MSE':<12} | {'LR':<10} | {'Epoch Time':<10} | {'Status'}"
    print("-" * 80)
    print(header)
    print("-" * 80)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    training_start_time = time.time()
    history = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()

            # Gradient clipping to stabilize recurrent and attention gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Improvement check
        status_msg = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "scaler_mean": scaler.mean,
                    "scaler_std": scaler.std
                },
                model_save_path
            )
            status_msg = ">>> Best Checkpoint Saved"
        else:
            patience_counter += 1
            status_msg = f"No improvement ({patience_counter}/{early_stopping_patience})"

        history.append({
            "epoch": epoch,
            "train_mse": round(train_loss, 6),
            "val_mse": round(val_loss, 6),
            "lr": current_lr,
            "epoch_time_sec": round(epoch_time, 2)
        })

        print(f"{epoch:02d}/{epochs:02d} | {train_loss:<12.6f} | {val_loss:<12.6f} | {current_lr:<10.6f} | {epoch_time:6.2f}s     | {status_msg}")

        # Early stopping trigger
        if patience_counter >= early_stopping_patience:
            print("-" * 80)
            print(f"[!] Early stopping triggered at Epoch {epoch}. Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.6f}")
            break

    total_training_time = time.time() - training_start_time
    print("-" * 80)
    print(f"[+] Training completed in {total_training_time:.2f} seconds ({total_training_time / 60.0:.2f} minutes).")

    # 5. Final Evaluation on Test Set using Best Checkpoint
    print("\n[*] Loading best checkpoint for Test set evaluation...")
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_test_evaluation(model, test_loader, scaler, device)

    # Compile final metrics report
    final_report = {
        "model": "ConvLSTMAttention",
        "dataset": "PEMS04",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_parameters": total_params,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "total_training_time_seconds": round(total_training_time, 2),
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "mape_percent": test_metrics["mape_percent"],
        "normalized_mae": test_metrics["normalized_mae"],
        "normalized_rmse": test_metrics["normalized_rmse"],
        "normalized_mse": test_metrics["normalized_mse"],
        "horizon_breakdown": test_metrics["horizons"]
    }

    # Save to JSON
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)

    print("\n" + "=" * 80)
    print("PROPOSED MODEL FINAL TEST SET BENCHMARK RESULTS")
    print("=" * 80)
    print(f"• Mean Absolute Error (MAE):       {final_report['mae']} veh/5min")
    print(f"• Root Mean Squared Error (RMSE):  {final_report['rmse']} veh/5min")
    print(f"• Mean Absolute Percentage Error:  {final_report['mape_percent']}%")
    print(f"• Normalized Test MSE:             {final_report['normalized_mse']}")
    print(f"• Normalized Test MAE:             {final_report['normalized_mae']}")
    print("\n  [Horizon Breakdown]")
    for h_name, h_vals in final_report["horizon_breakdown"].items():
        print(f"  - {h_name.replace('_', ' '):<20}: MAE = {h_vals['mae']:.2f} veh/5min, RMSE = {h_vals['rmse']:.2f} veh/5min")
    print("=" * 80)
    print(f"[+] Best model checkpoint saved to: {model_save_path.resolve()}")
    print(f"[+] Test metrics report saved to:    {metrics_save_path.resolve()}")
    print("=" * 80 + "\n")

    return final_report


if __name__ == "__main__":
    train_proposed_model(
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        early_stopping_patience=5
    )
