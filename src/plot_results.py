#!/usr/bin/env python3
"""
Traffic Forecasting Model Evaluation & Visualization Suite
Capstone Project: Deep Learning for Traffic Forecasting
Workspace: DL_PROJECT
Module: src/plot_results.py

Role: Data Visualization Specialist

This script:
1. Loads pre-trained weights for:
   - Baseline Vanilla LSTM ('models/baseline_lstm.pth')
   - Proposed ConvLSTM + Multi-Head Attention ('models/proposed_conv_lstm_attention.pth')
2. Generates test set predictions in physical vehicle volume units (veh/5min).
3. Produces high-resolution defense-grade visualizations:
   - Plot 1: 24-Hour Continuous Traffic Flow Timeline (Sensor 111 Bottleneck)
     Saved to 'plots/traffic_prediction_timeline.png'
   - Plot 2: Residual Error Distribution & KDE Density Overlay
     Saved to 'plots/error_distribution.png'
4. Logs detailed visualization metrics and execution confirmation.
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add src and root to path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import project modules
from src.dataset import get_dataloaders, StandardScaler
from src.baseline_lstm import VanillaLSTM
from src.conv_lstm_attention import ConvLSTMAttention

PLOTS_DIR = PROJECT_ROOT / "plots"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_models_and_predict(
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the test set for both models and return physical scale predictions.
    Returns:
        targets_real: Ground-truth array of shape [N, 12, 307]
        baseline_real: Baseline LSTM predictions of shape [N, 12, 307]
        proposed_real: Proposed model predictions of shape [N, 12, 307]
    """
    baseline_path = MODELS_DIR / "baseline_lstm.pth"
    proposed_path = MODELS_DIR / "proposed_conv_lstm_attention.pth"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint missing: {baseline_path}")
    if not proposed_path.exists():
        raise FileNotFoundError(f"Proposed checkpoint missing: {proposed_path}")

    print("[*] Loading test dataloader and fitted StandardScaler...")
    _, _, test_loader, scaler = get_dataloaders(
        batch_size=64,
        data_path=str(PROJECT_ROOT / "data" / "PEMS04.npz"),
        shuffle_train=False
    )

    # Initialize models
    print("[*] Initializing Baseline LSTM architecture...")
    baseline_model = VanillaLSTM(
        input_dim=307 * 3,
        hidden_size=64,
        num_layers=2,
        out_steps=12,
        num_sensors=307
    ).to(device)
    b_ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
    baseline_model.load_state_dict(b_ckpt["model_state_dict"])
    baseline_model.eval()

    print("[*] Initializing Proposed ConvLSTM + Attention architecture...")
    proposed_model = ConvLSTMAttention(
        num_sensors=307,
        in_channels=3,
        spatial_channels=16,
        lstm_hidden=16,
        d_model=64,
        num_heads=4,
        out_steps=12,
        dropout=0.1
    ).to(device)
    p_ckpt = torch.load(proposed_path, map_location=device, weights_only=False)
    proposed_model.load_state_dict(p_ckpt["model_state_dict"])
    proposed_model.eval()

    print("[*] Performing test set inference for both models...")
    all_targets = []
    all_baseline_preds = []
    all_proposed_preds = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)

            # Inference
            b_out = baseline_model(bx)
            p_out = proposed_model(bx)

            all_targets.append(by.numpy())
            all_baseline_preds.append(b_out.cpu().numpy())
            all_proposed_preds.append(p_out.cpu().numpy())

    targets_norm = np.concatenate(all_targets, axis=0)           # [N, 12, 307, 1]
    baseline_norm = np.concatenate(all_baseline_preds, axis=0)    # [N, 12, 307, 1]
    proposed_norm = np.concatenate(all_proposed_preds, axis=0)    # [N, 12, 307, 1]

    # Invert Z-score normalization to recover physical vehicle flow (veh/5min)
    print("[*] Inverting normalization to raw vehicle flow scale (veh/5min)...")
    targets_real = scaler.inverse_transform_target(targets_norm, channel_idx=0).squeeze(-1)
    baseline_real = scaler.inverse_transform_target(baseline_norm, channel_idx=0).squeeze(-1)
    proposed_real = scaler.inverse_transform_target(proposed_norm, channel_idx=0).squeeze(-1)

    return targets_real, baseline_real, proposed_real


def plot_24h_timeline(
    targets: np.ndarray,
    baseline_preds: np.ndarray,
    proposed_preds: np.ndarray,
    sensor_id: int = 111,
    horizon_step: int = 0  # Step 0 = 5-min ahead forecast
) -> Path:
    """
    Plot 1: 24-Hour Continuous Traffic Flow Timeline comparing Ground Truth,
    Baseline LSTM, and Proposed Model on a high-traffic highway bottleneck.
    """
    save_path = PLOTS_DIR / "traffic_prediction_timeline.png"
    print(f"\n[*] Generating Plot 1: 24-Hour Traffic Flow Timeline (Sensor {sensor_id})...")

    # 288 steps = 24 hours of 5-minute sampling
    num_steps = 288
    time_indices = np.arange(num_steps)

    gt_series = targets[:num_steps, horizon_step, sensor_id]
    base_series = baseline_preds[:num_steps, horizon_step, sensor_id]
    prop_series = proposed_preds[:num_steps, horizon_step, sensor_id]

    # Metrics for this specific 24h window
    base_mae = np.mean(np.abs(gt_series - base_series))
    prop_mae = np.mean(np.abs(gt_series - prop_series))
    base_rmse = np.sqrt(np.mean((gt_series - base_series) ** 2))
    prop_rmse = np.sqrt(np.mean((gt_series - prop_series) ** 2))

    # Set aesthetics
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)

    # Plot curves with publication styling
    ax.plot(
        time_indices, gt_series,
        label="Ground Truth (Actual Flow)",
        color="#1A202C",
        linewidth=2.4,
        alpha=0.95,
        zorder=4
    )
    ax.plot(
        time_indices, base_series,
        label=f"Baseline Vanilla LSTM (MAE: {base_mae:.2f})",
        color="#E53E3E",
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        zorder=3
    )
    ax.plot(
        time_indices, prop_series,
        label=f"Proposed ConvLSTM + Attention (MAE: {prop_mae:.2f})",
        color="#2B6CB0",
        linestyle="-",
        linewidth=2.2,
        alpha=0.9,
        zorder=5
    )

    # Highlight rush hour intervals (Morning: 7 AM - 10 AM, Evening: 4 PM - 7:30 PM)
    # Morning: steps 84 to 120; Evening: steps 192 to 234
    ax.axvspan(84, 120, color="#ED8936", alpha=0.12, label="Morning Peak (07:00 - 10:00)")
    ax.axvspan(192, 234, color="#9F7AEA", alpha=0.12, label="Evening Peak (16:00 - 19:30)")

    # Title and Subtitles
    ax.set_title(
        f"PeMS04 Traffic Flow Forecasting: 24-Hour Continuous Timeline\n"
        f"High-Density Highway Sensor Node #{sensor_id} (Horizon: 15-Min Step Ahead)",
        fontsize=14,
        fontweight="bold",
        pad=15,
        color="#2D3748"
    )
    ax.set_xlabel("Time Step (5-min intervals) / Time of Day", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Traffic Flow (vehicles / 5 min)", fontsize=11, fontweight="bold", labelpad=10)

    # Format X-axis with both step numbers and 24h clock labels
    hour_ticks = np.arange(0, num_steps + 1, 24)  # Every 2 hours (24 * 5 min = 120 min)
    hour_labels = [f"Step {s}\n({s * 5 // 60:02d}:00)" for s in hour_ticks]
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels, fontsize=8.5)
    ax.set_xlim(0, num_steps - 1)

    # Y-axis styling
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    ax.tick_params(axis="both", which="major", labelsize=9.5)

    # Callout info box
    info_text = (
        f"Sensor ID: #{sensor_id}\n"
        f"Mean Flow: {np.mean(gt_series):.1f} veh/5min\n"
        f"Peak Flow: {np.max(gt_series):.1f} veh/5min\n"
        f"---------------------------\n"
        f"Baseline MAE:  {base_mae:.2f}\n"
        f"Proposed MAE:  {prop_mae:.2f} ({((base_mae - prop_mae)/base_mae)*100:+.1f}%)\n"
        f"Proposed RMSE: {prop_rmse:.2f} ({((base_rmse - prop_rmse)/base_rmse)*100:+.1f}%)"
    )
    ax.text(
        0.015, 0.95, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#CBD5E0", alpha=0.92)
    )

    # Legend
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=9.5)
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved Plot 1: {save_path.resolve()}")
    return save_path


def plot_residual_error_distribution(
    targets: np.ndarray,
    baseline_preds: np.ndarray,
    proposed_preds: np.ndarray
) -> Path:
    """
    Plot 2: Residual Error Distribution with KDE density overlay curves
    comparing Baseline LSTM vs. Proposed ConvLSTM-Attention across all test sensors.
    """
    save_path = PLOTS_DIR / "error_distribution.png"
    print("\n[*] Generating Plot 2: Residual Error Distribution & KDE Overlay...")

    # Compute residuals: Error = Ground Truth - Predicted
    baseline_errors = (targets - baseline_preds).flatten()
    proposed_errors = (targets - proposed_preds).flatten()

    # Subsample for smooth, performant KDE plotting if large
    if len(baseline_errors) > 200_000:
        np.random.seed(42)
        sample_idx = np.random.choice(len(baseline_errors), size=200_000, replace=False)
        b_sample = baseline_errors[sample_idx]
        p_sample = proposed_errors[sample_idx]
    else:
        b_sample = baseline_errors
        p_sample = proposed_errors

    # Compute key statistics
    b_mean = np.mean(baseline_errors)
    b_std = np.std(baseline_errors)
    b_mae = np.mean(np.abs(baseline_errors))

    p_mean = np.mean(proposed_errors)
    p_std = np.std(proposed_errors)
    p_mae = np.mean(np.abs(proposed_errors))

    # Set aesthetics
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    # Error clip range for clean visualization (e.g., -100 to +100 veh/5min covers >99% of distribution)
    clip_val = 100
    bins = np.linspace(-clip_val, clip_val, 120)

    # Plot Histograms and KDE
    sns.histplot(
        b_sample,
        bins=bins,
        stat="density",
        color="#E53E3E",
        alpha=0.3,
        label=f"Baseline LSTM Residuals (Std: {b_std:.2f})",
        ax=ax,
        edgecolor="none"
    )
    sns.kdeplot(
        b_sample,
        clip=(-clip_val, clip_val),
        color="#C53030",
        linewidth=2.2,
        label="Baseline LSTM Density (KDE)",
        ax=ax
    )

    sns.histplot(
        p_sample,
        bins=bins,
        stat="density",
        color="#3182CE",
        alpha=0.35,
        label=f"Proposed ConvLSTM-Attention (Std: {p_std:.2f})",
        ax=ax,
        edgecolor="none"
    )
    sns.kdeplot(
        p_sample,
        clip=(-clip_val, clip_val),
        color="#2B6CB0",
        linewidth=2.4,
        label="Proposed Model Density (KDE)",
        ax=ax
    )

    # Zero-error reference line
    ax.axvline(0, color="#2D3748", linestyle="--", linewidth=1.6, alpha=0.85, label="Optimal Zero-Error ($e = 0$)")

    # Titles & Labels
    ax.set_title(
        "Residual Error Distribution Benchmark: Baseline vs. Proposed Model\n"
        "Testing Error = Ground Truth Flow - Predicted Flow (veh/5min)",
        fontsize=14,
        fontweight="bold",
        pad=15,
        color="#2D3748"
    )
    ax.set_xlabel("Residual Error (veh / 5 min)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_xlim(-clip_val, clip_val)

    # Statistics callout box
    stats_text = (
        "Statistical Comparison (Full Test Set):\n"
        "-------------------------------------\n"
        f"Baseline Mean Error:  {b_mean:+.2f} veh/5min\n"
        f"Baseline Std Dev:     {b_std:.2f} veh/5min\n"
        f"Baseline Overall MAE: {b_mae:.2f} veh/5min\n"
        "-------------------------------------\n"
        f"Proposed Mean Error:  {p_mean:+.2f} veh/5min\n"
        f"Proposed Std Dev:     {p_std:.2f} veh/5min ({((b_std - p_std)/b_std)*100:+.2f}% tighter)\n"
        f"Proposed Overall MAE: {p_mae:.2f} veh/5min ({((b_mae - p_mae)/b_mae)*100:+.2f}% better)"
    )
    ax.text(
        0.02, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#CBD5E0", alpha=0.93)
    )

    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=9.5)
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved Plot 2: {save_path.resolve()}")
    return save_path


def main():
    print("=" * 80)
    print("DATA VISUALIZATION SPECIALIST: TRAFFIC PREDICTION PLOTTING SUITE")
    print("=" * 80)
    print(f"• Figures Output Directory: {PLOTS_DIR.resolve()}")
    print("=" * 80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Hardware Device: {device}\n")

    start_time = time.time()
    targets_real, baseline_real, proposed_real = load_models_and_predict(device)

    # Plot 1: 24-Hour Timeline
    plot1_path = plot_24h_timeline(
        targets=targets_real,
        baseline_preds=baseline_real,
        proposed_preds=proposed_real,
        sensor_id=111,
        horizon_step=0
    )

    # Plot 2: Residual Error Distribution
    plot2_path = plot_residual_error_distribution(
        targets=targets_real,
        baseline_preds=baseline_real,
        proposed_preds=proposed_real
    )

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"• Time Elapsed: {total_time:.2f} seconds")
    print(f"• Plot 1 Created: {plot1_path.resolve()} (Size: {plot1_path.stat().st_size / 1024:.1f} KB)")
    print(f"• Plot 2 Created: {plot2_path.resolve()} (Size: {plot2_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
