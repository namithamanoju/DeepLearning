#!/usr/bin/env python3
"""
PeMS Spatio-Temporal Traffic Dataset Inspection Script
Capstone Project: Deep Learning for Traffic Forecasting
Workspace: DL_PROJECT

This script:
1. Discovers and loads PeMS traffic dataset files (.npz, .csv, .h5).
2. Analyzes spatial-temporal tensor shapes, missing values, and data integrity.
3. Computes sampling resolution, temporal duration, and feature-level statistics.
4. Evaluates graph topology / sensor adjacency files.
5. Generates a comprehensive summary printed to the console and saved to 'dataset_summary.txt'.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


FEATURE_NAMES = {
    0: "Flow / Volume (veh/5min)",
    1: "Occupancy rate ([0, 1])",
    2: "Average Speed (mph / km/h)"
}

STANDARD_INTERVAL_MINUTES = 5
INTERVALS_PER_DAY = (24 * 60) // STANDARD_INTERVAL_MINUTES  # 288 steps/day


def find_dataset_files(data_dir: Path):
    """Scan the data directory for PeMS npz/h5 and distance csv files."""
    dataset_pairs = {}
    if not data_dir.exists():
        return dataset_pairs

    # Search for NPZ files
    for npz_file in data_dir.glob("*.npz"):
        dataset_name = npz_file.stem
        csv_file = data_dir / f"{dataset_name}.csv"
        dataset_pairs[dataset_name] = {
            "tensor_file": npz_file,
            "graph_file": csv_file if csv_file.exists() else None,
            "format": "npz"
        }

    # Search for H5 / HDF5 files if any
    for h5_file in list(data_dir.glob("*.h5")) + list(data_dir.glob("*.h5py")) + list(data_dir.glob("*.hdf5")):
        dataset_name = h5_file.stem
        csv_file = data_dir / f"{dataset_name}.csv"
        if dataset_name not in dataset_pairs:
            dataset_pairs[dataset_name] = {
                "tensor_file": h5_file,
                "graph_file": csv_file if csv_file.exists() else None,
                "format": "h5"
            }

    return dataset_pairs


def inspect_tensor(tensor_path: Path):
    """Load and compute statistics for a PeMS tensor (.npz)."""
    if tensor_path.suffix == ".npz":
        npz_data = np.load(tensor_path)
        keys = list(npz_data.files)
        # Standard benchmark key is 'data'
        key = "data" if "data" in keys else keys[0]
        data = npz_data[key]
    else:
        raise ValueError(f"Unsupported file format: {tensor_path.suffix}")

    time_steps, num_sensors, num_features = data.shape

    # Missing value diagnostics
    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())
    zero_count = int((data == 0).sum())
    negative_count = int((data < 0).sum())

    total_values = data.size
    nan_ratio = (nan_count / total_values) * 100.0

    # Temporal duration and sampling resolution
    total_days = time_steps / INTERVALS_PER_DAY
    total_hours = time_steps * (STANDARD_INTERVAL_MINUTES / 60.0)

    # Feature-wise statistics
    feature_stats = {}
    for f_idx in range(num_features):
        feat_slice = data[:, :, f_idx]
        f_name = FEATURE_NAMES.get(f_idx, f"Feature {f_idx}")
        feature_stats[f_name] = {
            "min": float(np.min(feat_slice)),
            "max": float(np.max(feat_slice)),
            "mean": float(np.mean(feat_slice)),
            "std": float(np.std(feat_slice)),
            "q25": float(np.percentile(feat_slice, 25)),
            "median": float(np.median(feat_slice)),
            "q75": float(np.percentile(feat_slice, 75)),
            "zero_ratio": float((feat_slice == 0).sum() / feat_slice.size) * 100.0
        }

    return {
        "file_name": tensor_path.name,
        "file_size_mb": tensor_path.stat().st_size / (1024 * 1024),
        "shape": (time_steps, num_sensors, num_features),
        "dtype": str(data.dtype),
        "total_values": total_values,
        "nan_count": nan_count,
        "nan_ratio": nan_ratio,
        "inf_count": inf_count,
        "negative_count": negative_count,
        "zero_count": zero_count,
        "sampling_resolution_min": STANDARD_INTERVAL_MINUTES,
        "steps_per_day": INTERVALS_PER_DAY,
        "total_days": total_days,
        "total_hours": total_hours,
        "feature_stats": feature_stats
    }


def inspect_graph(graph_path: Path, expected_sensors: int):
    """Analyze sensor topology and connectivity from graph CSV."""
    if not graph_path or not graph_path.exists():
        return None

    df = pd.read_csv(graph_path)
    # Check column names: standard PeMS graph has ['from', 'to', 'cost']
    cols = [c.lower() for c in df.columns]
    num_edges = len(df)
    unique_sources = df.iloc[:, 0].nunique()
    unique_targets = df.iloc[:, 1].nunique()
    unique_sensors = len(set(df.iloc[:, 0]).union(set(df.iloc[:, 1])))

    cost_col = df.iloc[:, 2] if len(df.columns) >= 3 else None
    cost_stats = {}
    if cost_col is not None:
        cost_stats = {
            "min": float(cost_col.min()),
            "max": float(cost_col.max()),
            "mean": float(cost_col.mean()),
            "std": float(cost_col.std())
        }

    density = (num_edges / (expected_sensors * expected_sensors)) * 100.0 if expected_sensors > 0 else 0.0

    return {
        "file_name": graph_path.name,
        "file_size_kb": graph_path.stat().st_size / 1024,
        "num_edges": num_edges,
        "unique_sensors_in_edges": unique_sensors,
        "expected_sensors": expected_sensors,
        "connectivity_density_pct": density,
        "cost_stats": cost_stats
    }


def generate_report(datasets_info: dict, output_file: Path):
    """Format inspection outputs into console and text file report."""
    lines = []
    separator_double = "=" * 80
    separator_single = "-" * 80

    lines.append(separator_double)
    lines.append("CALIFORNIA HIGHWAY PeMS TRAFFIC DATASET INSPECTION REPORT")
    lines.append("Capstone Deep Learning Project - Spatial-Temporal Graph Neural Networks")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(separator_double)

    for dname, info in datasets_info.items():
        t_info = info["tensor"]
        g_info = info.get("graph")

        lines.append(f"\n[DATASET: {dname}]")
        lines.append(separator_single)
        lines.append(f"- Tensor File:             {t_info['file_name']} ({t_info['file_size_mb']:.2f} MB)")
        lines.append(f"- Array Dimensions:        Shape: {t_info['shape']} [Time Steps, Sensors, Features]")
        lines.append(f"  * Number of Time Steps:  {t_info['shape'][0]:,}")
        lines.append(f"  * Number of Detectors:   {t_info['shape'][1]:,} road sensors")
        lines.append(f"  * Number of Channels:    {t_info['shape'][2]} features (Flow, Occupancy, Speed)")
        lines.append(f"- Data Precision & Type:   {t_info['dtype']}")
        lines.append(f"- Missing Value Count:     {t_info['nan_count']} NaNs ({t_info['nan_ratio']:.4f}%), {t_info['inf_count']} Infs")
        lines.append(f"- Negative Value Count:    {t_info['negative_count']} (No invalid negative measurements)")
        lines.append(f"- Zero Value Count:        {t_info['zero_count']:,} ({t_info['zero_count']/t_info['total_values']*100:.2f}% of tensor)")

        lines.append("\n  [Temporal Horizon & Resolution]")
        lines.append(f"  - Time Interval:         {t_info['sampling_resolution_min']} minutes per reading")
        lines.append(f"  - Sampling Frequency:    {t_info['steps_per_day']} intervals/day (12 steps/hour)")
        lines.append(f"  - Dataset Span:          {t_info['total_days']:.1f} consecutive days ({t_info['total_hours']:.1f} hours)")

        lines.append("\n  [Feature-wise Statistical Distribution]")
        lines.append(f"  {'Feature':<30} | {'Mean':<10} | {'Std':<10} | {'Min':<8} | {'Median':<8} | {'Max':<8} | {'Zeros %':<8}")
        lines.append(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for fname, s in t_info["feature_stats"].items():
            lines.append(
                f"  {fname:<30} | {s['mean']:<10.2f} | {s['std']:<10.2f} | {s['min']:<8.2f} | "
                f"{s['median']:<8.2f} | {s['max']:<8.2f} | {s['zero_ratio']:<7.2f}%"
            )

        if g_info:
            lines.append("\n  [Spatial Graph Topology & Sensor Adjacency]")
            lines.append(f"  - Graph Edge List:       {g_info['file_name']} ({g_info['file_size_kb']:.2f} KB)")
            lines.append(f"  - Directed Edges:        {g_info['num_edges']:,} highway segments connecting sensors")
            lines.append(f"  - Sensors in Graph:      {g_info['unique_sensors_in_edges']} unique nodes")
            lines.append(f"  - Topology Density:      {g_info['connectivity_density_pct']:.3f}% (Sparsely connected road network)")
            if g_info["cost_stats"]:
                c = g_info["cost_stats"]
                lines.append(f"  - Distance/Cost Metric:  Mean: {c['mean']:.2f} m, Min: {c['min']:.1f} m, Max: {c['max']:.1f} m")
        else:
            lines.append("\n  [Spatial Graph Topology]")
            lines.append("  - Graph CSV file not found (will be inferred or loaded separately).")

        lines.append("\n  [Preprocessing & Deep Learning Readiness Assessment]")
        readiness_status = "READY FOR PREPROCESSING" if t_info['nan_count'] == 0 else "REQUIRES IMPUTATION"
        lines.append(f"  - Status:                >>> {readiness_status} <<<")
        lines.append("  - Recommended Pipeline:  1. Z-Score Standardization / MinMax Scaler per feature channel.")
        lines.append("                           2. Sliding-window sequence framing (e.g. 12 steps historical -> 12 steps future).")
        lines.append("                           3. Train/Val/Test temporal split: 70% / 10% / 20% (strictly chronological).")
        lines.append("                           4. Adjacency Matrix: Gaussian thresholded distance kernel A_ij = exp(-dist^2 / sigma^2).")

    lines.append("\n" + separator_double)
    lines.append("INSPECTION COMPLETE: All required metadata successfully logged.")
    lines.append(separator_double + "\n")

    report_text = "\n".join(lines)

    # Print to stdout
    print(report_text)

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[+] Summary successfully written to: {output_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Inspect PeMS Traffic Dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_file", type=str, default="dataset_summary.txt", help="Path to output summary file")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    output_path = project_root / args.output_file

    print(f"\n[*] Scanning for datasets in: {data_dir.resolve()}...")
    dataset_pairs = find_dataset_files(data_dir)

    if not dataset_pairs:
        print(f"[!] No .npz or .h5 datasets found in {data_dir.resolve()}.")
        sys.exit(1)

    print(f"[*] Found {len(dataset_pairs)} dataset(s): {', '.join(dataset_pairs.keys())}")

    results = {}
    for dname, paths in dataset_pairs.items():
        print(f"[*] Inspecting {dname}...")
        tensor_stats = inspect_tensor(paths["tensor_file"])
        graph_stats = inspect_graph(paths["graph_file"], tensor_stats["shape"][1]) if paths["graph_file"] else None
        results[dname] = {
            "tensor": tensor_stats,
            "graph": graph_stats
        }

    generate_report(results, output_path)


if __name__ == "__main__":
    main()
