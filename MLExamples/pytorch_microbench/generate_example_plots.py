#!/usr/bin/env python3
"""Generate example tutorial plots from validated pytorch_microbench runs."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = [
    ("baseline_resnet50_fp32.log", "ResNet50\nFP32"),
    ("densenet121_fp16.log", "DenseNet121\nFP16"),
    ("resnet50_compile.log", "ResNet50\ncompile"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate example plots from pytorch_microbench validation logs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/tmp/pytorch_microbench_plot_runs_20260321"),
        help="Directory containing benchmark and profiler logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("MLExamples/pytorch_microbench/images"),
        help="Directory where plot images will be written",
    )
    return parser.parse_args()


def require_match(pattern: str, text: str, context: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not find pattern {pattern!r} in {context}")
    return match.group(1)


def resolve_artifact_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    if path_text.startswith("/workspace/"):
        translated = REPO_ROOT / path.relative_to("/workspace")
        if translated.exists():
            return translated
    raise FileNotFoundError(f"Could not resolve artifact path: {path_text}")


def parse_benchmark_logs(log_dir: Path) -> pd.DataFrame:
    rows = []
    for filename, label in RUNS:
        log_path = log_dir / filename
        text = log_path.read_text()
        rows.append(
            {
                "label": label,
                "network": require_match(
                    r"Microbenchmark for network : ([^\n]+)", text, str(log_path)
                ),
                "dtype": require_match(r"Dtype: ([^\n]+)", text, str(log_path)),
                "time_per_batch": float(
                    require_match(r"Time per mini-batch : ([0-9.]+)", text, str(log_path))
                ),
                "throughput": float(
                    require_match(r"Throughput \[img/sec\] : ([0-9.]+)", text, str(log_path))
                ),
            }
        )
    return pd.DataFrame(rows)


def shorten_kernel_name(name: str) -> str:
    if name.startswith("void at::native::vectorized_elementwise_kernel"):
        short = "ATen vectorized elementwise kernel"
    elif name.startswith("Cijk_"):
        short = "Tensile GEMM kernel"
    else:
        short = name

    if len(short) > 52:
        short = short[:49] + "..."
    return short


def parse_hotspots(log_dir: Path, top_n: int = 8) -> pd.DataFrame:
    log_path = log_dir / "get_gpu_hotspots.log"
    text = log_path.read_text()
    csv_path = resolve_artifact_path(
        require_match(r"Kernel trace CSV: (.+_kernel_trace\.csv)", text, str(log_path))
    )

    totals: defaultdict[str, float] = defaultdict(float)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            duration_ms = (
                int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            ) / 1_000_000.0
            totals[row["Kernel_Name"]] += duration_ms

    top = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return pd.DataFrame(
        {
            "kernel_name": [name for name, _ in top],
            "total_duration_ms": [duration for _, duration in top],
            "short_name": [shorten_kernel_name(name) for name, _ in top],
        }
    )


def add_bar_labels(ax: plt.Axes, values: pd.Series, fmt: str) -> None:
    for idx, value in enumerate(values):
        ax.text(idx, value, fmt.format(value), ha="center", va="bottom", fontsize=9)


def plot_benchmark_examples(df: pd.DataFrame, output_path: Path) -> None:
    colors = ["#1f3c88", "#4f772d", "#c97b24"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

    axes[0].bar(df["label"], df["throughput"], color=colors)
    axes[0].set_title("Throughput")
    axes[0].set_ylabel("img/sec")
    axes[0].grid(axis="y", alpha=0.2)
    add_bar_labels(axes[0], df["throughput"], "{:.1f}")

    axes[1].bar(df["label"], df["time_per_batch"], color=colors)
    axes[1].set_title("Time per mini-batch")
    axes[1].set_ylabel("seconds")
    axes[1].grid(axis="y", alpha=0.2)
    add_bar_labels(axes[1], df["time_per_batch"], "{:.3f}")

    fig.suptitle(
        "pytorch_microbench example measurements from validated container runs",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hotspots(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values("total_duration_ms", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.barh(plot_df["short_name"], plot_df["total_duration_ms"], color="#1f3c88")
    ax.set_xlabel("Total duration (ms)")
    ax.set_title("pytorch_microbench GPU hotspots from validated container run")
    ax.grid(axis="x", alpha=0.2)

    for y, value in enumerate(plot_df["total_duration_ms"]):
        ax.text(value, y, f" {value:.2f}", va="center", ha="left", fontsize=9)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df = parse_benchmark_logs(args.log_dir)
    plot_benchmark_examples(
        benchmark_df,
        args.output_dir / "pytorch_microbench_example_runs.png",
    )

    hotspots_df = parse_hotspots(args.log_dir)
    plot_hotspots(
        hotspots_df,
        args.output_dir / "pytorch_microbench_gpu_hotspots.png",
    )

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
