#!/usr/bin/env python3
"""Generate example tutorial plots from validated TinyTransformer runs."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


VERSION_ORDER = [
    ("version1_pytorch_baseline", "V1"),
    ("version2_pytorch_fused", "V2"),
    ("version3_triton", "V3"),
    ("version4_pytorch_sdpa", "V4"),
]

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate example plots from TinyTransformer validation logs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/tmp/tinytransformer_validation_20260322"),
        help="Directory containing validation logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("MLExamples/TinyTransformer/images"),
        help="Directory where plot images will be written",
    )
    return parser.parse_args()


def require_match(pattern: str, text: str, context: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not find pattern {pattern!r} in {context}")
    return match.group(1)


def parse_baseline_metrics(log_dir: Path) -> pd.DataFrame:
    rows = []
    for version_dir, label in VERSION_ORDER:
        log_path = log_dir / f"{version_dir}__baseline.log"
        text = log_path.read_text()
        rows.append(
            {
                "version_dir": version_dir,
                "label": label,
                "avg_training_speed": float(
                    require_match(r"Average training speed:\s+([0-9.]+)", text, str(log_path))
                ),
                "avg_batch_time_ms": float(
                    require_match(r"Average batch time:\s+([0-9.]+)\s+ms", text, str(log_path))
                ),
                "peak_memory_mb": float(
                    require_match(r"Peak memory usage:\s+([0-9.]+)\s+MB", text, str(log_path))
                ),
            }
        )
    return pd.DataFrame(rows)


def shorten_kernel_name(name: str) -> str:
    if name.startswith("Cijk_"):
        short = name.split("_SN_")[0]
    elif name.startswith("void at::native::"):
        short = "ATen kernel: " + name.split("(", 1)[0].replace("void ", "")
    else:
        short = name

    if len(short) > 64:
        short = short[:61] + "..."
    return short


def parse_hotspots(log_dir: Path, version_dir: str, top_n: int = 8) -> pd.DataFrame:
    log_path = log_dir / f"{version_dir}__hotspots.log"
    text = log_path.read_text()
    csv_path = resolve_artifact_path(
        require_match(r"Top rows from (.+_kernel_stats\.csv):", text, str(log_path))
    )
    df = pd.read_csv(csv_path)
    top = df.sort_values("TotalDurationNs", ascending=False).head(top_n).copy()
    top["TotalDurationMs"] = top["TotalDurationNs"] / 1e6
    top["ShortName"] = top["Name"].map(shorten_kernel_name)
    return top


def resolve_artifact_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    if path_text.startswith("/workspace/"):
        translated = REPO_ROOT / path.relative_to("/workspace")
        if translated.exists():
            return translated
    raise FileNotFoundError(f"Could not resolve artifact path: {path_text}")


def add_bar_labels(ax: plt.Axes, values: pd.Series, fmt: str) -> None:
    for idx, value in enumerate(values):
        ax.text(idx, value, fmt.format(value), ha="center", va="bottom", fontsize=9)


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    colors = ["#1f3c88", "#4f772d", "#c97b24", "#7a3e9d"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    metrics = [
        ("avg_training_speed", "Average training speed", "samples/sec", "{:.1f}"),
        ("avg_batch_time_ms", "Average batch time", "ms", "{:.1f}"),
        ("peak_memory_mb", "Peak memory", "MB", "{:.1f}"),
    ]

    for ax, (column, title, ylabel, fmt) in zip(axes, metrics):
        ax.bar(df["label"], df[column], color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        add_bar_labels(ax, df[column], fmt)

    fig.suptitle(
        "TinyTransformer example measurements from validated container runs",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hotspots(top: pd.DataFrame, title: str, output_path: Path, color: str) -> None:
    plot_df = top.sort_values("TotalDurationMs", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.barh(plot_df["ShortName"], plot_df["TotalDurationMs"], color=color)
    ax.set_xlabel("Total duration (ms)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)

    for y, value in enumerate(plot_df["TotalDurationMs"]):
        ax.text(value, y, f" {value:.2f}", va="center", ha="left", fontsize=9)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = parse_baseline_metrics(args.log_dir)
    plot_comparison(
        baseline_df,
        args.output_dir / "tinytransformer_baseline_comparison.png",
    )

    v1_hotspots = parse_hotspots(args.log_dir, "version1_pytorch_baseline")
    plot_hotspots(
        v1_hotspots,
        "TinyTransformer V1 hotspot summary from validated container run",
        args.output_dir / "tinytransformer_version1_hotspots.png",
        "#1f3c88",
    )

    v3_hotspots = parse_hotspots(args.log_dir, "version3_triton")
    plot_hotspots(
        v3_hotspots,
        "TinyTransformer V3 hotspot summary from validated container run",
        args.output_dir / "tinytransformer_version3_hotspots.png",
        "#c97b24",
    )

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
