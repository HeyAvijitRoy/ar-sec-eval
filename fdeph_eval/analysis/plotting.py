# fdeph_eval/analysis/plotting.py
"""
Author: Avijit Roy
Project: FDEPH - Attack Efficiency Analysis

Purpose:
- Load long-format attack logs (CSV)
- Produce plots:
  - Distance vs Steps (median + IQR)
  - Distance vs Time (median + IQR)
  - Success rate vs Steps/Time
  - Time-to-success summary table
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class RunSummary:
    n_images: int
    n_succeeded: int
    success_rate: float
    median_steps_to_success: Optional[float]
    median_time_ms_to_success: Optional[float]


def load_attack_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop hyperparams/meta row if present
    if "image_id" in df.columns:
        df = df[df["image_id"] != "__HYPERPARAMS__"]

    # Basic type safety
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["elapsed_ms"] = pd.to_numeric(df["elapsed_ms"], errors="coerce")
    df["dist_raw"] = pd.to_numeric(df["dist_raw"], errors="coerce")
    df["dist_norm"] = pd.to_numeric(df["dist_norm"], errors="coerce")
    df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)

    # Drop rows missing core fields
    df = df.dropna(subset=["image_id", "step", "elapsed_ms", "dist_norm"])
    df["step"] = df["step"].astype(int)

    return df


def compute_success_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per image: first success step/time (if success occurs).
    """
    suc = df[df["success"] == 1].copy()
    if suc.empty:
        return pd.DataFrame(columns=["image_id", "success_step", "success_elapsed_ms"])

    # First success per image
    suc = suc.sort_values(["image_id", "step", "elapsed_ms"])
    first = suc.groupby("image_id", as_index=False).first()
    first = first.rename(columns={"step": "success_step", "elapsed_ms": "success_elapsed_ms"})
    return first[["image_id", "success_step", "success_elapsed_ms"]]


def summarize_run(df: pd.DataFrame) -> RunSummary:
    n_images = df["image_id"].nunique()
    events = compute_success_events(df)
    n_succeeded = len(events)
    success_rate = (n_succeeded / n_images) if n_images else 0.0

    med_steps = float(events["success_step"].median()) if n_succeeded else None
    med_time = float(events["success_elapsed_ms"].median()) if n_succeeded else None

    return RunSummary(
        n_images=n_images,
        n_succeeded=n_succeeded,
        success_rate=success_rate,
        median_steps_to_success=med_steps,
        median_time_ms_to_success=med_time,
    )


def _median_iqr_by_x(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    g = df.groupby(x_col)[y_col]
    out = pd.DataFrame({
        x_col: g.median().index,
        "median": g.median().values,
        "q25": g.quantile(0.25).values,
        "q75": g.quantile(0.75).values,
    })
    return out.sort_values(x_col)


def plot_distance_vs_steps(df: pd.DataFrame, out_path: Optional[str | Path] = None, title: str = "") -> None:
    curve = _median_iqr_by_x(df, "step", "dist_norm")

    plt.figure()
    plt.plot(curve["step"], curve["median"])
    plt.fill_between(curve["step"], curve["q25"], curve["q75"], alpha=0.2)
    plt.xlabel("Step")
    plt.ylabel("Normalized Hamming Distance")
    plt.title(title or "Distance vs Steps (median + IQR)")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_distance_vs_time(df: pd.DataFrame, bin_ms: int = 250, out_path: Optional[str | Path] = None, title: str = "") -> None:
    """
    Bin elapsed time to reduce jaggedness.
    """
    d = df.copy()
    d["time_bin_ms"] = (d["elapsed_ms"] // bin_ms) * bin_ms
    curve = _median_iqr_by_x(d, "time_bin_ms", "dist_norm")

    plt.figure()
    plt.plot(curve["time_bin_ms"], curve["median"])
    plt.fill_between(curve["time_bin_ms"], curve["q25"], curve["q75"], alpha=0.2)
    plt.xlabel(f"Elapsed time (ms, binned {bin_ms}ms)")
    plt.ylabel("Normalized Hamming Distance")
    plt.title(title or "Distance vs Time (median + IQR)")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_success_rate_vs_steps(df: pd.DataFrame, out_path: Optional[str | Path] = None, title: str = "") -> None:
    """
    Success rate at step s = fraction of images whose first success_step <= s.
    """
    n_images = df["image_id"].nunique()
    events = compute_success_events(df)

    if n_images == 0:
        return

    max_step = int(df["step"].max())
    steps = np.arange(1, max_step + 1)

    if events.empty:
        rates = np.zeros_like(steps, dtype=float)
    else:
        first_steps = events["success_step"].to_numpy()
        rates = np.array([(first_steps <= s).mean() for s in steps], dtype=float)

    plt.figure()
    plt.plot(steps, rates)
    plt.xlabel("Step")
    plt.ylabel("Success rate")
    plt.ylim(0, 1.0)
    plt.title(title or "Success Rate vs Steps")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_success_rate_vs_time(df: pd.DataFrame, bin_ms: int = 250, out_path: Optional[str | Path] = None, title: str = "") -> None:
    """
    Success rate at time t = fraction of images whose first success_elapsed_ms <= t.
    """
    n_images = df["image_id"].nunique()
    events = compute_success_events(df)

    if n_images == 0:
        return

    max_t = float(df["elapsed_ms"].max())
    bins = np.arange(0, max_t + bin_ms, bin_ms)

    if events.empty:
        rates = np.zeros_like(bins, dtype=float)
    else:
        first_times = events["success_elapsed_ms"].to_numpy()
        rates = np.array([(first_times <= t).mean() for t in bins], dtype=float)

    plt.figure()
    plt.plot(bins, rates)
    plt.xlabel(f"Elapsed time (ms, binned {bin_ms}ms)")
    plt.ylabel("Success rate")
    plt.ylim(0, 1.0)
    plt.title(title or "Success Rate vs Time")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def export_time_to_success_table(df: pd.DataFrame, out_csv: str | Path) -> pd.DataFrame:
    """
    Writes a per-image table with first success step/time.
    """
    events = compute_success_events(df)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_csv, index=False)
    return events

def time_to_success_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a one-row table with median and 95th percentile for steps/time.
    """
    events = compute_success_events(df)
    if events.empty:
        return pd.DataFrame([{
            "n_images": df["image_id"].nunique(),
            "n_succeeded": 0,
            "success_rate": 0.0,
            "median_steps": np.nan,
            "p95_steps": np.nan,
            "median_time_ms": np.nan,
            "p95_time_ms": np.nan,
        }])

    n_images = df["image_id"].nunique()
    n_succeeded = len(events)
    sr = n_succeeded / n_images if n_images else 0.0

    return pd.DataFrame([{
        "n_images": n_images,
        "n_succeeded": n_succeeded,
        "success_rate": sr,
        "median_steps": float(events["success_step"].median()),
        "p95_steps": float(events["success_step"].quantile(0.95)),
        "median_time_ms": float(events["success_elapsed_ms"].median()),
        "p95_time_ms": float(events["success_elapsed_ms"].quantile(0.95)),
    }])


def plot_time_to_success_hist(df: pd.DataFrame, out_path: Optional[str | Path] = None, title: str = "") -> None:
    """
    Histogram of time-to-success (ms) over images that succeeded.
    """
    events = compute_success_events(df)
    if events.empty:
        print("No successes to plot.")
        return

    plt.figure()
    plt.hist(events["success_elapsed_ms"], bins=30)
    plt.xlabel("Time to success (ms)")
    plt.ylabel("Number of images")
    plt.title(title or "Time-to-success histogram")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()