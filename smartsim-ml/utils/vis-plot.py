#!/usr/bin/env python3
"""
utils/vis-plot.py

Visualization for grid-search outputs:
- L x W heatmap using val RMSE
- Learning curve (train/val loss) from train.log
"""

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple


def _import_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("numpy is required for plotting.") from e
    return np


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("matplotlib is required for plotting.") from e
    return plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot LxW heatmap and learning curve")
    p.add_argument("--csv", required=True, help="summary CSV from Main_pype_grid.py")
    p.add_argument("--outdir", default="./figures")
    p.add_argument("--split", default="val", choices=["train", "val", "test"], help="which split to use for RMSE heatmap")
    p.add_argument("--agg", choices=["min", "mean"], default="min")
    p.add_argument("--run-name", default="", help="run_name for learning curve (default: best by split RMSE)")
    return p.parse_args()


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV has no rows: {path}")
    return rows


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def parse_hidden_to_lw(hidden: str) -> Tuple[int, int]:
    vals = [int(x) for x in hidden.split("-") if x]
    if not vals:
        return (0, 0)
    L = len(vals)
    # width axis for heatmap: representative width = first hidden size
    W = vals[0]
    return (L, W)


def build_lw_heatmap(rows: List[Dict[str, str]], split: str, agg: str):
    np = _import_numpy()
    pairs: Dict[Tuple[int, int], List[float]] = {}

    for r in rows:
        if r.get("split", "") != split:
            continue
        rmse = _to_float(r.get("rmse", "nan"))
        if not np.isfinite(rmse):
            continue
        L, W = parse_hidden_to_lw(r.get("hidden", ""))
        if L <= 0 or W <= 0:
            continue
        pairs.setdefault((L, W), []).append(rmse)

    if not pairs:
        raise RuntimeError(f"No valid rows for split='{split}' and hidden parsing")

    Ls = sorted({k[0] for k in pairs.keys()})
    Ws = sorted({k[1] for k in pairs.keys()})
    li = {v: i for i, v in enumerate(Ls)}
    wi = {v: i for i, v in enumerate(Ws)}

    M = np.full((len(Ls), len(Ws)), np.nan, dtype=np.float64)
    for (L, W), vals in pairs.items():
        if agg == "min":
            M[li[L], wi[W]] = float(np.min(vals))
        else:
            M[li[L], wi[W]] = float(np.mean(vals))

    return M, Ls, Ws


def plot_lw_heatmap(rows: List[Dict[str, str]], out_png: str, split: str, agg: str):
    np = _import_numpy()
    plt = _import_matplotlib_pyplot()

    M, Ls, Ws = build_lw_heatmap(rows, split=split, agg=agg)

    fig, ax = plt.subplots(figsize=(1.2 * max(5, len(Ws)), 0.9 * max(4, len(Ls))), constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap="magma")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{agg}(RMSE) [{split}]")

    ax.set_title(f"L x W heatmap ({split} RMSE)")
    ax.set_xlabel("W (hidden width)")
    ax.set_ylabel("L (num hidden layers)")

    ax.set_xticks(np.arange(len(Ws)))
    ax.set_xticklabels([str(w) for w in Ws])
    ax.set_yticks(np.arange(len(Ls)))
    ax.set_yticklabels([str(l) for l in Ls])

    for iy in range(M.shape[0]):
        for ix in range(M.shape[1]):
            if np.isfinite(M[iy, ix]):
                ax.text(ix, iy, f"{M[iy, ix]:.2e}", ha="center", va="center", fontsize=8, color="white")

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def find_best_row(rows: List[Dict[str, str]], split: str) -> Dict[str, str]:
    np = _import_numpy()
    cand = []
    for r in rows:
        if r.get("split", "") != split:
            continue
        rmse = _to_float(r.get("rmse", "nan"))
        if np.isfinite(rmse):
            cand.append((rmse, r))
    if not cand:
        raise RuntimeError(f"No row with split='{split}' and finite rmse")
    cand.sort(key=lambda x: x[0])
    return cand[0][1]


def find_row_by_run_name(rows: List[Dict[str, str]], run_name: str) -> Optional[Dict[str, str]]:
    for r in rows:
        if r.get("run_name", "") == run_name:
            return r
    return None


def parse_train_log(log_path: str):
    epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    pat = re.compile(r"epoch\s+(\d+)/(\d+)\s+train_loss=([0-9eE+\-.]+)\s+val_loss=([0-9eE+\-.]+)")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(3)))
            val_losses.append(float(m.group(4)))

    if not epochs:
        raise RuntimeError(f"Could not parse epoch train/val losses from: {log_path}")

    return epochs, train_losses, val_losses


def plot_learning_curve(row: Dict[str, str], out_png: str):
    plt = _import_matplotlib_pyplot()

    run_dir = row.get("run_dir", "")
    run_name = row.get("run_name", "unknown")
    if not run_dir:
        raise RuntimeError("summary CSV row has no run_dir; cannot locate train.log")

    log_path = os.path.join(run_dir, "train.log")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"train.log not found: {log_path}")

    epochs, train_losses, val_losses = parse_train_log(log_path)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.plot(epochs, train_losses, marker="o", label="train_loss")
    ax.plot(epochs, val_losses, marker="s", label="val_loss")
    ax.set_title(f"Learning curve: {run_name}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    heatmap_png = os.path.join(args.outdir, f"grid_heatmap_LxW_rmse_{args.split}.png")
    plot_lw_heatmap(rows, heatmap_png, split=args.split, agg=args.agg)

    if args.run_name:
        row = find_row_by_run_name(rows, args.run_name)
        if row is None:
            raise RuntimeError(f"run_name not found in summary CSV: {args.run_name}")
    else:
        row = find_best_row(rows, split=args.split)

    curve_png = os.path.join(args.outdir, f"learning_curve_{row.get('run_name', 'best')}.png")
    plot_learning_curve(row, curve_png)

    print(f"[INFO] wrote heatmap: {heatmap_png}")
    print(f"[INFO] wrote learning curve: {curve_png}")


if __name__ == "__main__":
    main()

