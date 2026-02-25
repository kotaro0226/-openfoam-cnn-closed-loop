#!/usr/bin/env python3
"""
utils/vis-plot.py

Visualization for grid-search outputs:
- L x W heatmap using RMSE on selected split
- Learning curve (train/val loss) from train.log

Key behavior:
- lr / dropout / lambda-cont are handled independently.
- Heatmaps are generated per (lr, dropout, lambda_cont) combination by default.
- Optionally filter to one combination via CLI flags.
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
    p.add_argument("--split", default="val", choices=["train", "val", "test"], help="split for RMSE heatmap")
    p.add_argument("--agg", choices=["min", "mean"], default="min", help="aggregation across duplicate runs of same LxW")

    # Keep LR/dropout/lambda-cont independent: optional single-combo filter
    p.add_argument("--lr", default="", help="filter by lr (string match on CSV value)")
    p.add_argument("--dropout", default="", help="filter by dropout (string match on CSV value)")
    p.add_argument("--lambda-cont", dest="lambda_cont", default="", help="filter by lambda_cont (string match)")

    # Learning curve selection
    p.add_argument("--run-name", default="", help="run_name for learning curve (default: best RMSE in filtered rows)")
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


def _sanitize_tag(v: str) -> str:
    s = str(v)
    return s.replace("/", "_").replace(" ", "")


def parse_hidden_to_lw(hidden: str) -> Tuple[int, int]:
    vals = [int(x) for x in hidden.split("-") if x]
    if not vals:
        return (0, 0)
    L = len(vals)
    # representative width = first hidden width
    W = vals[0]
    return (L, W)


def _match_optional_filter(v: str, f: str) -> bool:
    if not f:
        return True
    return str(v) == str(f)


def filter_rows(rows: List[Dict[str, str]], split: str, lr: str, dropout: str, lambda_cont: str) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        if r.get("split", "") != split:
            continue
        if not _match_optional_filter(r.get("lr", ""), lr):
            continue
        if not _match_optional_filter(r.get("dropout", ""), dropout):
            continue
        if not _match_optional_filter(r.get("lambda_cont", ""), lambda_cont):
            continue
        out.append(r)
    return out


def split_by_hparams(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for r in rows:
        key = (r.get("lr", ""), r.get("dropout", ""), r.get("lambda_cont", ""))
        groups.setdefault(key, []).append(r)
    return groups


def build_lw_heatmap(rows: List[Dict[str, str]], agg: str):
    np = _import_numpy()
    pairs: Dict[Tuple[int, int], List[float]] = {}

    for r in rows:
        rmse = _to_float(r.get("rmse", "nan"))
        if not np.isfinite(rmse):
            continue
        L, W = parse_hidden_to_lw(r.get("hidden", ""))
        if L <= 0 or W <= 0:
            continue
        pairs.setdefault((L, W), []).append(rmse)

    if not pairs:
        raise RuntimeError("No valid rows after parsing hidden/rmse")

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


def plot_lw_heatmap(rows: List[Dict[str, str]], out_png: str, split: str, agg: str, tag: str):
    np = _import_numpy()
    plt = _import_matplotlib_pyplot()

    M, Ls, Ws = build_lw_heatmap(rows, agg=agg)

    fig, ax = plt.subplots(figsize=(1.2 * max(5, len(Ws)), 0.9 * max(4, len(Ls))), constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap="magma")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{agg}(RMSE) [{split}]")

    ax.set_title(f"L x W heatmap ({split} RMSE) [{tag}]")
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


def find_best_row(rows: List[Dict[str, str]]) -> Dict[str, str]:
    np = _import_numpy()
    cand = []
    for r in rows:
        rmse = _to_float(r.get("rmse", "nan"))
        if np.isfinite(rmse):
            cand.append((rmse, r))
    if not cand:
        raise RuntimeError("No row with finite rmse in filtered rows")
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

    filtered = filter_rows(
        rows,
        split=args.split,
        lr=args.lr,
        dropout=args.dropout,
        lambda_cont=args.lambda_cont,
    )
    if not filtered:
        raise RuntimeError("No rows after applying split/lr/dropout/lambda_cont filters")

    # If specific lr/dropout/lambda requested -> one heatmap.
    # Otherwise -> one heatmap per (lr,dropout,lambda_cont) combination.
    if args.lr or args.dropout or args.lambda_cont:
        tag = f"lr{_sanitize_tag(args.lr or 'all')}_do{_sanitize_tag(args.dropout or 'all')}_lc{_sanitize_tag(args.lambda_cont or 'all')}"
        heatmap_png = os.path.join(args.outdir, f"grid_heatmap_LxW_rmse_{args.split}_{tag}.png")
        plot_lw_heatmap(filtered, heatmap_png, split=args.split, agg=args.agg, tag=tag)
        print(f"[INFO] wrote heatmap: {heatmap_png}")
    else:
        groups = split_by_hparams(filtered)
        for (lr, dropout, lambda_cont), grows in groups.items():
            tag = f"lr{_sanitize_tag(lr)}_do{_sanitize_tag(dropout)}_lc{_sanitize_tag(lambda_cont)}"
            heatmap_png = os.path.join(args.outdir, f"grid_heatmap_LxW_rmse_{args.split}_{tag}.png")
            plot_lw_heatmap(grows, heatmap_png, split=args.split, agg=args.agg, tag=tag)
            print(f"[INFO] wrote heatmap: {heatmap_png}")

    # Learning curve target is chosen from the same filtered set.
    if args.run_name:
        row = find_row_by_run_name(filtered, args.run_name)
        if row is None:
            raise RuntimeError(f"run_name not found in filtered rows: {args.run_name}")
    else:
        row = find_best_row(filtered)

    curve_png = os.path.join(args.outdir, f"learning_curve_{row.get('run_name', 'best')}.png")
    plot_learning_curve(row, curve_png)
    print(f"[INFO] wrote learning curve: {curve_png}")


if __name__ == "__main__":
    main()

