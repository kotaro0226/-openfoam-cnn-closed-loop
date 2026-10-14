#!/usr/bin/env python3
import argparse
import csv
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.normalization import load_norm, apply_zscore, invert_zscore


def parse_args():
    p = argparse.ArgumentParser(description="Compute timestep-wise loss from cell NPZ and TorchScript model")
    p.add_argument("--jit", required=True)
    p.add_argument("--cell-npz", required=True)
    p.add_argument("--norm-path", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-png", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--fields-per-face", type=int, default=3)
    p.add_argument("--num-faces-per-cell", type=int, default=6)
    p.add_argument("--dt", type=float, default=None, help="physical deltaT used in CFD")
    p.add_argument("--write-interval", type=int, default=1, help="number of CFD steps between saved snapshots")
    p.add_argument(
        "--x-mode",
        default="index",
        choices=["index", "time"],
        help="plot against snapshot index or physical time",
    )
    p.add_argument(
        "--logy",
        action="store_true",
        help="use log scale for y axis; useful when plotting MSE together with RMSE/MAE",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    z = np.load(args.cell_npz, allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float32)
    Y = np.asarray(z["Y"], dtype=np.float32)
    T_effective = int(np.asarray(z["T_effective"]).item())
    num_hypercubes = int(np.asarray(z["num_hypercubes"]).item())

    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"Expected X/Y as 3D arrays, got X={X.shape} Y={Y.shape}")
    if X.shape[0] != T_effective * num_hypercubes:
        raise ValueError(
            f"Unexpected block axis: X.shape[0]={X.shape[0]} vs T_effective*num_hypercubes={T_effective*num_hypercubes}"
        )

    model = torch.jit.load(args.jit, map_location=device)
    model.eval()
    x_mean, x_std, y_mean, y_std = load_norm(args.norm_path, device=device)

    rows = []
    xvals = []
    mse_list = []
    rmse_list = []
    mae_list = []

    with torch.no_grad():
        for t in range(T_effective):
            sl = slice(t * num_hypercubes, (t + 1) * num_hypercubes)
            Xt = X[sl]  # (H,Ns,18)
            Yt = Y[sl]  # (H,Ns,6)
            H, Ns, Cin = Xt.shape

            Xc = torch.from_numpy(Xt.reshape(H * Ns, Cin)).to(device)
            Y6 = torch.from_numpy(Yt.reshape(H * Ns, args.num_faces_per_cell)).to(device)

            B = Xc.shape[0]
            Xfaces = Xc.view(B, args.num_faces_per_cell, args.fields_per_face).reshape(
                B * args.num_faces_per_cell, args.fields_per_face
            )
            pred_n = model(apply_zscore(Xfaces, x_mean, x_std)).reshape(B * args.num_faces_per_cell, 1)
            pred_phys = invert_zscore(pred_n, y_mean, y_std).reshape(B, args.num_faces_per_cell)

            err = pred_phys - Y6
            mse = float(torch.mean(err ** 2).cpu().item())
            rmse = float(np.sqrt(mse))
            mae = float(torch.mean(torch.abs(err)).cpu().item())

            physical_time = None
            if args.dt is not None:
                physical_time = (t + 1) * args.write_interval * args.dt

            rows.append({
                "snapshot_index": t,
                "physical_time": physical_time,
                "num_cells": B,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
            })
            xvals.append(physical_time if args.x_mode == "time" and physical_time is not None else t)
            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["snapshot_index", "physical_time", "num_cells", "mse", "rmse", "mae"],
        )
        w.writeheader()
        w.writerows(rows)

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    ax.plot(xvals, mse_list, marker="o", markersize=3, linewidth=1.2, label="MSE")
    ax.plot(xvals, rmse_list, marker="s", markersize=3, linewidth=1.2, label="RMSE")
    ax.plot(xvals, mae_list, marker="^", markersize=3, linewidth=1.2, label="MAE")
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("physical time" if args.x_mode == "time" else "snapshot index")
    ax.set_ylabel("loss")
    ax.set_title("Timestep-wise Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(args.out_png, dpi=180)

    print(f"[INFO] wrote csv: {args.out_csv}")
    print(f"[INFO] wrote png: {args.out_png}")


if __name__ == "__main__":
    main()
