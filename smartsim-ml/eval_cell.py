#!/usr/bin/env python3
# smartsim-ml/eval_cell.py
#
# Cell-wise evaluation for subsampled_ofCell_*.npz
# - Uses split indices saved by train.py (--split-save)
# - Loads normalization stats saved by train.py (--norm-path)
# - Runs inference on selected subset (train/val/test)
# - Reports metrics in PHYSICAL space:
#     pred_phys vs Y6 (GT)
# - Reports continuity loss (PHYSICAL) using continuity_loss_cell6
#
# Required:
#   utils/split.py           (load_split)
#   utils/normalization.py   (load_norm, apply_zscore, invert_zscore)
#   utils/losses.py          (continuity_loss_cell6)

import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

from utils.split import load_split
from utils.normalization import load_norm, apply_zscore, invert_zscore
from utils.losses import continuity_loss_cell6


def stats_line(name: str, a: np.ndarray) -> str:
    a = np.asarray(a)
    return (
        f"{name}: min={float(np.nanmin(a)):.6e} "
        f"max={float(np.nanmax(a)):.6e} "
        f"mean={float(np.nanmean(a)):.6e} "
        f"std={float(np.nanstd(a)):.6e}"
    )


def load_cell_npz(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    for k in ("X", "Y"):
        if k not in z.files:
            raise KeyError(f"{npz_path} missing '{k}'. keys={z.files}")

    X = z["X"]  # (T, Ns_cell, 18)
    Y = z["Y"]  # (T, Ns_cell, 6)

    # Keys used in your current train.py expectation:
    if "V_sub_list" not in z.files or "cell_face_sign_sub_list" not in z.files:
        raise KeyError(
            f"{npz_path} must include 'V_sub_list' and 'cell_face_sign_sub_list'. keys={z.files}"
        )

    V = np.asarray(z["V_sub_list"], dtype=np.float32)              # (T, Ns_cell)
    S = np.asarray(z["cell_face_sign_sub_list"], dtype=np.float32) # (T, Ns_cell, 6)

    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(f"Expected X/Y with 3 dims. got X.shape={X.shape} Y.shape={Y.shape}")

    T, Ns, Cin = X.shape
    if Cin != 18:
        # 18 = 6 faces * 3 fields-per-face (default)
        # If you later change, this script supports it via args.
        pass

    if Y.shape[0] != T or Y.shape[1] != Ns or Y.shape[2] != 6:
        raise ValueError(f"Expected Y(T,Ns,6). got Y.shape={Y.shape} vs X.shape={X.shape}")

    if V.shape[0] != T or V.shape[1] != Ns:
        raise ValueError(f"Expected V(T,Ns). got V.shape={V.shape}")
    if S.shape[0] != T or S.shape[1] != Ns or S.shape[2] != 6:
        raise ValueError(f"Expected cell_face_sign_sub_list(T,Ns,6). got S.shape={S.shape}")

    # Flatten along time to match train.py behavior:
    Xf = X.reshape(T * Ns, Cin).astype(np.float32)   # (Ncells,18)
    Yf = Y.reshape(T * Ns, 6).astype(np.float32)     # (Ncells,6)
    Vf = V.reshape(T * Ns).astype(np.float32)        # (Ncells,)
    Sf = S.reshape(T * Ns, 6).astype(np.float32)     # (Ncells,6)

    ds = TensorDataset(
        torch.from_numpy(Xf),
        torch.from_numpy(Yf),
        torch.from_numpy(Sf),
        torch.from_numpy(Vf),
    )
    return ds, (Cin,), (6,), z.files


def compute_metrics(Pred: np.ndarray, Y: np.ndarray) -> dict:
    # Pred, Y: (N,6) in physical space
    err = Pred - Y
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    y_mean = float(np.mean(Y))
    ss_res = float(np.sum((Pred - Y) ** 2))
    ss_tot = float(np.sum((Y - y_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    y_std = float(np.std(Y))
    rel_rmse = float(rmse / y_std) if y_std > 0 else float("nan")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "rel_rmse": rel_rmse}


def main():
    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--jit", required=True, help="TorchScript .jit (MLP)")
    ap.add_argument("--cell_npz", required=True, help="subsampled_ofCell_*.npz")
    ap.add_argument("--split_npz", required=True, help="split indices saved by train.py (--split-save)")
    ap.add_argument("--which", choices=["train", "val", "test"], default="test")

    # Normalization stats (required)
    ap.add_argument("--norm_path", required=True, help="norm stats saved by train.py (--norm-path)")

    # Cell layout
    ap.add_argument("--fields_per_face", type=int, default=3)
    ap.add_argument("--phi_in_face", type=int, default=1)
    ap.add_argument("--num_faces_per_cell", type=int, default=6)

    # Runtime
    ap.add_argument("--batch_cells", type=int, default=8192)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")

    # Outputs
    ap.add_argument("--metrics_csv", default="", help="append metrics to CSV")
    ap.add_argument("--save_pred_npz", default="", help="optional: save predictions (physical) for the chosen split")

    # Debug
    ap.add_argument("--debug_stats", action="store_true")
    ap.add_argument("--debug_norm_stats", action="store_true")

    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    pin_memory = bool(args.pin_memory) and (device == "cuda")

    # Load model
    model = torch.jit.load(args.jit, map_location=device)
    model.eval()

    # Load norm stats (on device)
    x_mean, x_std, y_mean, y_std = load_norm(args.norm_path, device=device)
    if args.debug_norm_stats:
        print("[DBG] norm x_mean:", x_mean.detach().cpu().numpy())
        print("[DBG] norm x_std :", x_std.detach().cpu().numpy())
        print("[DBG] norm y_mean:", y_mean.detach().cpu().numpy())
        print("[DBG] norm y_std :", y_std.detach().cpu().numpy())

    # Load dataset
    dataset, input_shape, output_shape, keys = load_cell_npz(args.cell_npz)
    N = len(dataset)
    print(f"[INFO] cell_npz={os.path.basename(args.cell_npz)}  Ncells={N}  keys={list(keys)}")

    # Load split indices
    train_idx, val_idx, test_idx = load_split(args.split_npz)
    if args.which == "train":
        idx = train_idx
    elif args.which == "val":
        idx = val_idx
    else:
        idx = test_idx

    ds = Subset(dataset, idx)
    print(f"[INFO] split={args.which}  Nsel={len(ds)}  split_npz={os.path.basename(args.split_npz)}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_cells,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Eval buffers (optional saving)
    save_pred = bool(args.save_pred_npz)
    if save_pred:
        Pred_all = np.empty((len(ds), args.num_faces_per_cell), dtype=np.float32)
        Y_all    = np.empty((len(ds), args.num_faces_per_cell), dtype=np.float32)
        Cont_all = np.empty((len(ds),), dtype=np.float32)

    # Accumulators
    cont_sum = 0.0
    cont_cnt = 0
    pred_list = []
    y_list = []

    with torch.no_grad():
        write_ptr = 0
        for batch in loader:
            Xc, Y6, S6, V = batch
            Xc = Xc.to(device)  # (B,18)
            Y6 = Y6.to(device)  # (B,6)
            S6 = S6.to(device)  # (B,6)
            V  = V.to(device)   # (B,)

            B = Xc.shape[0]

            # (B,18) -> (B*6,3)
            Xfaces = Xc.view(B, args.num_faces_per_cell, args.fields_per_face).reshape(
                B * args.num_faces_per_cell, args.fields_per_face
            )

            # Normalize Xfaces -> predict normalized -> inverse to physical
            Xn = apply_zscore(Xfaces, x_mean, x_std)
            pred_n = model(Xn).reshape(B * args.num_faces_per_cell, 1)  # (B*6,1) normalized
            pred_phys = invert_zscore(pred_n, y_mean, y_std).reshape(B, args.num_faces_per_cell)  # (B,6) physical

            # continuity (physical)
            c = continuity_loss_cell6(
                pEqnFlux_pred6=pred_phys,
                X_cell=Xc,
                cell_face_sign=S6,
                V=V,
                fields_per_face=args.fields_per_face,
                phi_in_face=args.phi_in_face,
            )
            c_val = float(c.detach().cpu().item())
            cont_sum += c_val * B
            cont_cnt += B

            # collect for metrics (physical)
            pred_list.append(pred_phys.detach().cpu().numpy().astype(np.float32))
            y_list.append(Y6.detach().cpu().numpy().astype(np.float32))

            if save_pred:
                pb = pred_phys.detach().cpu().numpy().astype(np.float32)
                yb = Y6.detach().cpu().numpy().astype(np.float32)
                Pred_all[write_ptr:write_ptr + B, :] = pb
                Y_all[write_ptr:write_ptr + B, :] = yb
                Cont_all[write_ptr:write_ptr + B] = c_val
                write_ptr += B

    Pred = np.concatenate(pred_list, axis=0)  # (Nsel,6) physical
    Ygt  = np.concatenate(y_list, axis=0)     # (Nsel,6) physical

    metrics = compute_metrics(Pred, Ygt)
    cont_mean = float(cont_sum / max(cont_cnt, 1))
    print(
        f"[INFO] split={args.which} metrics: "
        f"MSE={metrics['mse']:.6e} RMSE={metrics['rmse']:.6e} MAE={metrics['mae']:.6e} "
        f"R2={metrics['r2']:.6f} relRMSE={metrics['rel_rmse']:.6e}  "
        f"cont_mean={cont_mean:.6e}"
    )

    if args.debug_stats:
        print("[DBG] " + stats_line("Ygt", Ygt))
        print("[DBG] " + stats_line("Pred (phys)", Pred))
        print("[DBG] " + stats_line("Err", (Pred - Ygt)))

    # Save predictions (optional)
    if save_pred:
        os.makedirs(os.path.dirname(args.save_pred_npz) or ".", exist_ok=True)
        np.savez_compressed(
            args.save_pred_npz,
            split=args.which,
            pred_pEqnFlux_6=Pred_all,         # (Nsel,6) physical
            gt_pEqnFlux_6=Y_all,              # (Nsel,6) physical
            continuity_loss=Cont_all,         # (Nsel,) repeated scalar per batch (same within batch)
            **{f"metric_{k}": v for k, v in metrics.items()},
            metric_cont_mean=cont_mean,
            cell_npz=os.path.basename(args.cell_npz),
            split_npz=os.path.basename(args.split_npz),
            norm_path=os.path.basename(args.norm_path),
        )
        print(f"[INFO] wrote {args.save_pred_npz}")

    # Append CSV (optional)
    if args.metrics_csv:
        row = {
            "jit": os.path.basename(args.jit),
            "cell_npz": os.path.basename(args.cell_npz),
            "split_npz": os.path.basename(args.split_npz),
            "split": args.which,
            "Nsel": int(len(ds)),
            "norm_path": os.path.basename(args.norm_path),
            "cont_mean": cont_mean,
        }
        row.update(metrics)

        os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
        write_header = (not os.path.exists(args.metrics_csv)) or (os.path.getsize(args.metrics_csv) == 0)
        with open(args.metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"[INFO] appended metrics to {args.metrics_csv}")


if __name__ == "__main__":
    main()



