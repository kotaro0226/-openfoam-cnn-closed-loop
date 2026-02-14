#!/usr/bin/env python3
# smartsim-ml/train.py
#
# Features added:
# - Train/Val/Test split (random, reproducible, saved/loaded via NPZ)
# - Z-score normalization for BOTH X and Y (fit on TRAIN split only)
# - Dropout already supported by your archs/mlp.py (passed via --dropout)
# - For CELL mode: data loss is computed in normalized space,
#   continuity loss is computed in PHYSICAL space by inverse-normalizing pred.
#
# Expected utils:
#   utils/split.py
#   utils/normalization.py
# And existing:
#   utils/losses.py (continuity_loss_cell6)

import argparse
import importlib
import os
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset

from utils.losses import continuity_loss_cell6
from utils.split import make_split_indices, save_split, load_split
from utils.normalization import (
    RunningZScore,
    apply_zscore,
    invert_zscore,
    save_norm,
    load_norm,
)


# ============================================================
# Args
# ============================================================
def parse_args():
    archs = [s.split(".")[0] for s in os.listdir("archs") if s[0:1] != "_"]
    p = argparse.ArgumentParser()

    p.add_argument("arch", type=str, choices=archs, help="model arch in archs/")

    p.add_argument("--src", choices=["random", "redis", "sickle"], default="random")

    p.add_argument("-b", "--batch", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)

    # SICKLE (FACE) datasets
    p.add_argument("--sickle_x", type=str, default="")
    p.add_argument("--sickle_y", type=str, default="")
    p.add_argument("--sickle_z", type=str, default="")
    p.add_argument("--add_orient", action="store_true")

    # SICKLE (CELL) dataset (1 file)
    p.add_argument("--cell_npz", type=str, default="")

    # CELL loss controls
    p.add_argument("--lambda-cont", type=float, default=0.0)
    p.add_argument(
        "--phi-in-face",
        type=int,
        default=1,
        help="0:rAUf 1:phiHbyA 2:div (inside each face triplet)",
    )
    p.add_argument(
        "--fields-per-face",
        type=int,
        default=3,
        help="default 3 (rAUf,phiHbyA,div)",
    )
    p.add_argument("--num-faces-per-cell", type=int, default=6)

    # MLP hyperparams
    p.add_argument("--hidden", nargs="+", type=int, default=[64, 64])
    p.add_argument("--act", type=str, default="tanh")
    p.add_argument("--dropout", type=float, default=0.0)

    # Split
    p.add_argument("--split", type=str, default="0.8,0.1,0.1", help="train,val,test")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--split-save", type=str, default="", help="NPZ path to save split indices")
    p.add_argument("--split-load", type=str, default="", help="NPZ path to load split indices")

    # Normalization (both X and Y)
    # - fit stats on TRAIN split only, save to --norm-path
    # - load stats from --norm-path
    p.add_argument("--norm-path", type=str, default="", help="NPZ path for norm stats")
    p.add_argument("--fit-norm", action="store_true", help="fit norm stats on TRAIN split and save")
    p.add_argument("--load-norm", action="store_true", help="load norm stats from --norm-path")

    # Save / JIT
    p.add_argument("--save-jit", type=str, default="mlp_model.jit")
    p.add_argument("--jit-mode", choices=["trace", "script"], default="trace")

    # Misc
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true", help="pin memory for loaders")

    return p.parse_args()


# ============================================================
# Dataset loaders
# ============================================================
def sickle_npz_xy(npz_path: str, orient_id: Optional[int]):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]  # (T, Ns, Cin)
    Y = z["Y"]  # (T, Ns, Cout)
    T, Ns, Cin = X.shape
    Cout = Y.shape[2]

    Xf = X.reshape(T * Ns, Cin).astype(np.float32)
    Yf = Y.reshape(T * Ns, Cout).astype(np.float32)

    if orient_id is not None:
        orient_col = np.full((Xf.shape[0], 1), float(orient_id), dtype=np.float32)
        Xf = np.concatenate([Xf, orient_col], axis=1)

    return Xf, Yf, (Xf.shape[1],), (Cout,)


def sickle_xy_face(args):
    sets = []
    shapes = None

    def add(path, oid):
        nonlocal shapes
        if not path:
            return
        X, Y, in_shape, out_shape = sickle_npz_xy(path, oid if args.add_orient else None)
        ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32))
        sets.append(ds)
        if shapes is None:
            shapes = (in_shape, out_shape)
        else:
            if shapes[0] != in_shape or shapes[1] != out_shape:
                raise ValueError("Shape mismatch among sickle FACE datasets")

    add(args.sickle_x, 0)
    add(args.sickle_y, 1)
    add(args.sickle_z, 2)

    if not sets:
        raise ValueError("No sickle FACE datasets provided.")
    return ConcatDataset(sets), shapes[0], shapes[1]


def sickle_xy_cell(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)

    X = z["X"]  # (T, Ns_cell, 18)
    Y = z["Y"]  # (T, Ns_cell, 6)

    # NOTE: your SICKLE cell NPZ must contain these:
    if "V_sub_list" not in z or "cell_face_sign_sub_list" not in z:
        raise KeyError("CELL npz must include V_sub_list and cell_face_sign_sub_list")

    V = np.asarray(z["V_sub_list"], dtype=np.float32)              # (T, Ns_cell)
    S = np.asarray(z["cell_face_sign_sub_list"], dtype=np.float32) # (T, Ns_cell, 6)

    T, Ns, Cin = X.shape
    Cout = Y.shape[2]

    Xf = X.reshape(T * Ns, Cin).astype(np.float32)        # (Ncells,18)
    Yf = Y.reshape(T * Ns, Cout).astype(np.float32)       # (Ncells,6)
    Vf = V.reshape(T * Ns).astype(np.float32)             # (Ncells,)
    Sf = S.reshape(T * Ns, 6).astype(np.float32)          # (Ncells,6)

    ds = TensorDataset(
        torch.from_numpy(Xf),
        torch.from_numpy(Yf),
        torch.from_numpy(Sf),
        torch.from_numpy(Vf),
    )
    return ds, (Cin,), (Cout,)


# ============================================================
# Helpers
# ============================================================
def _parse_split(s: str) -> Tuple[float, float, float]:
    tr, va, te = [float(x) for x in s.split(",")]
    if abs((tr + va + te) - 1.0) > 1e-6:
        raise ValueError(f"--split must sum to 1.0, got {s} -> {tr+va+te}")
    if min(tr, va, te) < 0:
        raise ValueError(f"--split must be non-negative, got {s}")
    return tr, va, te


@torch.no_grad()
def _fit_norm_on_train(
    args,
    is_cell: bool,
    ds_train: Subset,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    device: str,
):
    """
    Fit Z-score stats on TRAIN split only.

    FACE:
      X dim = Cin (or Cin+1 if add_orient)
      Y dim = Cout (often 1)

    CELL:
      We normalize the actual MLP input/output space:
      - Xfaces dim = fields_per_face (e.g., 3)
      - Yfaces dim = 1 (pEqnFlux per face)
      (even though dataset stores Xc=18 and Y6=6)
    """
    if is_cell:
        xdim = args.fields_per_face
        ydim = 1
    else:
        xdim = int(input_shape[0])
        ydim = int(output_shape[0])

    x_run = RunningZScore(dim=xdim, device="cpu")
    y_run = RunningZScore(dim=ydim, device="cpu")

    tmp_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=False, num_workers=0)

    for batch in tmp_loader:
        if is_cell:
            Xc, Y6, S6, V = batch
            B = Xc.shape[0]
            Xfaces = Xc.view(B, args.num_faces_per_cell, args.fields_per_face).reshape(B * 6, args.fields_per_face)
            Yfaces = Y6.reshape(B * 6, 1)
            x_run.update(Xfaces.float())
            y_run.update(Yfaces.float())
        else:
            X, Y = batch
            x_run.update(X.float())
            y_run.update(Y.float())

    x_mean, x_std = x_run.finalize()
    y_mean, y_std = y_run.finalize()

    # Move stats to training device
    x_mean = x_mean.to(device)
    x_std  = x_std.to(device)
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)
    return x_mean, x_std, y_mean, y_std


def _build_model(args, face_in_shape: Tuple[int, ...]):
    model_module = importlib.import_module("archs." + args.arch)
    model = model_module.build_model(
        face_in_shape,
        hidden=args.hidden,
        act=args.act,
        dropout=args.dropout,
    )
    return model


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    if args.src != "sickle":
        raise ValueError("This script expects --src sickle for your current pipeline.")

    is_cell = bool(args.cell_npz)

    # --- dataset load ---
    if is_cell:
        dataset, input_shape, output_shape = sickle_xy_cell(args.cell_npz)
        face_in = (args.fields_per_face,)
    else:
        dataset, input_shape, output_shape = sickle_xy_face(args)
        face_in = input_shape  # face pipeline as-is

    # NOTE: if add_orient is used, MLP input becomes 4, but your archs/mlp.py currently enforces 3.
    # Keep it explicit so you don't silently train nonsense.
    if (not is_cell) and args.add_orient and face_in[0] != 3:
        # face_in is (Cin+1,) when add_orient. Your current mlp.py will likely error.
        # If you add a new arch that supports this, it will work.
        pass

    # --- device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = bool(args.pin_memory) and (device == "cuda")

    # --- split indices ---
    tr, va, te = _parse_split(args.split)
    N = len(dataset)

    if args.split_load:
        train_idx, val_idx, test_idx = load_split(args.split_load)
        print(f"[INFO] loaded split: {args.split_load}  (N={N})")
    else:
        train_idx, val_idx, test_idx = make_split_indices(N, tr, va, te, args.seed)
        if args.split_save:
            meta = {"N": N, "split": args.split, "seed": args.seed, "is_cell": int(is_cell)}
            save_split(args.split_save, train_idx, val_idx, test_idx, meta=meta)
            print(f"[INFO] saved split: {args.split_save}  (N={N})")

    ds_train = Subset(dataset, train_idx)
    ds_val   = Subset(dataset, val_idx)
    ds_test  = Subset(dataset, test_idx)

    loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    loader_val   = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    loader_test  = DataLoader(ds_test, batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)

    # --- normalization stats ---
    if (args.fit_norm or args.load_norm) and (not args.norm_path):
        raise ValueError("--fit-norm/--load-norm requires --norm-path")

    if args.fit_norm:
        x_mean, x_std, y_mean, y_std = _fit_norm_on_train(
            args=args,
            is_cell=is_cell,
            ds_train=ds_train,
            input_shape=input_shape,
            output_shape=output_shape,
            device=device,
        )
        save_norm(
            args.norm_path,
            x_mean, x_std,
            y_mean, y_std,
            meta={"fit_on": "train", "split": args.split, "seed": args.seed, "is_cell": int(is_cell)},
        )
        print(f"[INFO] saved norm stats: {args.norm_path}")
    elif args.load_norm:
        x_mean, x_std, y_mean, y_std = load_norm(args.norm_path, device=device)
        print(f"[INFO] loaded norm stats: {args.norm_path}")
    else:
        raise ValueError(
            "You asked to 'normalize everything'. Use either --fit-norm or --load-norm with --norm-path."
        )

    # --- model ---
    model = _build_model(args, face_in).to(device)

    # --- optim ---
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ============================================================
    # Train
    # ============================================================
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in loader_train:
            optimizer.zero_grad(set_to_none=True)

            if is_cell:
                Xc, Y6, S6, V = batch
                Xc = Xc.to(device)   # (B,18)
                Y6 = Y6.to(device)   # (B,6)
                S6 = S6.to(device)   # (B,6)
                V  = V.to(device)    # (B,)

                B = Xc.shape[0]
                # MLP input per face: (B*6, 3)
                Xfaces = Xc.view(B, args.num_faces_per_cell, args.fields_per_face).reshape(B * 6, args.fields_per_face)
                # Normalize inputs
                Xn = apply_zscore(Xfaces, x_mean, x_std)

                # Model output in normalized space: (B,6)
                pred_n = model(Xn).view(B, args.num_faces_per_cell)

                # Normalize targets (Y per face)
                Yn = apply_zscore(Y6.reshape(B * 6, 1), y_mean, y_std).view(B, args.num_faces_per_cell)

                loss_data = criterion(pred_n, Yn)

                # Continuity loss uses PHYSICAL pEqnFlux:
                loss_cont = torch.tensor(0.0, device=device)
                if args.lambda_cont > 0.0:
                    pred_phys = invert_zscore(pred_n.reshape(B * 6, 1), y_mean, y_std).view(B, args.num_faces_per_cell)
                    loss_cont = continuity_loss_cell6(
                        pEqnFlux_pred6=pred_phys,  # physical
                        X_cell=Xc,                 # physical
                        cell_face_sign=S6,
                        V=V,
                        fields_per_face=args.fields_per_face,
                        phi_in_face=args.phi_in_face,
                    )

                loss = loss_data + args.lambda_cont * loss_cont

            else:
                X, Y = batch
                X = X.to(device)
                Y = Y.to(device)

                Xn = apply_zscore(X, x_mean, x_std)
                pred_n = model(Xn)

                Yn = apply_zscore(Y, y_mean, y_std)
                loss = criterion(pred_n, Yn)

            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * (Xc.shape[0] if is_cell else X.shape[0])
            train_count += int(Xc.shape[0] if is_cell else X.shape[0])

        train_loss = train_loss_sum / max(train_count, 1)

        # ============================================================
        # Val
        # ============================================================
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_count = 0
            for batch in loader_val:
                if is_cell:
                    Xc, Y6, S6, V = batch
                    Xc = Xc.to(device)
                    Y6 = Y6.to(device)
                    B = Xc.shape[0]
                    Xfaces = Xc.view(B, args.num_faces_per_cell, args.fields_per_face).reshape(B * 6, args.fields_per_face)
                    pred_n = model(apply_zscore(Xfaces, x_mean, x_std)).view(B, args.num_faces_per_cell)
                    Yn = apply_zscore(Y6.reshape(B * 6, 1), y_mean, y_std).view(B, args.num_faces_per_cell)
                    l = criterion(pred_n, Yn)
                    val_loss_sum += float(l.item()) * B
                    val_count += B
                else:
                    X, Y = batch
                    X = X.to(device); Y = Y.to(device)
                    l = criterion(
                        model(apply_zscore(X, x_mean, x_std)),
                        apply_zscore(Y, y_mean, y_std),
                    )
                    val_loss_sum += float(l.item()) * X.shape[0]
                    val_count += int(X.shape[0])

            val_loss = val_loss_sum / max(val_count, 1)

        print(f"[INFO] epoch {epoch+1}/{args.epochs} train_loss={train_loss:.6e} val_loss={val_loss:.6e}")

    # ============================================================
    # Test (normalized-space MSE)
    # ============================================================
    model.eval()
    with torch.no_grad():
        test_loss_sum = 0.0
        test_count = 0
        for batch in loader_test:
            if is_cell:
                Xc, Y6, S6, V = batch
                Xc = Xc.to(device)
                Y6 = Y6.to(device)
                B = Xc.shape[0]
                Xfaces = Xc.view(B, args.num_faces_per_cell, args.fields_per_face).reshape(B * 6, args.fields_per_face)
                pred_n = model(apply_zscore(Xfaces, x_mean, x_std)).view(B, args.num_faces_per_cell)
                Yn = apply_zscore(Y6.reshape(B * 6, 1), y_mean, y_std).view(B, args.num_faces_per_cell)
                l = criterion(pred_n, Yn)
                test_loss_sum += float(l.item()) * B
                test_count += B
            else:
                X, Y = batch
                X = X.to(device); Y = Y.to(device)
                l = criterion(
                    model(apply_zscore(X, x_mean, x_std)),
                    apply_zscore(Y, y_mean, y_std),
                )
                test_loss_sum += float(l.item()) * X.shape[0]
                test_count += int(X.shape[0])

        test_loss = test_loss_sum / max(test_count, 1)
    print(f"[INFO] test_loss={test_loss:.6e}")

    # ============================================================
    # JIT save
    # ============================================================
    model.eval()
    example = torch.rand(1, *face_in, device=device)

    if args.jit_mode == "script":
        scripted = torch.jit.script(model)
    else:
        scripted = torch.jit.trace(model, example)

    os.makedirs(os.path.dirname(args.save_jit) or ".", exist_ok=True)
    scripted.save(args.save_jit)
    print(f"[INFO] saved jit: {args.save_jit}")

    # Helpful reminder for inference:
    print("[INFO] NOTE: For inference, load norm stats from --norm-path and apply to inputs,")
    print("             then inverse-normalize outputs using y_mean/y_std before writing pred.")


if __name__ == "__main__":
    main()




