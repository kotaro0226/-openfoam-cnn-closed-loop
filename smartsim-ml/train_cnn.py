#!/usr/bin/env python3

import argparse
import csv
import math
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from utils.losses import continuity_loss_cell6
from utils.split import load_split, make_split_indices, save_split


class GridPatchDataset(Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=True)
        if "X" not in z or "Y" not in z:
            raise KeyError("dataset npz must include X and Y")

        X = np.asarray(z["X"], dtype=np.float32)
        Y = np.asarray(z["Y"], dtype=np.float32)
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(f"expected X,Y as (TB,Ns,C), got X={X.shape} Y={Y.shape}")

        tb, ns, cin = X.shape
        tb2, ns2, cout = Y.shape
        if (tb2, ns2) != (tb, ns):
            raise ValueError(f"X/Y shape mismatch: X={X.shape}, Y={Y.shape}")

        side = int(round(math.sqrt(ns)))
        if side * side != ns:
            raise ValueError(f"Ns must be a perfect square for 2D CNN, got Ns={ns}")

        self.X = torch.from_numpy(X.reshape(tb, side, side, cin).transpose(0, 3, 1, 2))
        self.Y = torch.from_numpy(Y.reshape(tb, side, side, cout).transpose(0, 3, 1, 2))
        self.side = side
        self.in_channels = cin
        self.out_channels = cout

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


class SparseGrid3DDataset(Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=True)
        for key in ("X", "Y", "indices_list", "dims_cells", "V_sub_list", "cell_face_sign_sub_list"):
            if key not in z:
                raise KeyError(f"dataset npz must include {key}")

        self.X_src = np.asarray(z["X"], dtype=np.float32)
        self.Y_src = np.asarray(z["Y"], dtype=np.float32)
        self.indices = z["indices_list"]
        self.dims = tuple(int(v) for v in np.asarray(z["dims_cells"], dtype=np.int32))
        self.V_sub = np.asarray(z["V_sub_list"], dtype=np.float32)
        self.S_sub = np.asarray(z["cell_face_sign_sub_list"], dtype=np.int8)
        tb, ns, cin = self.X_src.shape
        tb2, ns2, cout = self.Y_src.shape
        if (tb2, ns2) != (tb, ns):
            raise ValueError(f"X/Y shape mismatch: X={self.X_src.shape}, Y={self.Y_src.shape}")
        if len(self.indices) != tb:
            raise ValueError(f"indices_list length mismatch: {len(self.indices)} vs TB={tb}")

        self.n_samples = tb
        self.ns = ns
        self.in_channels = cin
        self.out_channels = cout
        if self.V_sub.shape != (tb, ns):
            raise ValueError(f"Expected V_sub_list shape {(tb, ns)}, got {self.V_sub.shape}")
        if self.S_sub.shape != (tb, ns, 6):
            raise ValueError(f"Expected cell_face_sign_sub_list shape {(tb, ns, 6)}, got {self.S_sub.shape}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        nx, ny, nz = self.dims
        x_grid = np.zeros((self.in_channels, nx, ny, nz), dtype=np.float32)

        ids = np.asarray(self.indices[idx], dtype=np.int64)
        vals_x = self.X_src[idx]
        vals_y = self.Y_src[idx]

        k = ids % nz
        j = (ids // nz) % ny
        i = ids // (ny * nz)

        x_grid[:, i, j, k] = vals_x.T

        return (
            torch.from_numpy(x_grid),
            torch.from_numpy(ids.astype(np.int64, copy=False)),
            torch.from_numpy(vals_x.astype(np.float32, copy=False)),
            torch.from_numpy(vals_y.astype(np.float32, copy=False)),
            torch.from_numpy(self.V_sub[idx].astype(np.float32, copy=False)),
            torch.from_numpy(self.S_sub[idx].astype(np.float32, copy=False)),
        )


class DenseBlock3DDataset(Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=True)
        for key in ("X", "Y", "V", "cell_face_sign", "core_shape"):
            if key not in z:
                raise KeyError(f"dataset npz must include {key}")

        self.X = torch.from_numpy(np.asarray(z["X"], dtype=np.float32))
        self.Y = torch.from_numpy(np.asarray(z["Y"], dtype=np.float32))
        self.V = torch.from_numpy(np.asarray(z["V"], dtype=np.float32))
        self.S = torch.from_numpy(np.asarray(z["cell_face_sign"], dtype=np.float32))
        self.core_shape = tuple(int(v) for v in np.asarray(z["core_shape"], dtype=np.int32))
        self.halo = int(np.asarray(z["halo"], dtype=np.int32)) if "halo" in z else 0
        self.kept_times = np.asarray(z["kept_times"], dtype=np.int32) if "kept_times" in z else None

        if self.X.ndim != 5 or self.Y.ndim != 5:
            raise ValueError(f"Expected X,Y as 5D tensors, got X={self.X.shape} Y={self.Y.shape}")
        if self.V.ndim != 4 or self.S.ndim != 5:
            raise ValueError(f"Expected V as 4D and S as 5D, got V={self.V.shape} S={self.S.shape}")

        bsz, cin, dx, dy, dz = self.X.shape
        bsz2, cout, cx, cy, cz = self.Y.shape
        if (bsz2, cx, cy, cz) != (bsz, *self.core_shape):
            raise ValueError(
                f"X/Y shape mismatch or core_shape mismatch: X={self.X.shape} Y={self.Y.shape} core={self.core_shape}"
            )
        if self.V.shape != (bsz, *self.core_shape):
            raise ValueError(f"Expected V shape {(bsz, *self.core_shape)}, got {self.V.shape}")
        if self.S.shape != (bsz, 6, *self.core_shape):
            raise ValueError(f"Expected cell_face_sign shape {(bsz, 6, *self.core_shape)}, got {self.S.shape}")

        self.in_channels = cin
        self.out_channels = cout
        self.input_dims = (dx, dy, dz)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.V[idx], self.S[idx]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-npz", required=True)
    p.add_argument("--grid-mode", choices=["2d-flat", "3d-sparse", "3d-block"], default="3d-sparse")
    p.add_argument("--split", type=str, default="0.8,0.1,0.1")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--split-save", type=str, default="")
    p.add_argument("--split-load", type=str, default="")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 disables)")
    p.add_argument("--phase1-steps", type=int, default=0)
    p.add_argument("--phase2-steps", type=int, default=0)
    p.add_argument("--lr-phase1", type=float, default=None)
    p.add_argument("--lr-phase2", type=float, default=None)
    p.add_argument("--lambda-cont", type=float, default=0.0)
    p.add_argument(
        "--lambda-cont-warmup-steps",
        type=int,
        default=0,
        help="optimizer steps to linearly ramp continuity-loss weight from 0 to --lambda-cont",
    )
    p.add_argument("--phi-in-face", type=int, default=1)
    p.add_argument("--fields-per-face", type=int, default=3)
    p.add_argument(
        "--block-data-loss",
        choices=["full", "masked"],
        default="full",
        help="for 3d-block mode, use full-block MSE or patch-masked MSE on the core target",
    )
    p.add_argument(
        "--block-mask-ratio",
        type=float,
        default=0.25,
        help="for --block-data-loss masked: fraction of core voxels supervised on average via patch mask",
    )
    p.add_argument(
        "--block-mask-policy",
        choices=["random", "gradient", "entropy"],
        default="random",
        help="for --block-data-loss masked: patch selection policy inside each 3D block",
    )
    p.add_argument(
        "--block-mask-size",
        type=int,
        default=4,
        help="for --block-data-loss masked: edge length of each supervised 3D patch",
    )
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--act", choices=["relu", "gelu", "silu", "elu"], default="gelu")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--norm", choices=["none", "batch", "group"], default="none")
    p.add_argument("--residual", action="store_true")
    p.add_argument(
        "--padding-mode",
        choices=["zeros", "reflect", "replicate", "circular"],
        default="zeros",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0, help="clip gradient norm; 0 disables")
    p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--time-focus-min", type=int, default=-1, help="minimum kept_time to upweight in training; -1 disables")
    p.add_argument("--time-focus-max", type=int, default=-1, help="maximum kept_time to upweight in training; -1 disables")
    p.add_argument("--time-focus-weight", type=float, default=1.0, help="sampling weight for focused kept_time range")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--history-csv", type=str, default="")
    p.add_argument("--batch-loss-csv", type=str, default="")
    p.add_argument("--save-model", type=str, default="")
    p.add_argument("--save-best-model", type=str, default="")
    return p.parse_args()


def parse_split(spec: str) -> Tuple[float, float, float]:
    vals = [float(x) for x in spec.split(",")]
    if len(vals) != 3:
        raise ValueError("--split must be train,val,test")
    if abs(sum(vals) - 1.0) > 1e-6:
        raise ValueError("--split must sum to 1")
    return vals[0], vals[1], vals[2]


def choose_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_kept_time(dataset, sample_index: int) -> int:
    kept_times = getattr(dataset, "kept_times", None)
    if kept_times is None or len(kept_times) == 0 or len(dataset) % len(kept_times) != 0:
        return -1
    blocks_per_time = len(dataset) // len(kept_times)
    return int(kept_times[sample_index // blocks_per_time])


def make_train_sampler(dataset, train_indices, args):
    if args.time_focus_weight <= 1.0 and args.time_focus_min < 0 and args.time_focus_max < 0:
        return None
    if args.time_focus_weight <= 0.0:
        raise ValueError(f"--time-focus-weight must be positive, got {args.time_focus_weight}")
    if getattr(dataset, "kept_times", None) is None:
        raise ValueError("--time-focus-* requires kept_times in the dataset")

    weights = []
    focused = 0
    for idx in train_indices:
        t = sample_kept_time(dataset, int(idx))
        in_range = True
        if args.time_focus_min >= 0 and t < args.time_focus_min:
            in_range = False
        if args.time_focus_max >= 0 and t > args.time_focus_max:
            in_range = False
        w = args.time_focus_weight if in_range else 1.0
        focused += int(in_range)
        weights.append(w)

    if focused == 0:
        raise ValueError(
            f"time focus range selected no train samples: min={args.time_focus_min} max={args.time_focus_max}"
        )
    print(
        f"[INFO] train weighted sampling enabled: focused_samples={focused}/{len(train_indices)} "
        f"time_min={args.time_focus_min} time_max={args.time_focus_max} weight={args.time_focus_weight}",
        flush=True,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def fit_norm(subset: Subset):
    idx = np.asarray(subset.indices, dtype=np.int64)
    if hasattr(subset.dataset, "X"):
        x = subset.dataset.X[idx]
        y = subset.dataset.Y[idx]
        spatial_dims_x = tuple(range(2, x.ndim))
        spatial_dims_y = tuple(range(2, y.ndim))
        x_mean = x.mean(dim=(0, *spatial_dims_x), keepdim=True)
        x_std = x.std(dim=(0, *spatial_dims_x), keepdim=True).clamp_min(1e-6)
        y_mean = y.mean(dim=(0, *spatial_dims_y), keepdim=True)
        y_std = y.std(dim=(0, *spatial_dims_y), keepdim=True).clamp_min(1e-6)
        return x_mean, x_std, y_mean, y_std

    sums_x = None
    sums_x2 = None
    n_x = 0

    for ds_idx in idx:
        x, *_ = subset.dataset[int(ds_idx)]
        x = x.float()
        if sums_x is None:
            sums_x = torch.zeros((x.shape[0],), dtype=torch.float64)
            sums_x2 = torch.zeros((x.shape[0],), dtype=torch.float64)

        sums_x += x.reshape(x.shape[0], -1).double().sum(dim=1)
        sums_x2 += (x.reshape(x.shape[0], -1).double() ** 2).sum(dim=1)
        n_x += x.shape[1] * x.shape[2] * x.shape[3]

    y_src = torch.from_numpy(subset.dataset.Y_src[idx]).float()
    y_mean = y_src.mean(dim=(0, 1), keepdim=True).view(1, -1, 1, 1, 1)
    y_std = y_src.std(dim=(0, 1), keepdim=True).clamp_min(1e-6).view(1, -1, 1, 1, 1)

    x_mean = (sums_x / max(1, n_x)).float().view(1, -1, 1, 1, 1)
    x_var = (sums_x2 / max(1, n_x)).float() - x_mean.view(-1) ** 2
    x_std = x_var.clamp_min(0.0).sqrt().clamp_min(1e-6).view(1, -1, 1, 1, 1)

    return x_mean, x_std, y_mean, y_std


def normalize_batch(x, y, x_mean, x_std, y_mean, y_std):
    return (x - x_mean) / x_std, (y - y_mean) / y_std


def sparse_index_mse(pred: torch.Tensor, target: torch.Tensor, ids: torch.Tensor, dims: Tuple[int, int, int]) -> torch.Tensor:
    pred_sampled = gather_sparse_values(pred, ids, dims)
    return ((pred_sampled - target) ** 2).mean()


def build_patch_mask(
    batch_size: int,
    core_shape: Tuple[int, int, int],
    patch_size: int,
    keep_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    cx, cy, cz = core_shape
    x_starts = list(range(0, cx, patch_size))
    y_starts = list(range(0, cy, patch_size))
    z_starts = list(range(0, cz, patch_size))
    patch_coords = [(sx, sy, sz) for sx in x_starts for sy in y_starts for sz in z_starts]
    n_patches = len(patch_coords)

    keep_patch = torch.rand((batch_size, n_patches), device=device) < keep_ratio
    empty = ~keep_patch.any(dim=1)
    if empty.any():
        fallback = torch.randint(0, n_patches, (int(empty.sum().item()),), device=device)
        keep_patch[empty, fallback] = True

    mask = torch.zeros((batch_size, 1, cx, cy, cz), dtype=torch.float32, device=device)
    for patch_idx, (sx, sy, sz) in enumerate(patch_coords):
        ex = min(sx + patch_size, cx)
        ey = min(sy + patch_size, cy)
        ez = min(sz + patch_size, cz)
        mask[:, :, sx:ex, sy:ey, sz:ez] += keep_patch[:, patch_idx].view(batch_size, 1, 1, 1, 1)

    return mask.clamp_max_(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sqerr = (pred - target) ** 2
    weighted = sqerr * mask
    denom = (mask.sum() * pred.shape[1]).clamp_min(1.0)
    return weighted.sum() / denom


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    if delta <= 0.0:
        raise ValueError(f"huber delta must be positive, got {delta}")
    err = pred - target
    abs_err = err.abs()
    point_loss = torch.where(
        abs_err <= delta,
        0.5 * err ** 2,
        delta * (abs_err - 0.5 * delta),
    )
    weighted = point_loss * mask
    denom = (mask.sum() * pred.shape[1]).clamp_min(1.0)
    return weighted.sum() / denom


def masked_data_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, args) -> torch.Tensor:
    if args.loss == "huber":
        return masked_huber(pred, target, mask, args.huber_delta)
    return masked_mse(pred, target, mask)


def unfold_patch_means(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    bsz, channels, cx, cy, cz = x.shape
    px = math.ceil(cx / patch_size)
    py = math.ceil(cy / patch_size)
    pz = math.ceil(cz / patch_size)
    pad_x = px * patch_size - cx
    pad_y = py * patch_size - cy
    pad_z = pz * patch_size - cz
    if pad_x or pad_y or pad_z:
        x = torch.nn.functional.pad(x, (0, pad_z, 0, pad_y, 0, pad_x))
    x = x.reshape(bsz, channels, px, patch_size, py, patch_size, pz, patch_size)
    return x.mean(dim=(3, 5, 7))


def compute_patch_scores(
    x_center: torch.Tensor,
    patch_size: int,
    fields_per_face: int,
    phi_in_face: int,
    policy: str,
) -> torch.Tensor:
    if policy == "random":
        raise ValueError("random policy does not use patch scores")

    bsz, channels, _, _, _ = x_center.shape
    if fields_per_face <= 0 or channels % fields_per_face != 0:
        raise ValueError(
            f"in_channels={channels} must be divisible by fields_per_face={fields_per_face}"
        )

    n_faces = channels // fields_per_face
    x_faces = x_center.view(bsz, n_faces, fields_per_face, *x_center.shape[2:])
    phi = x_faces[:, :, phi_in_face].mean(dim=1, keepdim=True)

    if policy == "gradient":
        grad = torch.zeros_like(phi)
        grad[:, :, 1:, :, :] += (phi[:, :, 1:, :, :] - phi[:, :, :-1, :, :]).abs()
        grad[:, :, :-1, :, :] += (phi[:, :, 1:, :, :] - phi[:, :, :-1, :, :]).abs()
        grad[:, :, :, 1:, :] += (phi[:, :, :, 1:, :] - phi[:, :, :, :-1, :]).abs()
        grad[:, :, :, :-1, :] += (phi[:, :, :, 1:, :] - phi[:, :, :, :-1, :]).abs()
        grad[:, :, :, :, 1:] += (phi[:, :, :, :, 1:] - phi[:, :, :, :, :-1]).abs()
        grad[:, :, :, :, :-1] += (phi[:, :, :, :, 1:] - phi[:, :, :, :, :-1]).abs()
        return unfold_patch_means(grad, patch_size).reshape(bsz, -1)

    if policy == "entropy":
        patch_means = unfold_patch_means(phi, patch_size)
        phi_centered = phi - phi.mean(dim=(2, 3, 4), keepdim=True)
        patch_var = unfold_patch_means(phi_centered ** 2, patch_size)
        entropy_like = torch.log1p(patch_var / patch_means.abs().clamp_min(1e-6))
        return entropy_like.reshape(bsz, -1)

    raise ValueError(f"Unknown block-mask-policy={policy}")


def build_importance_patch_mask(
    x_center: torch.Tensor,
    core_shape: Tuple[int, int, int],
    patch_size: int,
    keep_ratio: float,
    policy: str,
    fields_per_face: int,
    phi_in_face: int,
) -> torch.Tensor:
    scores = compute_patch_scores(
        x_center=x_center,
        patch_size=patch_size,
        fields_per_face=fields_per_face,
        phi_in_face=phi_in_face,
        policy=policy,
    )
    batch_size = x_center.shape[0]
    n_patches = scores.shape[1]
    keep_count = max(1, int(math.ceil(n_patches * keep_ratio)))
    keep_idx = torch.topk(scores, k=keep_count, dim=1).indices

    cx, cy, cz = core_shape
    x_starts = list(range(0, cx, patch_size))
    y_starts = list(range(0, cy, patch_size))
    z_starts = list(range(0, cz, patch_size))
    patch_coords = [(sx, sy, sz) for sx in x_starts for sy in y_starts for sz in z_starts]

    mask = torch.zeros((batch_size, 1, cx, cy, cz), dtype=torch.float32, device=x_center.device)
    for b in range(batch_size):
        for patch_idx in keep_idx[b].tolist():
            sx, sy, sz = patch_coords[patch_idx]
            ex = min(sx + patch_size, cx)
            ey = min(sy + patch_size, cy)
            ez = min(sz + patch_size, cz)
            mask[b, :, sx:ex, sy:ey, sz:ez] = 1.0
    return mask


def resolve_lr(global_step: int, args) -> float:
    if args.lr_phase1 is None and args.lr_phase2 is None:
        return args.lr

    lr1 = args.lr if args.lr_phase1 is None else args.lr_phase1
    lr2 = lr1 if args.lr_phase2 is None else args.lr_phase2

    if args.phase1_steps <= 0:
        return lr2
    if global_step < args.phase1_steps:
        return lr1
    if args.phase2_steps > 0 and global_step >= (args.phase1_steps + args.phase2_steps):
        return lr2
    return lr2


def resolve_lambda_cont(global_step: int, args) -> float:
    if args.lambda_cont <= 0.0:
        return 0.0
    if args.lambda_cont_warmup_steps <= 0:
        return args.lambda_cont
    progress = min(1.0, max(0.0, global_step / float(args.lambda_cont_warmup_steps)))
    return args.lambda_cont * progress


def invert_zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def gather_sparse_values(grid: torch.Tensor, ids: torch.Tensor, dims: Tuple[int, int, int]) -> torch.Tensor:
    nx, ny, nz = dims
    out = []
    for b in range(grid.shape[0]):
        ids_b = ids[b].long()
        k = ids_b % nz
        j = (ids_b // nz) % ny
        i = ids_b // (ny * nz)
        out.append(grid[b, :, i, j, k].transpose(0, 1))
    return torch.stack(out, dim=0)


def center_crop_3d(x: torch.Tensor, core_shape: Tuple[int, int, int]) -> torch.Tensor:
    _, _, dx, dy, dz = x.shape
    cx, cy, cz = core_shape
    sx = (dx - cx) // 2
    sy = (dy - cy) // 2
    sz = (dz - cz) // 2
    return x[:, :, sx:sx + cx, sy:sy + cy, sz:sz + cz]


def compute_loss_terms(
    model,
    batch,
    criterion,
    device,
    x_mean,
    x_std,
    y_mean,
    y_std,
    mode: str,
    dataset,
    args,
    lambda_cont: float,
):
    if mode == "3d-sparse":
        x, ids, x_cell, y_sample, v_sub, s_sub = batch
        ids = ids.to(device)
        x_cell = x_cell.to(device)
        y_sample = y_sample.to(device)
        v_sub = v_sub.to(device)
        s_sub = s_sub.to(device)
    elif mode == "3d-block":
        x, y, v_sub, s_sub = batch
        ids = None
    else:
        x, y = batch
        ids = None
    x = x.to(device)
    x_phys = x
    if mode != "3d-sparse":
        y = y.to(device)
    if mode == "3d-block":
        v_sub = v_sub.to(device)
        s_sub = s_sub.to(device)

    if mode == "3d-sparse":
        x = (x - x_mean) / x_std
        y_sample = (y_sample - y_mean.view(1, 1, -1)) / y_std.view(1, 1, -1)
    else:
        x, y = normalize_batch(x, y, x_mean, x_std, y_mean, y_std)
    pred = model(x)
    if mode == "3d-sparse":
        pred_sampled = gather_sparse_values(pred, ids, dataset.dims)
        data_loss = ((pred_sampled - y_sample) ** 2).mean()
    elif mode == "3d-block":
        pred_center = center_crop_3d(pred, dataset.core_shape)
        if args.block_data_loss == "masked":
            if args.block_mask_policy == "random":
                mask = build_patch_mask(
                    batch_size=pred_center.shape[0],
                    core_shape=dataset.core_shape,
                    patch_size=args.block_mask_size,
                    keep_ratio=args.block_mask_ratio,
                    device=pred_center.device,
                )
            else:
                x_center = center_crop_3d(x, dataset.core_shape)
                mask = build_importance_patch_mask(
                    x_center=x_center,
                    core_shape=dataset.core_shape,
                    patch_size=args.block_mask_size,
                    keep_ratio=args.block_mask_ratio,
                    policy=args.block_mask_policy,
                    fields_per_face=args.fields_per_face,
                    phi_in_face=args.phi_in_face,
                )
            data_loss = masked_data_loss(pred_center, y, mask, args)
        else:
            data_loss = criterion(pred_center, y)
    else:
        data_loss = criterion(pred, y)

    loss_cont = torch.zeros((), device=device)
    if mode == "3d-sparse" and lambda_cont > 0.0:
        pred_phys = invert_zscore(pred, y_mean, y_std)
        pred_sampled = gather_sparse_values(pred_phys, ids, dataset.dims)
        loss_cont = continuity_loss_cell6(
            pEqnFlux_pred6=pred_sampled.reshape(-1, dataset.out_channels),
            X_cell=x_cell.reshape(-1, x_cell.shape[-1]),
            cell_face_sign=s_sub.reshape(-1, s_sub.shape[-1]),
            V=v_sub.reshape(-1),
            fields_per_face=args.fields_per_face,
            phi_in_face=args.phi_in_face,
        )
    elif mode == "3d-block" and lambda_cont > 0.0:
        pred_phys = invert_zscore(pred_center, y_mean, y_std)
        x_center = center_crop_3d(x_phys, dataset.core_shape)
        x_cell = x_center.permute(0, 2, 3, 4, 1).reshape(-1, dataset.in_channels)
        pred_cell = pred_phys.permute(0, 2, 3, 4, 1).reshape(-1, dataset.out_channels)
        sign_cell = s_sub.permute(0, 2, 3, 4, 1).reshape(-1, 6)
        loss_cont = continuity_loss_cell6(
            pEqnFlux_pred6=pred_cell,
            X_cell=x_cell,
            cell_face_sign=sign_cell,
            V=v_sub.reshape(-1),
            fields_per_face=args.fields_per_face,
            phi_in_face=args.phi_in_face,
        )

    total_loss = data_loss + lambda_cont * loss_cont
    return total_loss, data_loss, loss_cont, x.shape[0]


def eval_epoch(model, loader, criterion, device, x_mean, x_std, y_mean, y_std, mode: str, dataset, args):
    model.eval()
    total = 0.0
    total_data = 0.0
    total_cont = 0.0
    count = 0
    lambda_cont = args.lambda_cont
    with torch.no_grad():
        for batch in loader:
            total_loss, data_loss, cont_loss, batch_size = compute_loss_terms(
                model, batch, criterion, device, x_mean, x_std, y_mean, y_std, mode, dataset, args, lambda_cont
            )
            total += float(total_loss.item()) * batch_size
            total_data += float(data_loss.item()) * batch_size
            total_cont += float(cont_loss.item()) * batch_size
            count += batch_size
    denom = max(1, count)
    return {
        "total_loss": total / denom,
        "data_loss": total_data / denom,
        "cont_loss": total_cont / denom,
    }


def make_checkpoint_payload(model, dataset, mode: str, args, x_mean, x_std, y_mean, y_std, extra: dict | None = None):
    payload = {
        "model_state": model.state_dict(),
        "x_mean": x_mean.cpu(),
        "x_std": x_std.cpu(),
        "y_mean": y_mean.cpu(),
        "y_std": y_std.cpu(),
        "in_channels": dataset.in_channels,
        "out_channels": dataset.out_channels,
        "grid_mode": args.grid_mode,
        "width": args.width,
        "depth": args.depth,
        "act": args.act,
        "dropout": args.dropout,
        "norm": args.norm,
        "residual": args.residual,
        "padding_mode": args.padding_mode,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "loss": args.loss,
        "huber_delta": args.huber_delta,
        "time_focus_min": args.time_focus_min,
        "time_focus_max": args.time_focus_max,
        "time_focus_weight": args.time_focus_weight,
    }
    if mode == "3d-sparse":
        payload["dims"] = dataset.dims
    elif mode == "3d-block":
        payload["input_dims"] = dataset.input_dims
        payload["core_shape"] = dataset.core_shape
    else:
        payload["side"] = dataset.side
    if extra:
        payload.update(extra)
    return payload


def save_checkpoint(path: str, payload: dict, label: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    print(f"[INFO] saved {label}: {path}", flush=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.phi_in_face < 0 or args.phi_in_face >= args.fields_per_face:
        raise ValueError(
            f"--phi-in-face must be in [0, {args.fields_per_face - 1}], got {args.phi_in_face}"
        )
    device = choose_device(args.device)
    if args.grid_mode == "2d-flat":
        dataset = GridPatchDataset(args.cell_npz)
        mode = "2d-flat"
        input_shape = (dataset.in_channels, dataset.side, dataset.side)
    elif args.grid_mode == "3d-block":
        dataset = DenseBlock3DDataset(args.cell_npz)
        mode = "3d-block"
        input_shape = (dataset.in_channels, *dataset.input_dims)
    else:
        dataset = SparseGrid3DDataset(args.cell_npz)
        mode = "3d-sparse"
        input_shape = (dataset.in_channels, *dataset.dims)

    if args.split_load:
        tr_idx, va_idx, te_idx = load_split(args.split_load)
    else:
        tr, va, te = parse_split(args.split)
        tr_idx, va_idx, te_idx = make_split_indices(len(dataset), tr, va, te, seed=args.seed)
        if args.split_save:
            save_split(args.split_save, tr_idx, va_idx, te_idx)

    train_ds = Subset(dataset, tr_idx)
    val_ds = Subset(dataset, va_idx)
    test_ds = Subset(dataset, te_idx)

    train_sampler = make_train_sampler(dataset, tr_idx, args)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    x_mean, x_std, y_mean, y_std = fit_norm(train_ds)
    x_mean = x_mean.to(device)
    x_std = x_std.to(device)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)

    from archs.cnn import build_model

    model = build_model(
        input_shape,
        out_channels=dataset.out_channels,
        width=args.width,
        depth=args.depth,
        act=args.act,
        dropout=args.dropout,
        norm=args.norm,
        residual=args.residual,
        padding_mode=args.padding_mode,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[INFO] model={model.__class__.__name__} params={n_params} "
        f"width={args.width} depth={args.depth} act={args.act} norm={args.norm} "
        f"residual={args.residual} padding_mode={args.padding_mode} "
        f"loss={args.loss} huber_delta={args.huber_delta}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.loss == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.HuberLoss(delta=args.huber_delta)

    history = []
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    best_step = 0
    batch_loss_file = None
    batch_loss_writer = None
    if args.batch_loss_csv:
        os.makedirs(os.path.dirname(args.batch_loss_csv) or ".", exist_ok=True)
        batch_loss_file = open(args.batch_loss_csv, "w", newline="")
        batch_loss_writer = csv.DictWriter(
            batch_loss_file,
            fieldnames=[
                "epoch",
                "step",
                "batch_size",
                "lr",
                "lambda_cont",
                "train_batch_loss",
                "train_data_loss",
                "train_cont_loss",
            ],
        )
        batch_loss_writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        total_data = 0.0
        total_cont = 0.0
        count = 0
        stop_early = False
        for batch in train_loader:
            current_lr = resolve_lr(global_step, args)
            current_lambda_cont = resolve_lambda_cont(global_step, args)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            loss, data_loss, loss_cont, batch_size = compute_loss_terms(
                model, batch, criterion, device, x_mean, x_std, y_mean, y_std, mode, dataset, args, current_lambda_cont
            )
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.item()) * batch_size
            total_data += float(data_loss.item()) * batch_size
            total_cont += float(loss_cont.item()) * batch_size
            count += batch_size
            global_step += 1
            if batch_loss_writer is not None:
                batch_loss_writer.writerow(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "batch_size": batch_size,
                        "lr": optimizer.param_groups[0]["lr"],
                        "lambda_cont": current_lambda_cont,
                        "train_batch_loss": float(loss.item()),
                        "train_data_loss": float(data_loss.item()),
                        "train_cont_loss": float(loss_cont.item()),
                    }
                )
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_early = True
                break

        denom = max(1, count)
        train_total_loss = total / denom
        train_data_loss = total_data / denom
        train_cont_loss = total_cont / denom
        val_metrics = eval_epoch(model, val_loader, criterion, device, x_mean, x_std, y_mean, y_std, mode, dataset, args)
        history.append(
            epoch_record := {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_total_loss,
                "train_total_loss": train_total_loss,
                "train_data_loss": train_data_loss,
                "train_cont_loss": train_cont_loss,
                "val_loss": val_metrics["total_loss"],
                "val_total_loss": val_metrics["total_loss"],
                "val_data_loss": val_metrics["data_loss"],
                "val_cont_loss": val_metrics["cont_loss"],
            }
        )
        if args.save_best_model and val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            best_step = global_step
            payload = make_checkpoint_payload(
                model,
                dataset,
                mode,
                args,
                x_mean,
                x_std,
                y_mean,
                y_std,
                extra={
                    "checkpoint_kind": "best",
                    "best_metric": "val_total_loss",
                    "best_epoch": best_epoch,
                    "best_step": best_step,
                    "best_metrics": epoch_record,
                },
            )
            save_checkpoint(args.save_best_model, payload, "best model")
        print(
            f"[INFO] epoch {epoch}/{args.epochs} "
            f"train_total={train_total_loss:.6e} train_data={train_data_loss:.6e} train_cont={train_cont_loss:.6e} "
            f"val_total={val_metrics['total_loss']:.6e} val_data={val_metrics['data_loss']:.6e} val_cont={val_metrics['cont_loss']:.6e} "
            f"lr={optimizer.param_groups[0]['lr']:.6e} "
            f"lambda_cont={resolve_lambda_cont(global_step, args):.6e}",
            flush=True,
        )
        if stop_early:
            print(f"[INFO] reached max_steps={args.max_steps}; stopping training early", flush=True)
            break

    test_metrics = eval_epoch(model, test_loader, criterion, device, x_mean, x_std, y_mean, y_std, mode, dataset, args)
    print(
        f"[INFO] test_total={test_metrics['total_loss']:.6e} "
        f"test_data={test_metrics['data_loss']:.6e} "
        f"test_cont={test_metrics['cont_loss']:.6e}",
        flush=True,
    )

    if args.history_csv:
        os.makedirs(os.path.dirname(args.history_csv) or ".", exist_ok=True)
        with open(args.history_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "step",
                    "train_loss",
                    "train_total_loss",
                    "train_data_loss",
                    "train_cont_loss",
                    "val_loss",
                    "val_total_loss",
                    "val_data_loss",
                    "val_cont_loss",
                ],
            )
            w.writeheader()
            w.writerows(history)

    if args.save_model:
        payload = make_checkpoint_payload(
            model,
            dataset,
            mode,
            args,
            x_mean,
            x_std,
            y_mean,
            y_std,
            extra={
                "checkpoint_kind": "final",
                "final_step": global_step,
                "best_epoch": best_epoch,
                "best_step": best_step,
                "best_val_loss": best_val_loss,
            },
        )
        save_checkpoint(args.save_model, payload, "model")

    if batch_loss_file is not None:
        batch_loss_file.close()
        print(f"[INFO] saved batch losses: {args.batch_loss_csv}", flush=True)


if __name__ == "__main__":
    main()
