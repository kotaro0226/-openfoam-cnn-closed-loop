#!/usr/bin/env python3

import argparse
import csv
import heapq
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from archs.cnn import build_model
from train_cnn import DenseBlock3DDataset, center_crop_3d
from utils.split import load_split, make_split_indices


def parse_split(spec: str) -> tuple[float, float, float]:
    vals = [float(x) for x in spec.split(",")]
    if len(vals) != 3:
        raise ValueError("--split-ratio must be train,val,test")
    if abs(sum(vals) - 1.0) > 1e-6:
        raise ValueError("--split-ratio must sum to 1")
    return vals[0], vals[1], vals[2]


def choose_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    required = [
        "model_state",
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
        "in_channels",
        "out_channels",
        "input_dims",
        "core_shape",
    ]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(f"checkpoint is missing required keys: {missing}")
    if ckpt.get("grid_mode") != "3d-block":
        raise ValueError(f"expected a 3d-block checkpoint, got grid_mode={ckpt.get('grid_mode')}")
    return ckpt


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(
        (int(ckpt["in_channels"]), *tuple(int(v) for v in ckpt["input_dims"])),
        out_channels=int(ckpt["out_channels"]),
        width=int(ckpt.get("width", 8)),
        depth=int(ckpt.get("depth", 3)),
        act=str(ckpt.get("act", "gelu")),
        dropout=float(ckpt.get("dropout", 0.0)),
        norm=str(ckpt.get("norm", "none")),
        residual=bool(ckpt.get("residual", False)),
        padding_mode=str(ckpt.get("padding_mode", "zeros")),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def split_indices(dataset_len: int, args) -> np.ndarray:
    if args.split_load:
        train_idx, val_idx, test_idx = load_split(args.split_load)
    else:
        tr, va, te = parse_split(args.split_ratio)
        train_idx, val_idx, test_idx = make_split_indices(dataset_len, tr, va, te, seed=args.seed)

    choices = {
        "train": np.asarray(train_idx, dtype=np.int64),
        "val": np.asarray(val_idx, dtype=np.int64),
        "test": np.asarray(test_idx, dtype=np.int64),
        "all": np.arange(dataset_len, dtype=np.int64),
    }
    out = choices[args.eval_split]
    if args.max_samples > 0:
        out = out[: args.max_samples]
    return out


def sample_kept_time(sample_index: int, kept_times: np.ndarray | None, blocks_per_time: int | None) -> int:
    if kept_times is None or blocks_per_time is None:
        return -1
    return int(kept_times[sample_index // blocks_per_time])


def filter_indices_by_time(
    indices: np.ndarray,
    kept_times: np.ndarray | None,
    blocks_per_time: int | None,
    time_min: int,
    time_max: int,
) -> np.ndarray:
    if time_min < 0 and time_max < 0:
        return indices
    if kept_times is None or blocks_per_time is None:
        raise ValueError("--time-min/--time-max require kept_times in the dataset")

    out = []
    for idx in indices:
        t = sample_kept_time(int(idx), kept_times, blocks_per_time)
        if time_min >= 0 and t < time_min:
            continue
        if time_max >= 0 and t > time_max:
            continue
        out.append(int(idx))
    return np.asarray(out, dtype=np.int64)


def summarize_split_time_counts(dataset_len: int, kept_times: np.ndarray | None, blocks_per_time: int | None, args) -> list[dict]:
    if kept_times is None or blocks_per_time is None:
        return []
    if args.split_load:
        split_map = dict(zip(("train", "val", "test"), load_split(args.split_load)))
    else:
        tr, va, te = parse_split(args.split_ratio)
        split_map = dict(zip(("train", "val", "test"), make_split_indices(dataset_len, tr, va, te, seed=args.seed)))

    rows = []
    for split_name, split_idx in split_map.items():
        counts = {int(t): 0 for t in kept_times}
        for idx in np.asarray(split_idx, dtype=np.int64):
            counts[sample_kept_time(int(idx), kept_times, blocks_per_time)] += 1
        for t in sorted(counts):
            rows.append({"split": split_name, "kept_time": t, "num_samples": counts[t]})
    return rows


def divergence_field(
    x_core: torch.Tensor,
    y_flux: torch.Tensor,
    signs: torch.Tensor,
    volumes: torch.Tensor,
    fields_per_face: int,
    phi_in_face: int,
) -> torch.Tensor:
    bsz, channels, cx, cy, cz = x_core.shape
    if channels % fields_per_face != 0:
        raise ValueError(f"in_channels={channels} is not divisible by fields_per_face={fields_per_face}")
    n_faces = channels // fields_per_face
    if n_faces != y_flux.shape[1]:
        raise ValueError(f"expected y channels to match faces: X has {n_faces}, Y has {y_flux.shape[1]}")

    x_faces = x_core.view(bsz, n_faces, fields_per_face, cx, cy, cz)
    phi_hbya = x_faces[:, :, phi_in_face]
    phi = phi_hbya - y_flux
    net_flux = (phi * signs).sum(dim=1)
    return net_flux / volumes.clamp_min(1e-6)


def update_basic_sums(sums: dict, pred: torch.Tensor, target: torch.Tensor) -> None:
    err = pred - target
    sums["sq"] += float((err * err).sum().detach().cpu())
    sums["abs"] += float(err.abs().sum().detach().cpu())
    sums["n"] += int(err.numel())


def summarize_abs(name: str, values: list[np.ndarray]) -> dict:
    if not values:
        return {}
    a = np.concatenate(values).astype(np.float64, copy=False)
    return {
        f"{name}_mean_abs": float(np.mean(a)),
        f"{name}_max_abs": float(np.max(a)),
        f"{name}_p95_abs": float(np.percentile(a, 95)),
        f"{name}_p99_abs": float(np.percentile(a, 99)),
        f"{name}_mean_square": float(np.mean(a * a)),
    }


def make_region_masks(core_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    cx, cy, cz = core_shape
    i, j, k = np.indices(core_shape)
    dist = np.minimum.reduce([i, j, k, cx - 1 - i, cy - 1 - j, cz - 1 - k])
    return {
        "all": np.ones(core_shape, dtype=bool),
        "boundary_1": dist == 0,
        "boundary_2": dist <= 1,
        "interior_excluding_boundary_1": dist >= 1,
        "interior_excluding_boundary_2": dist >= 2,
        "face_x_min": i == 0,
        "face_x_max": i == cx - 1,
        "face_y_min": j == 0,
        "face_y_max": j == cy - 1,
        "face_z_min": k == 0,
        "face_z_max": k == cz - 1,
    }


def summarize_region_values(region_values: dict[str, list[np.ndarray]], value_name: str) -> list[dict]:
    rows = []
    for region, chunks in region_values.items():
        if not chunks:
            continue
        values = np.concatenate(chunks).astype(np.float64, copy=False)
        rows.append(
            {
                "region": region,
                "value": value_name,
                "n": int(values.size),
                "mean_abs": float(np.mean(values)),
                "max_abs": float(np.max(values)),
                "p50_abs": float(np.percentile(values, 50)),
                "p95_abs": float(np.percentile(values, 95)),
                "p99_abs": float(np.percentile(values, 99)),
                "mean_square": float(np.mean(values * values)),
            }
        )
    return rows


def summarize_group_rows(rows: list[dict], group_key: str, metric_keys: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)

    out = []
    for group_value, group_rows in sorted(groups.items(), key=lambda item: int(item[0])):
        item = {group_key: group_value, "num_samples": len(group_rows)}
        for key in metric_keys:
            vals = np.asarray([float(r[key]) for r in group_rows], dtype=np.float64)
            item[f"{key}_mean"] = float(np.mean(vals))
            item[f"{key}_max"] = float(np.max(vals))
            item[f"{key}_p95"] = float(np.percentile(vals, 95))
        out.append(item)
    return out


def center_slice(a: np.ndarray, axis: str, index: int | None) -> np.ndarray:
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    i = a.shape[ax] // 2 if index is None else index
    if i < 0 or i >= a.shape[ax]:
        raise ValueError(f"slice index {i} out of bounds for axis {axis} with size {a.shape[ax]}")
    return np.take(a, i, axis=ax)


def add_image(fig, axes, pos, image, title, cmap="viridis", symmetric=False):
    ax = axes[pos]
    if symmetric:
        vmax = float(np.nanmax(np.abs(image)))
        vmin = -vmax
    else:
        vmin = vmax = None
    im = ax.imshow(image.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_sample(sample: dict, out_path: Path, channel: int, axis: str, slice_index: int | None) -> None:
    target = sample["target"][channel]
    pred = sample["pred"][channel]
    err = pred - target
    div_pred = sample["div_pred"]
    div_target = sample["div_target"]
    div_err = div_pred - div_target

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    axes = axes.reshape(-1)
    add_image(fig, axes, 0, center_slice(target, axis, slice_index), f"target ch{channel}")
    add_image(fig, axes, 1, center_slice(pred, axis, slice_index), f"pred ch{channel}")
    add_image(fig, axes, 2, center_slice(err, axis, slice_index), f"error ch{channel}", cmap="coolwarm", symmetric=True)
    add_image(fig, axes, 3, center_slice(div_target, axis, slice_index), "target div", cmap="coolwarm", symmetric=True)
    add_image(fig, axes, 4, center_slice(div_pred, axis, slice_index), "pred div", cmap="coolwarm", symmetric=True)
    add_image(fig, axes, 5, center_slice(div_err, axis, slice_index), "div error", cmap="coolwarm", symmetric=True)
    fig.suptitle(f"sample_index={sample['sample_index']} block_start={sample['block_start'].tolist()}", fontsize=11)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_metrics(path: Path, metrics: dict) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_sample_payload(
    sample_index: int,
    block_start: np.ndarray,
    kept_time: int,
    target: np.ndarray,
    pred: np.ndarray,
    div_target: np.ndarray,
    div_pred: np.ndarray,
) -> dict:
    return {
        "sample_index": int(sample_index),
        "block_start": np.asarray(block_start, dtype=np.int32),
        "kept_time": int(kept_time),
        "target": target,
        "pred": pred,
        "div_pred": div_pred,
        "div_target": div_target,
    }


def save_sample_npz(path: Path, samples: list[dict]) -> None:
    np.savez_compressed(
        path,
        sample_index=np.asarray([s["sample_index"] for s in samples], dtype=np.int64),
        block_start=np.stack([s["block_start"] for s in samples], axis=0) if samples else np.empty((0, 3)),
        kept_time=np.asarray([s["kept_time"] for s in samples], dtype=np.int32),
        target=np.stack([s["target"] for s in samples], axis=0) if samples else np.empty((0,)),
        pred=np.stack([s["pred"] for s in samples], axis=0) if samples else np.empty((0,)),
        div_target=np.stack([s["div_target"] for s in samples], axis=0) if samples else np.empty((0,)),
        div_pred=np.stack([s["div_pred"] for s in samples], axis=0) if samples else np.empty((0,)),
    )


def main():
    p = argparse.ArgumentParser(description="Evaluate and visualize a 3D block CNN checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cell-npz", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--split-ratio", default="0.8,0.1,0.1")
    p.add_argument("--split-load", default="")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-samples", type=int, default=0, help="cap evaluated samples; 0 evaluates the full split")
    p.add_argument("--time-min", type=int, default=-1, help="minimum kept_time to evaluate; -1 disables")
    p.add_argument("--time-max", type=int, default=-1, help="maximum kept_time to evaluate; -1 disables")
    p.add_argument("--num-plot-samples", type=int, default=4)
    p.add_argument("--num-worst-plot-samples", type=int, default=8)
    p.add_argument(
        "--worst-by",
        choices=["pred_div_max_abs", "div_error_max_abs", "data_mse"],
        default="pred_div_max_abs",
    )
    p.add_argument("--plot-channel", type=int, default=0)
    p.add_argument("--slice-axis", choices=["x", "y", "z"], default="z")
    p.add_argument("--slice-index", type=int, default=-1, help="-1 uses the center slice")
    p.add_argument("--fields-per-face", type=int, default=3)
    p.add_argument("--phi-in-face", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    worst_plot_dir = out_dir / "worst_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    worst_plot_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    dataset = DenseBlock3DDataset(args.cell_npz)
    with np.load(args.cell_npz, allow_pickle=True) as z:
        dataset.block_starts = np.asarray(z["block_starts"], dtype=np.int32) if "block_starts" in z.files else None
        dataset.kept_times = np.asarray(z["kept_times"], dtype=np.int32) if "kept_times" in z.files else None
    blocks_per_time = None
    if dataset.kept_times is not None and len(dataset.kept_times) > 0 and len(dataset) % len(dataset.kept_times) == 0:
        blocks_per_time = len(dataset) // len(dataset.kept_times)
    idx = split_indices(len(dataset), args)
    idx = filter_indices_by_time(idx, dataset.kept_times, blocks_per_time, args.time_min, args.time_max)
    if len(idx) == 0:
        raise ValueError("selected split/time range has no samples")
    subset = Subset(dataset, idx.tolist())
    loader = DataLoader(subset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    if tuple(dataset.input_dims) != tuple(int(v) for v in ckpt["input_dims"]):
        raise ValueError(f"dataset input_dims={dataset.input_dims} differs from checkpoint {ckpt['input_dims']}")
    if tuple(dataset.core_shape) != tuple(int(v) for v in ckpt["core_shape"]):
        raise ValueError(f"dataset core_shape={dataset.core_shape} differs from checkpoint {ckpt['core_shape']}")
    if args.plot_channel < 0 or args.plot_channel >= dataset.out_channels:
        raise ValueError(f"--plot-channel must be in [0, {dataset.out_channels - 1}], got {args.plot_channel}")

    model = build_model_from_checkpoint(ckpt, device)
    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)

    data_norm_sums = {"sq": 0.0, "abs": 0.0, "n": 0}
    data_phys_sums = {"sq": 0.0, "abs": 0.0, "n": 0}
    abs_div_pred = []
    abs_div_target = []
    abs_div_err = []
    per_sample_rows = []
    samples_to_save = []
    worst_heap = []
    slice_index = None if args.slice_index < 0 else args.slice_index
    region_masks = make_region_masks(dataset.core_shape)
    pred_div_regions = {name: [] for name in region_masks}
    div_error_regions = {name: [] for name in region_masks}

    with torch.no_grad():
        seen = 0
        for batch in loader:
            x, y, volumes, signs = batch
            x = x.to(device)
            y = y.to(device)
            volumes = volumes.to(device)
            signs = signs.to(device)

            pred_norm = center_crop_3d(model((x - x_mean) / x_std), dataset.core_shape)
            y_norm = (y - y_mean) / y_std
            pred = pred_norm * y_std + y_mean
            x_core = center_crop_3d(x, dataset.core_shape)
            div_pred = divergence_field(x_core, pred, signs, volumes, args.fields_per_face, args.phi_in_face)
            div_target = divergence_field(x_core, y, signs, volumes, args.fields_per_face, args.phi_in_face)

            update_basic_sums(data_norm_sums, pred_norm, y_norm)
            update_basic_sums(data_phys_sums, pred, y)
            abs_div_pred.append(div_pred.abs().reshape(-1).detach().cpu().numpy())
            abs_div_target.append(div_target.abs().reshape(-1).detach().cpu().numpy())
            abs_div_err.append((div_pred - div_target).abs().reshape(-1).detach().cpu().numpy())

            pred_np = pred.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            pred_norm_np = pred_norm.detach().cpu().numpy()
            y_norm_np = y_norm.detach().cpu().numpy()
            div_pred_np = div_pred.detach().cpu().numpy()
            div_target_np = div_target.detach().cpu().numpy()
            batch_div_pred_abs = np.abs(div_pred_np)
            batch_div_err_abs = np.abs(div_pred_np - div_target_np)
            for region, mask in region_masks.items():
                pred_div_regions[region].append(batch_div_pred_abs[:, mask].reshape(-1))
                div_error_regions[region].append(batch_div_err_abs[:, mask].reshape(-1))

            for b in range(x.shape[0]):
                sample_index = int(idx[seen + b])
                block_start = (
                    np.asarray(dataset.block_starts[sample_index], dtype=np.int32)
                    if dataset.block_starts is not None
                    else np.asarray([-1, -1, -1], dtype=np.int32)
                )
                if blocks_per_time is not None:
                    kept_time = sample_kept_time(sample_index, dataset.kept_times, blocks_per_time)
                    block_in_time = int(sample_index % blocks_per_time)
                else:
                    kept_time = -1
                    block_in_time = -1

                data_err = pred_norm_np[b] - y_norm_np[b]
                phys_err = pred_np[b] - y_np[b]
                div_err = div_pred_np[b] - div_target_np[b]
                div_pred_abs = np.abs(div_pred_np[b])
                div_err_abs = np.abs(div_err)
                max_pos = np.unravel_index(int(np.argmax(div_pred_abs)), div_pred_abs.shape)
                div_err_max_pos = np.unravel_index(int(np.argmax(div_err_abs)), div_err_abs.shape)
                row = {
                    "rank_in_split": int(seen + b),
                    "sample_index": sample_index,
                    "kept_time": kept_time,
                    "block_in_time": block_in_time,
                    "block_start_x": int(block_start[0]),
                    "block_start_y": int(block_start[1]),
                    "block_start_z": int(block_start[2]),
                    "data_mse": float(np.mean(data_err * data_err)),
                    "data_mae": float(np.mean(np.abs(data_err))),
                    "phys_mse": float(np.mean(phys_err * phys_err)),
                    "phys_mae": float(np.mean(np.abs(phys_err))),
                    "pred_div_mean_abs": float(np.mean(div_pred_abs)),
                    "pred_div_max_abs": float(np.max(div_pred_abs)),
                    "pred_div_p95_abs": float(np.percentile(div_pred_abs, 95)),
                    "pred_div_p99_abs": float(np.percentile(div_pred_abs, 99)),
                    "pred_div_mean_square": float(np.mean(div_pred_np[b] * div_pred_np[b])),
                    "target_div_mean_abs": float(np.mean(np.abs(div_target_np[b]))),
                    "target_div_max_abs": float(np.max(np.abs(div_target_np[b]))),
                    "target_div_mean_square": float(np.mean(div_target_np[b] * div_target_np[b])),
                    "div_error_mean_abs": float(np.mean(div_err_abs)),
                    "div_error_max_abs": float(np.max(div_err_abs)),
                    "div_error_p95_abs": float(np.percentile(div_err_abs, 95)),
                    "div_error_p99_abs": float(np.percentile(div_err_abs, 99)),
                    "div_error_mean_square": float(np.mean(div_err * div_err)),
                    "pred_div_max_i": int(max_pos[0]),
                    "pred_div_max_j": int(max_pos[1]),
                    "pred_div_max_k": int(max_pos[2]),
                    "div_error_max_i": int(div_err_max_pos[0]),
                    "div_error_max_j": int(div_err_max_pos[1]),
                    "div_error_max_k": int(div_err_max_pos[2]),
                }
                per_sample_rows.append(row)

                if args.num_worst_plot_samples > 0:
                    payload = make_sample_payload(
                        sample_index,
                        block_start,
                        kept_time,
                        y_np[b],
                        pred_np[b],
                        div_target_np[b],
                        div_pred_np[b],
                    )
                    score = float(row[args.worst_by])
                    entry = (score, sample_index, row, payload)
                    if len(worst_heap) < args.num_worst_plot_samples:
                        heapq.heappush(worst_heap, entry)
                    elif score > worst_heap[0][0]:
                        heapq.heapreplace(worst_heap, entry)

            n_plot_remaining = args.num_plot_samples - len(samples_to_save)
            if n_plot_remaining > 0:
                take = min(n_plot_remaining, x.shape[0])
                for b in range(take):
                    sample_index = int(idx[seen + b])
                    block_start = (
                        np.asarray(dataset.block_starts[sample_index], dtype=np.int32)
                        if dataset.block_starts is not None
                        else np.asarray([-1, -1, -1], dtype=np.int32)
                    )
                    kept_time = (
                        sample_kept_time(sample_index, dataset.kept_times, blocks_per_time)
                        if blocks_per_time is not None
                        else -1
                    )
                    samples_to_save.append(
                        make_sample_payload(
                            sample_index,
                            block_start,
                            kept_time,
                            y_np[b],
                            pred_np[b],
                            div_target_np[b],
                            div_pred_np[b],
                        )
                    )
            seen += x.shape[0]
            print(f"[INFO] evaluated {seen}/{len(subset)} samples", flush=True)

    mse = data_norm_sums["sq"] / max(1, data_norm_sums["n"])
    phys_mse = data_phys_sums["sq"] / max(1, data_phys_sums["n"])
    metrics = {
        "checkpoint": os.path.basename(args.checkpoint),
        "cell_npz": args.cell_npz,
        "eval_split": args.eval_split,
        "time_min": args.time_min,
        "time_max": args.time_max,
        "num_samples": int(len(idx)),
        "data_mse": mse,
        "data_rmse": float(np.sqrt(mse)),
        "data_mae": data_norm_sums["abs"] / max(1, data_norm_sums["n"]),
        "phys_mse": phys_mse,
        "phys_rmse": float(np.sqrt(phys_mse)),
        "phys_mae": data_phys_sums["abs"] / max(1, data_phys_sums["n"]),
    }
    metrics.update(summarize_abs("pred_div", abs_div_pred))
    metrics.update(summarize_abs("target_div", abs_div_target))
    metrics.update(summarize_abs("div_error", abs_div_err))
    region_rows = summarize_region_values(pred_div_regions, "pred_div")
    region_rows.extend(summarize_region_values(div_error_regions, "div_error"))
    split_time_count_rows = summarize_split_time_counts(len(dataset), dataset.kept_times, blocks_per_time, args)

    write_metrics(out_dir / "metrics.csv", metrics)
    per_sample_rows_sorted = sorted(per_sample_rows, key=lambda r: r[args.worst_by], reverse=True)
    time_rows = summarize_group_rows(
        per_sample_rows,
        "kept_time",
        [
            "data_mse",
            "pred_div_mean_abs",
            "pred_div_max_abs",
            "pred_div_p99_abs",
            "pred_div_mean_square",
            "div_error_mean_abs",
            "div_error_max_abs",
            "div_error_p99_abs",
            "div_error_mean_square",
        ],
    )
    write_rows(out_dir / "per_sample_metrics.csv", per_sample_rows_sorted)
    write_rows(out_dir / "time_metrics.csv", time_rows)
    write_rows(out_dir / "region_metrics.csv", region_rows)
    write_rows(out_dir / "split_time_counts.csv", split_time_count_rows)
    save_sample_npz(out_dir / "sample_predictions.npz", samples_to_save)
    worst_entries = sorted(worst_heap, key=lambda item: item[0], reverse=True)
    worst_samples = [entry[3] for entry in worst_entries]
    save_sample_npz(out_dir / "worst_predictions.npz", worst_samples)

    for s in samples_to_save:
        out_png = plot_dir / f"sample_{s['sample_index']:06d}_ch{args.plot_channel}_{args.slice_axis}.png"
        plot_sample(s, out_png, args.plot_channel, args.slice_axis, slice_index)
    for rank, s in enumerate(worst_samples, start=1):
        out_png = worst_plot_dir / f"rank_{rank:02d}_sample_{s['sample_index']:06d}_{args.worst_by}_ch{args.plot_channel}_{args.slice_axis}.png"
        plot_sample(s, out_png, args.plot_channel, args.slice_axis, slice_index)

    print("[INFO] metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"[INFO] wrote {out_dir / 'metrics.csv'}")
    print(f"[INFO] wrote {out_dir / 'per_sample_metrics.csv'}")
    print(f"[INFO] wrote {out_dir / 'time_metrics.csv'}")
    print(f"[INFO] wrote {out_dir / 'region_metrics.csv'}")
    print(f"[INFO] wrote {out_dir / 'split_time_counts.csv'}")
    print(f"[INFO] wrote {out_dir / 'sample_predictions.npz'}")
    print(f"[INFO] wrote {out_dir / 'worst_predictions.npz'}")
    print(f"[INFO] wrote {len(samples_to_save)} plots to {plot_dir}")
    print(f"[INFO] wrote {len(worst_samples)} worst plots to {worst_plot_dir}")


if __name__ == "__main__":
    main()
