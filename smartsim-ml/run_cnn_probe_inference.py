#!/usr/bin/env python3
"""Run one-timestep TorchScript CNN inference from a C++ 18-channel CSV probe."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="cnn_input_18ch.csv written by writeCnnInput.H")
    p.add_argument("--cell-centres", required=True, help="OpenFOAM 0/C file for the same case")
    p.add_argument("--model", required=True, help="TorchScript model exported by export_cnn_torchscript.py")
    p.add_argument("--out", required=True, help="output NPZ for one-timestep block predictions")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--halo", type=int, default=2)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return p.parse_args()


def read_cell_centres(path: str) -> np.ndarray:
    text = Path(path).read_text(errors="ignore")
    m = re.search(r"nonuniform\s+List<vector>\s*(\d+)\s*\(\s*", text)
    if not m:
        raise RuntimeError(f"Could not parse nonuniform List<vector> from {path}")
    n = int(m.group(1))
    vals = [float(x) for x in _NUM_RE.findall(text[m.end():])]
    if len(vals) < 3 * n:
        raise RuntimeError(f"Parsed {len(vals)//3} vectors, expected {n}")
    return np.asarray(vals[: 3 * n], dtype=np.float64).reshape(n, 3)


def infer_ijk(C: np.ndarray, tol_decimals: int = 12) -> tuple[np.ndarray, tuple[int, int, int]]:
    x = np.round(C[:, 0], tol_decimals)
    y = np.round(C[:, 1], tol_decimals)
    z = np.round(C[:, 2], tol_decimals)
    xu = np.unique(x); xu.sort()
    yu = np.unique(y); yu.sort()
    zu = np.unique(z); zu.sort()
    expected_n = round(C.shape[0] ** (1.0 / 3.0))
    if expected_n ** 3 == C.shape[0] and (len(xu), len(yu), len(zu)) != (expected_n, expected_n, expected_n):
        mins = C.min(axis=0)
        maxs = C.max(axis=0)
        spacing = (maxs - mins) / float(expected_n - 1)
        ijk = np.rint((C - mins) / spacing).astype(np.int32)
        if ijk.min() < 0 or ijk.max() >= expected_n:
            raise RuntimeError("uniform-grid ijk inference produced out-of-range indices")
        linear = ijk[:, 0] * expected_n * expected_n + ijk[:, 1] * expected_n + ijk[:, 2]
        if np.unique(linear).size != C.shape[0]:
            raise RuntimeError("uniform-grid ijk inference did not produce one cell per grid slot")
        return ijk, (expected_n, expected_n, expected_n)

    xmap = {v: i for i, v in enumerate(xu)}
    ymap = {v: i for i, v in enumerate(yu)}
    zmap = {v: i for i, v in enumerate(zu)}
    ijk = np.stack(
        [
            np.fromiter((xmap[v] for v in x), dtype=np.int32, count=len(x)),
            np.fromiter((ymap[v] for v in y), dtype=np.int32, count=len(y)),
            np.fromiter((zmap[v] for v in z), dtype=np.int32, count=len(z)),
        ],
        axis=1,
    )
    return ijk, (len(xu), len(yu), len(zu))


def read_probe_csv(path: str, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((n_cells, 18), dtype=np.float32)
    valid = np.zeros((n_cells,), dtype=bool)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != 20:
            raise RuntimeError(f"Expected 20 columns, got {len(header)}")
        for row in reader:
            cell_id = int(row[0])
            valid[cell_id] = bool(int(row[1]))
            X[cell_id] = np.asarray(row[2:], dtype=np.float32)
    return X, valid


def compute_block_starts(valid_size: int, block_size: int, halo: int, stride: int) -> list[int]:
    min_start = halo
    max_start = valid_size - halo - block_size
    starts = list(range(min_start, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def choose_device(spec: str) -> torch.device:
    if spec == "cuda":
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    C = read_cell_centres(args.cell_centres)
    cell_ijk, dims = infer_ijk(C)
    X_cell, valid = read_probe_csv(args.csv, C.shape[0])

    X_grid = np.zeros((*dims, 18), dtype=np.float32)
    valid_grid = np.zeros(dims, dtype=bool)
    X_grid[cell_ijk[:, 0], cell_ijk[:, 1], cell_ijk[:, 2], :] = X_cell
    valid_grid[cell_ijk[:, 0], cell_ijk[:, 1], cell_ijk[:, 2]] = valid

    X_valid = X_grid[1:-1, 1:-1, 1:-1, :]
    valid_interior = valid_grid[1:-1, 1:-1, 1:-1]
    if not valid_interior.all():
        raise RuntimeError(f"Interior valid mask has {(~valid_interior).sum()} invalid cells")

    valid_dims = X_valid.shape[:3]
    starts_i = compute_block_starts(valid_dims[0], args.block_size, args.halo, args.stride)
    starts_j = compute_block_starts(valid_dims[1], args.block_size, args.halo, args.stride)
    starts_k = compute_block_starts(valid_dims[2], args.block_size, args.halo, args.stride)
    block_starts = np.asarray([(i, j, k) for i in starts_i for j in starts_j for k in starts_k], dtype=np.int32)

    device = choose_device(args.device)
    model = torch.jit.load(args.model, map_location=device).eval()

    input_size = args.block_size + 2 * args.halo
    pred = np.empty((len(block_starts), 6, args.block_size, args.block_size, args.block_size), dtype=np.float32)
    with torch.no_grad():
        for out_i, (i0, j0, k0) in enumerate(block_starts):
            x_block = X_valid[
                i0 - args.halo : i0 - args.halo + input_size,
                j0 - args.halo : j0 - args.halo + input_size,
                k0 - args.halo : k0 - args.halo + input_size,
                :,
            ]
            x = torch.from_numpy(np.transpose(x_block, (3, 0, 1, 2))[None]).to(device)
            y = model(x).cpu().numpy()[0]
            pred[out_i] = y

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pred=pred,
        block_starts=block_starts,
        valid_dims=np.asarray(valid_dims, dtype=np.int32),
        input_shape=np.asarray([input_size, input_size, input_size], dtype=np.int32),
        core_shape=np.asarray([args.block_size, args.block_size, args.block_size], dtype=np.int32),
        source_csv=np.asarray(args.csv),
        model=np.asarray(args.model),
    )
    print(f"[INFO] wrote {out_path} pred={pred.shape} blocks={len(block_starts)} device={device}")


if __name__ == "__main__":
    main()
