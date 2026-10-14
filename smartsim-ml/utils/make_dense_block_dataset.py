#!/usr/bin/env python3
"""Build dense 3D block datasets from existing CFD face/cell maps.

This converts the existing face-wise raw dataset plus cell connectivity into
contiguous dense cell blocks suitable for 3D CNN training without sparse masks.

Output NPZ keys:
  - X: (B, Cin, Din, Hin, Win) float32
  - Y: (B, Cout, Dcore, Hcore, Wcore) float32
  - V: (B, Dcore, Hcore, Wcore) float32
  - cell_face_sign: (B, 6, Dcore, Hcore, Wcore) float32
  - block_starts: (B, 3) int32      core block starts in the valid 62^3 domain
  - input_shape: (3,) int32         (Din, Hin, Win)
  - core_shape: (3,) int32          (Dcore, Hcore, Wcore)
  - halo: () int32
  - valid_dims: (3,) int32          dense valid domain dims after removing
                                     physical boundaries from the 64^3 case
  - kept_times: copied from raw npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-npz", required=True, help="face-wise raw dataset, e.g. of-raw/et50_dt0p01.npz")
    p.add_argument("--cell-maps", required=True, help="cell_maps_rank0_internal.npz")
    p.add_argument("--out", required=True, help="output dense block dataset npz")
    p.add_argument("--block-size", type=int, default=16, help="core block edge length")
    p.add_argument("--halo", type=int, default=2, help="halo width around the core block")
    p.add_argument("--stride", type=int, default=16, help="core-block stride in valid cell space")
    return p.parse_args()


def compute_block_starts(valid_size: int, block_size: int, halo: int, stride: int) -> list[int]:
    min_start = halo
    max_start = valid_size - halo - block_size
    if max_start < min_start:
        raise ValueError(
            f"block_size={block_size} halo={halo} do not fit inside valid_size={valid_size}"
        )

    starts = list(range(min_start, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def main():
    args = parse_args()

    raw = np.load(args.raw_npz, allow_pickle=True)
    maps = np.load(args.cell_maps, allow_pickle=True)

    X_face = np.asarray(raw["X"], dtype=np.float32)  # (T, Nf, 3)
    Y_face = np.asarray(raw["Y"], dtype=np.float32)  # (T, Nf, 1)
    cell_faces = np.asarray(maps["cell_faces_idx"], dtype=np.int64)  # (Nc, 6)
    cell_sign = np.asarray(maps["cell_face_sign"], dtype=np.float32)  # (Nc, 6)
    cell_ijk = np.asarray(maps["cell_ijk"], dtype=np.int32)           # (Nc, 3), 1..62
    V = np.asarray(maps["V"], dtype=np.float32)                       # (Nc,)

    dims_cells = tuple(int(v) for v in np.asarray(maps["dims_cells"], dtype=np.int32))
    valid_dims = tuple(d - 2 for d in dims_cells)
    if valid_dims[0] <= 0 or valid_dims[1] <= 0 or valid_dims[2] <= 0:
        raise ValueError(f"Unexpected dims_cells={dims_cells}")

    # Shift structured valid domain from 1..62 -> 0..61.
    cell_ijk_valid = cell_ijk - 1
    if (cell_ijk_valid.min(axis=0) != 0).any() or (cell_ijk_valid.max(axis=0) != np.asarray(valid_dims) - 1).any():
        raise ValueError("cell_ijk does not span the expected dense valid domain")

    n_valid = int(np.prod(valid_dims))
    order = np.lexsort((cell_ijk_valid[:, 2], cell_ijk_valid[:, 1], cell_ijk_valid[:, 0]))
    cell_ijk_sorted = cell_ijk_valid[order]
    cell_faces_sorted = cell_faces[order]
    cell_sign_sorted = cell_sign[order]
    V_sorted = V[order]

    dense_linear = (
        cell_ijk_sorted[:, 0] * (valid_dims[1] * valid_dims[2])
        + cell_ijk_sorted[:, 1] * valid_dims[2]
        + cell_ijk_sorted[:, 2]
    )
    expected = np.arange(n_valid, dtype=np.int64)
    if not np.array_equal(dense_linear, expected):
        raise ValueError("cell_ijk cannot be reshaped into a dense valid domain")

    faces_grid = cell_faces_sorted.reshape(*valid_dims, 6)
    sign_grid = cell_sign_sorted.reshape(*valid_dims, 6)
    V_grid = V_sorted.reshape(*valid_dims)

    core = args.block_size
    halo = args.halo
    stride = args.stride
    input_size = core + 2 * halo

    starts_i = compute_block_starts(valid_dims[0], core, halo, stride)
    starts_j = compute_block_starts(valid_dims[1], core, halo, stride)
    starts_k = compute_block_starts(valid_dims[2], core, halo, stride)
    block_starts = np.asarray(
        [(i, j, k) for i in starts_i for j in starts_j for k in starts_k],
        dtype=np.int32,
    )

    T = X_face.shape[0]
    B_per_t = block_starts.shape[0]
    B_total = T * B_per_t
    Cin = 18
    Cout = 6

    X_blocks = np.empty((B_total, Cin, input_size, input_size, input_size), dtype=np.float32)
    Y_blocks = np.empty((B_total, Cout, core, core, core), dtype=np.float32)
    V_blocks = np.empty((B_total, core, core, core), dtype=np.float32)
    S_blocks = np.empty((B_total, 6, core, core, core), dtype=np.float32)
    starts_all = np.empty((B_total, 3), dtype=np.int32)

    out_idx = 0
    for t in range(T):
        x_cell_t = X_face[t, faces_grid, :].reshape(*valid_dims, 18)
        y_cell_t = Y_face[t, faces_grid, 0].reshape(*valid_dims, 6)

        for start in block_starts:
            i0, j0, k0 = (int(v) for v in start)
            ii0, jj0, kk0 = i0 - halo, j0 - halo, k0 - halo
            ii1, jj1, kk1 = ii0 + input_size, jj0 + input_size, kk0 + input_size
            ci1, cj1, ck1 = i0 + core, j0 + core, k0 + core

            x_block = x_cell_t[ii0:ii1, jj0:jj1, kk0:kk1, :]
            y_block = y_cell_t[i0:ci1, j0:cj1, k0:ck1, :]
            v_block = V_grid[i0:ci1, j0:cj1, k0:ck1]
            s_block = sign_grid[i0:ci1, j0:cj1, k0:ck1, :]

            if x_block.shape != (input_size, input_size, input_size, Cin):
                raise ValueError(f"Bad X block shape: {x_block.shape}")
            if y_block.shape != (core, core, core, Cout):
                raise ValueError(f"Bad Y block shape: {y_block.shape}")

            X_blocks[out_idx] = np.transpose(x_block, (3, 0, 1, 2))
            Y_blocks[out_idx] = np.transpose(y_block, (3, 0, 1, 2))
            V_blocks[out_idx] = v_block
            S_blocks[out_idx] = np.transpose(s_block, (3, 0, 1, 2))
            starts_all[out_idx] = start
            out_idx += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X_blocks,
        Y=Y_blocks,
        V=V_blocks,
        cell_face_sign=S_blocks,
        block_starts=starts_all,
        input_shape=np.asarray([input_size, input_size, input_size], dtype=np.int32),
        core_shape=np.asarray([core, core, core], dtype=np.int32),
        halo=np.asarray(halo, dtype=np.int32),
        valid_dims=np.asarray(valid_dims, dtype=np.int32),
        kept_times=raw["kept_times"] if "kept_times" in raw.files else np.arange(T, dtype=np.int32),
    )
    print(
        f"[INFO] wrote {out_path} with {B_total} blocks "
        f"({T} timesteps x {B_per_t} blocks/timestep), input={input_size}^3 core={core}^3"
    )


if __name__ == "__main__":
    main()
