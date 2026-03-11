#!/usr/bin/env python3
"""Add continuity metadata keys to a cell training NPZ.

This utility supports two input styles for --cell-npz:

1) Cell-wise sampled NPZ (already from SICKLE-like cell sampling):
   - X: (TB, Ns, Cin)
   - Y: (TB, Ns, Cout)
   - indices_list: sampled cell ids per block
   => Adds only V_sub_list and cell_face_sign_sub_list.

2) Face-wise OF raw NPZ (from make_cfd_dataset/assemble):
   - X: (T, Nf, Fx)
   - Y: (T, Nf, Fy)
   - no indices_list
   + cell_maps_*.npz with cell_faces_idx(Nc,6), V(Nc), cell_face_sign(Nc,6)
   => Builds full cell-wise data (TB=T, Ns=Nc):
      X_cell (T,Nc,6*Fx), Y_cell (T,Nc,6*Fy), indices_list, V_sub_list, cell_face_sign_sub_list.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np

REQUIRED_MAP_KEYS = ("V", "cell_face_sign")
TARGET_KEYS = ("V_sub_list", "cell_face_sign_sub_list")


def _load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _extract_indices(indices_obj: np.ndarray, expected_tb: int, expected_ns: int) -> np.ndarray:
    if indices_obj.ndim == 2 and indices_obj.dtype != object:
        idx = np.asarray(indices_obj, dtype=np.int64)
    elif indices_obj.ndim == 1 and indices_obj.dtype == object:
        rows = [np.asarray(r, dtype=np.int64).reshape(-1) for r in indices_obj.tolist()]
        idx = np.stack(rows, axis=0)
    elif indices_obj.ndim == 2 and indices_obj.dtype == object:
        idx = np.asarray(indices_obj, dtype=np.int64)
    else:
        raise ValueError(
            "indices_list must be either 2D numeric (TB,Ns) or 1D object array of per-block 1D indices. "
            f"got shape={indices_obj.shape}, dtype={indices_obj.dtype}"
        )

    if idx.shape != (expected_tb, expected_ns):
        raise ValueError(
            f"indices_list shape mismatch: expected {(expected_tb, expected_ns)} from X/Y, got {idx.shape}"
        )
    return idx


def _build_meta(indices: np.ndarray, V_all: np.ndarray, sign_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if indices.min() < 0 or indices.max() >= V_all.shape[0]:
        raise IndexError(
            f"indices_list out of range for V/cell_face_sign: min={indices.min()} max={indices.max()} "
            f"valid=[0,{V_all.shape[0]-1}]"
        )
    v_sub = V_all[indices]
    s_sub = sign_all[indices, :]
    return v_sub.astype(np.float32, copy=False), s_sub.astype(np.int8, copy=False)


def _tile_meta_full_cells(num_t: int, V_all: np.ndarray, sign_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nc = V_all.shape[0]
    idx = np.broadcast_to(np.arange(nc, dtype=np.int64), (num_t, nc)).copy()
    v_sub = np.broadcast_to(V_all[None, :], (num_t, nc)).copy().astype(np.float32, copy=False)
    s_sub = np.broadcast_to(sign_all[None, :, :], (num_t, nc, 6)).copy().astype(np.int8, copy=False)
    return idx, v_sub, s_sub


def _convert_face_raw_to_cell(Xf: np.ndarray, Yf: np.ndarray, cell_faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Xf: (T,Nf,Fx), Yf: (T,Nf,Fy), cell_faces: (Nc,6)
    if Xf.ndim != 3 or Yf.ndim != 3:
        raise ValueError(f"X/Y must be 3D. got X={Xf.shape}, Y={Yf.shape}")

    t, nf, fx = Xf.shape
    t2, nf2, fy = Yf.shape
    if (t2, nf2) != (t, nf):
        raise ValueError(f"X/Y first dims mismatch for face raw: X={Xf.shape}, Y={Yf.shape}")

    if cell_faces.ndim != 2 or cell_faces.shape[1] != 6:
        raise ValueError(f"cell_faces_idx must be (Nc,6). got {cell_faces.shape}")

    if cell_faces.min() < 0 or cell_faces.max() >= nf:
        raise IndexError(
            f"cell_faces_idx out of range for face axis Nf={nf}: min={cell_faces.min()} max={cell_faces.max()}"
        )

    x6 = Xf[:, cell_faces, :]  # (T,Nc,6,Fx)
    y6 = Yf[:, cell_faces, :]  # (T,Nc,6,Fy)
    xc = x6.reshape(t, cell_faces.shape[0], 6 * fx)
    yc = y6.reshape(t, cell_faces.shape[0], 6 * fy)
    return xc.astype(np.float32, copy=False), yc.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrofit/add continuity keys for cell training NPZ")
    p.add_argument("--cell-npz", required=True, help="Input NPZ (cell-wise sampled OR face-wise raw)")
    p.add_argument("--cell-maps", required=True, help="cell_maps_*.npz containing V/cell_face_sign (+cell_faces_idx for raw)")
    p.add_argument("--out", default="", help="Output NPZ path (or use --inplace)")
    p.add_argument("--inplace", action="store_true", help="Overwrite --cell-npz")
    p.add_argument("--force", action="store_true", help="Overwrite existing target keys if present")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.inplace and args.out:
        raise ValueError("Use either --inplace or --out, not both")
    if not args.inplace and not args.out:
        raise ValueError("Specify --out (or use --inplace)")

    out_path = args.cell_npz if args.inplace else args.out

    cell = _load_npz(args.cell_npz)
    maps = _load_npz(args.cell_maps)

    for k in REQUIRED_MAP_KEYS:
        if k not in maps.files:
            raise KeyError(f"{args.cell_maps} missing required key '{k}'. keys={maps.files}")

    V_all = np.asarray(maps["V"], dtype=np.float32).reshape(-1)
    sign_all = np.asarray(maps["cell_face_sign"], dtype=np.int8)
    if sign_all.ndim != 2 or sign_all.shape[1] != 6:
        raise ValueError(f"cell_face_sign must be (Nc,6). got {sign_all.shape}")
    if sign_all.shape[0] != V_all.shape[0]:
        raise ValueError(f"V/cell_face_sign length mismatch. V={V_all.shape}, sign={sign_all.shape}")

    if "X" not in cell.files or "Y" not in cell.files:
        raise KeyError(f"{args.cell_npz} must include X and Y. keys={cell.files}")

    out = {k: cell[k] for k in cell.files}
    if (not args.force) and all(k in out for k in TARGET_KEYS):
        print("[INFO] target keys already exist. nothing to do (use --force to rebuild).")
        return

    X = np.asarray(cell["X"])
    Y = np.asarray(cell["Y"])

    # Mode A: existing cell-wise sampled npz with indices_list
    if "indices_list" in cell.files:
        TB, Ns = X.shape[0], X.shape[1]
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(f"X/Y must be 3D. got X={X.shape}, Y={Y.shape}")
        if (Y.shape[0], Y.shape[1]) != (TB, Ns):
            raise ValueError(f"X/Y first 2 dims mismatch. X={X.shape}, Y={Y.shape}")

        idx = _extract_indices(np.asarray(cell["indices_list"], dtype=object), TB, Ns)
        V_sub, S_sub = _build_meta(idx, V_all, sign_all)

        out["V_sub_list"] = V_sub
        out["cell_face_sign_sub_list"] = S_sub
        mode = "cell-wise sampled"

    # Mode B: face-wise raw npz -> build full cell-wise tensor and metadata
    else:
        if "cell_faces_idx" not in maps.files:
            raise KeyError(
                f"{args.cell_maps} missing 'cell_faces_idx'. This key is required when --cell-npz is face-wise raw. keys={maps.files}"
            )
        cell_faces = np.asarray(maps["cell_faces_idx"], dtype=np.int64)
        X_cell, Y_cell = _convert_face_raw_to_cell(X, Y, cell_faces)

        idx, V_sub, S_sub = _tile_meta_full_cells(X_cell.shape[0], V_all, sign_all)

        out["X"] = X_cell
        out["Y"] = Y_cell
        out["indices_list"] = idx
        out["V_sub_list"] = V_sub
        out["cell_face_sign_sub_list"] = S_sub
        out["num_hypercubes"] = np.int32(1)
        out["num_samples"] = np.int32(X_cell.shape[1])
        out["T_effective"] = np.int32(X_cell.shape[0])
        if "dims_cells" in maps.files:
            out["dims_cells"] = maps["dims_cells"]
        mode = "face-wise raw -> full cell-wise"

    outdir = os.path.dirname(out_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    np.savez(out_path, **out)

    print(f"[INFO] mode={mode}")
    print(f"[INFO] wrote: {out_path}")
    print(f"[INFO] X={np.asarray(out['X']).shape} Y={np.asarray(out['Y']).shape}")
    print(
        f"[INFO] indices={np.asarray(out['indices_list']).shape} "
        f"V_sub_list={np.asarray(out['V_sub_list']).shape} "
        f"cell_face_sign_sub_list={np.asarray(out['cell_face_sign_sub_list']).shape}"
    )


if __name__ == "__main__":
    main()

