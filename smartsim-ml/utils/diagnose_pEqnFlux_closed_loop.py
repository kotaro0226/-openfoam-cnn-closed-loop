#!/usr/bin/env python3
"""Diagnose pEqnFlux_native/CNN/used fields saved from closed-loop solver runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from openfoam_field_utils import (
    face_axis,
    infer_cell_ijk,
    numeric_time_dirs,
    read_label_list,
    read_scalar_internal,
    read_vector_internal,
    time_name,
)


FACE_NAMES = ("native", "cnn", "used")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, help="OpenFOAM case with pEqnFlux_native and pEqnFlux_cnn fields")
    p.add_argument("--out-dir", default="outputs/pEqnFlux_diagnostics")
    p.add_argument("--alpha", type=float, default=None, help="alpha used to reconstruct pEqnFlux_used if absent")
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--stop", type=float, default=None)
    p.add_argument("--top-k", type=int, default=100)
    return p.parse_args()


def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean_abs": float("nan"),
            f"{prefix}_rms": float("nan"),
            f"{prefix}_max_abs": float("nan"),
        }
    return {
        f"{prefix}_n": int(finite.size),
        f"{prefix}_mean_abs": float(np.mean(np.abs(finite))),
        f"{prefix}_rms": float(np.sqrt(np.mean(finite * finite))),
        f"{prefix}_max_abs": float(np.max(np.abs(finite))),
    }


def min_distance(value: float, planes: list[float]) -> float:
    return min(abs(value - p) for p in planes) if planes else float("inf")


def main() -> None:
    args = parse_args()
    case = Path(args.case).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    owner_all = read_label_list(case / "constant/polyMesh/owner")
    neighbour = read_label_list(case / "constant/polyMesh/neighbour")
    n_internal = neighbour.size
    owner = owner_all[:n_internal]
    centres = read_vector_internal(case / "0" / "C")
    cell_ijk, dims = infer_cell_ijk(centres)

    # Mirrors the current 64^3/valid 62^3/blockCore=16/halo=2 solver tiling.
    core_min = 3
    core_max = dims[0] - 4
    block_planes = [18.5, 34.5, 44.5, 50.5] if dims == (64, 64, 64) else []

    summary_rows: list[dict] = []
    axis_rows: list[dict] = []
    top_rows: list[dict] = []

    for t, d in numeric_time_dirs(case):
        if args.start is not None and t < args.start:
            continue
        if args.stop is not None and t > args.stop:
            continue
        native_path = d / "pEqnFlux_native"
        cnn_path = d / "pEqnFlux_cnn"
        if not native_path.exists() or not cnn_path.exists():
            continue

        native = read_scalar_internal(native_path)
        cnn = read_scalar_internal(cnn_path)
        if native.shape != cnn.shape:
            raise RuntimeError(f"{d}: native/cnn shape mismatch {native.shape} != {cnn.shape}")

        mask_path = d / "pEqnFlux_cnn_mask"
        mask = read_scalar_internal(mask_path) > 0.5 if mask_path.exists() else np.ones_like(native, dtype=bool)
        mask = mask[:n_internal]

        used_path = d / "pEqnFlux_used"
        if used_path.exists():
            used = read_scalar_internal(used_path)
        elif args.alpha is not None:
            used = native.copy()
            masked_idx = np.flatnonzero(mask)
            used[masked_idx] = (1.0 - args.alpha) * native[masked_idx] + args.alpha * cnn[masked_idx]
        else:
            used = np.full_like(native, np.nan)

        active = np.zeros_like(mask, dtype=bool)
        active[: min(mask.size, n_internal)] = mask[: min(mask.size, n_internal)]
        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            continue

        row = {"time": time_name(t), "masked_internal_faces": int(active_idx.size)}
        row.update(stats(native[:n_internal][active], "native"))
        row.update(stats(cnn[:n_internal][active], "cnn"))
        row.update(stats(used[:n_internal][active], "used"))
        diff = cnn[:n_internal][active] - native[:n_internal][active]
        row.update(stats(diff, "cnn_minus_native"))
        denom = float(np.linalg.norm(native[:n_internal][active]))
        row["cnn_native_rel_l2"] = float(np.linalg.norm(diff) / denom) if denom > 0 else float("nan")
        summary_rows.append(row)

        axes = np.asarray([face_axis(cell_ijk[owner[i]], cell_ijk[neighbour[i]]) for i in active_idx])
        for axis in ("x", "y", "z", "unknown"):
            idx = active_idx[axes == axis]
            if idx.size == 0:
                continue
            arow = {"time": time_name(t), "axis": axis, "faces": int(idx.size)}
            arow.update(stats(native[idx], "native"))
            arow.update(stats(cnn[idx], "cnn"))
            arow.update(stats(used[idx], "used"))
            arow.update(stats(cnn[idx] - native[idx], "cnn_minus_native"))
            axis_rows.append(arow)

        abs_diff = np.abs(cnn[active_idx] - native[active_idx])
        order = np.argsort(abs_diff)[-args.top_k :][::-1]
        for rank, local in enumerate(order, start=1):
            face_i = int(active_idx[local])
            oi = cell_ijk[owner[face_i]]
            ni = cell_ijk[neighbour[face_i]]
            mid = 0.5 * (oi + ni)
            abs_native = abs(float(native[face_i]))
            top_rows.append(
                {
                    "time": time_name(t),
                    "rank_by_abs_cnn_minus_native": rank,
                    "face": face_i,
                    "owner_cell": int(owner[face_i]),
                    "neighbour_cell": int(neighbour[face_i]),
                    "axis": face_axis(oi, ni),
                    "owner_i": int(oi[0]),
                    "owner_j": int(oi[1]),
                    "owner_k": int(oi[2]),
                    "neigh_i": int(ni[0]),
                    "neigh_j": int(ni[1]),
                    "neigh_k": int(ni[2]),
                    "mid_i": float(mid[0]),
                    "mid_j": float(mid[1]),
                    "mid_k": float(mid[2]),
                    "pEqnFlux_native": float(native[face_i]),
                    "pEqnFlux_cnn": float(cnn[face_i]),
                    "pEqnFlux_used": float(used[face_i]),
                    "cnn_minus_native": float(cnn[face_i] - native[face_i]),
                    "abs_cnn_minus_native": float(abs(cnn[face_i] - native[face_i])),
                    "relative_difference": float(abs(cnn[face_i] - native[face_i]) / max(abs_native, 1e-300)),
                    "near_mask_edge": bool(
                        np.any((oi == core_min) | (oi == core_max) | (ni == core_min) | (ni == core_max))
                    ),
                    "near_block_plane": bool(min(min_distance(float(v), block_planes) for v in mid) < 0.51),
                }
            )

    if not summary_rows:
        raise SystemExit("no pEqnFlux_native/pEqnFlux_cnn time directories found")

    for name, rows in (
        ("pEqnFlux_summary.csv", summary_rows),
        ("pEqnFlux_axis_summary.csv", axis_rows),
        ("pEqnFlux_top_abs_cnn_minus_native_faces.csv", top_rows),
    ):
        path = out_dir / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] wrote {path}")


if __name__ == "__main__":
    main()
