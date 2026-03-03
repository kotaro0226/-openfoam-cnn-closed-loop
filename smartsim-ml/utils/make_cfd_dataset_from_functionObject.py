#!/usr/bin/env python3
"""
Build per-rank cfd_xy_rank####.npz files (same schema as make_cfd_dataset.py)
from OpenFOAM functionObject outputs written under processor*/<time>/.

Expected use-case:
- mlTeacherFieldWriter (or similar) wrote face scalar fields per rank/time.
- Each field file is a surfaceScalarField with nonuniform internalField list.

Output NPZ keys are aligned with make_cfd_dataset.py:
- X: (T_keep, Nf_local, Fx)
- Y: (T_keep, Nf_local, Fy=1)
- kept_times: (T_keep,) int32
- plus metadata keys: rank, nproc, x_fields, y_field, start, end, stride, X_shape, Y_shape, ...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    return a.reshape(a.shape[0], -1)


def _nan_to_num_inplace(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _iter_ranks(case: Path, nproc: int, processor_prefix: str) -> Iterable[int]:
    for r in range(nproc):
        proc_dir = case / f"{processor_prefix}{r}"
        if not proc_dir.exists():
            raise FileNotFoundError(f"Missing processor directory: {proc_dir}")
        yield r


def _list_time_dirs(proc_dir: Path) -> list[Path]:
    out: list[tuple[float, Path]] = []
    for p in proc_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            tv = float(p.name)
        except ValueError:
            continue
        out.append((tv, p))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def _parse_surface_scalar_internal(path: Path, dtype: np.dtype) -> np.ndarray:
    txt = path.read_text(errors="replace")

    if "internalField" not in txt:
        raise RuntimeError(f"{path}: no internalField entry")

    m_uni = re.search(r"internalField\s+uniform\s+([-+0-9eE\.]+)\s*;", txt)
    if m_uni:
        raise RuntimeError(f"{path}: uniform internalField not supported for this dataset")

    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\((.*?)\)\s*;",
        txt,
        flags=re.S,
    )
    if not m:
        raise RuntimeError(f"{path}: failed to parse nonuniform List<scalar> internalField")

    n = int(m.group(1))
    body = m.group(2)

    vals = np.fromstring(body.replace("\n", " "), sep=" ", dtype=dtype)
    if vals.size != n:
        raise RuntimeError(f"{path}: declared N={n} but parsed {vals.size} values")
    return vals


def _time_name_to_index(name: str) -> int:
    v = float(name)
    iv = int(round(v))
    if abs(v - iv) > 1e-9:
        raise ValueError(
            f"time '{name}' is non-integer. Use integer time names for kept_times compatibility."
        )
    return iv


def fetch_rank_over_time(
    *,
    proc_dir: Path,
    x_fields: list[str],
    y_field: str,
    dtype: np.dtype,
    start: float | None,
    end: float | None,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    kept: list[int] = []
    kept_names: list[str] = []

    expected_N: int | None = None

    tdirs = _list_time_dirs(proc_dir)
    if start is not None:
        tdirs = [p for p in tdirs if float(p.name) >= start]
    if end is not None:
        tdirs = [p for p in tdirs if float(p.name) <= end]
    if stride > 1:
        tdirs = tdirs[::stride]

    for tp in tdirs:
        tname = tp.name
        try:
            x_parts = []
            for xf in x_fields:
                f = tp / xf
                if not f.exists():
                    gz = tp / f"{xf}.gz"
                    if gz.exists():
                        raise RuntimeError(f"{gz}: .gz not supported by this utility")
                    raise FileNotFoundError(f"missing {f}")
                x_parts.append(_ensure_2d(_parse_surface_scalar_internal(f, dtype)))
            x_r = np.concatenate(x_parts, axis=1)

            yf = tp / y_field
            if not yf.exists():
                gz = tp / f"{y_field}.gz"
                if gz.exists():
                    raise RuntimeError(f"{gz}: .gz not supported by this utility")
                raise FileNotFoundError(f"missing {yf}")
            y_r = _ensure_2d(_parse_surface_scalar_internal(yf, dtype))

            if x_r.shape[0] != y_r.shape[0]:
                raise RuntimeError(f"N mismatch: X={x_r.shape[0]} Y={y_r.shape[0]}")

            if expected_N is None:
                expected_N = x_r.shape[0]
            elif x_r.shape[0] != expected_N:
                raise RuntimeError(f"N changed: expected {expected_N} but got {x_r.shape[0]}")

            x_r = _nan_to_num_inplace(x_r)
            y_r = _nan_to_num_inplace(y_r)

            X_list.append(x_r)
            Y_list.append(y_r)
            kept.append(_time_name_to_index(tname))
            kept_names.append(tname)

        except Exception as e:
            print(f"[WARN] {proc_dir.name} skip t={tname} due to: {e}")
            continue

    if not X_list:
        raise RuntimeError(f"{proc_dir.name}: no valid (X,Y) fetched from functionObject outputs")

    X_time = np.stack(X_list, axis=0)
    Y_time = np.stack(Y_list, axis=0)
    kept_times = np.asarray(kept, dtype=np.int32)
    return X_time, Y_time, kept_times, kept_names


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, help="Top OpenFOAM case containing processor*/<time>/")
    p.add_argument("--outdir", default="data/of-raw-ranks", help="output directory for per-rank npz")
    p.add_argument(
        "--x-fields",
        nargs="+",
        default=["rAUf_ml_teacher", "phiHbyA_ml_teacher", "divPhiHbyA_f_ml_teacher"],
        help="X field names written by functionObject",
    )
    p.add_argument("--y-field", default="pEqnFlux_ml_teacher", help="Y field name written by functionObject")
    p.add_argument("--nproc", type=int, required=True, help="number of MPI ranks (processors)")
    p.add_argument("--processor-prefix", default="processor", help="processor directory prefix")
    p.add_argument("--stride", type=int, default=1, help="take every N time directories after sorting")
    p.add_argument("--start", type=float, default=None, help="start time (inclusive)")
    p.add_argument("--end", type=float, default=None, help="end time (inclusive)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="output dtype")
    args = p.parse_args()

    case = Path(args.case).resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dtype = np.float32 if args.dtype == "float32" else np.float64

    print(f"[INFO] case={case}")
    print(f"[INFO] ranks: 0..{args.nproc-1}")

    for r in _iter_ranks(case, args.nproc, args.processor_prefix):
        proc_dir = case / f"{args.processor_prefix}{r}"
        Xr, Yr, kept_times, kept_names = fetch_rank_over_time(
            proc_dir=proc_dir,
            x_fields=args.x_fields,
            y_field=args.y_field,
            dtype=dtype,
            start=args.start,
            end=args.end,
            stride=args.stride,
        )

        info = {
            "fn_name": np.array(["functionObject_files"], dtype=object),
            "rank": np.int32(r),
            "nproc": np.int32(args.nproc),
            "x_fields": np.array(args.x_fields, dtype=object),
            "y_field": np.array([args.y_field], dtype=object),
            "start": np.float64(args.start if args.start is not None else np.nan),
            "end": np.float64(args.end if args.end is not None else np.nan),
            "stride": np.int32(args.stride),
            "kept_times": kept_times,
            "kept_time_names": np.array(kept_names, dtype=object),
            "X_shape": np.array(Xr.shape, dtype=np.int64),
            "Y_shape": np.array(Yr.shape, dtype=np.int64),
        }

        out_path = outdir / f"cfd_xy_rank{r:04d}.npz"
        np.savez_compressed(out_path, X=Xr, Y=Yr, **info)
        print(f"[INFO] wrote {out_path}  X={Xr.shape}  Y={Yr.shape}  kept_times={len(kept_times)}")

    print("[INFO] done (per-rank NPZ written from functionObject outputs)")


if __name__ == "__main__":
    main()

