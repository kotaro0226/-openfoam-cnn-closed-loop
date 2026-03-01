#!/usr/bin/env python3
"""
Fetch face-wise tensors from Redis (via SmartRedis) and write per-rank NPZ files.

WHY per-rank?
- In parallel OpenFOAM, Redis tensors are in *processor-local face ordering*.
- Downstream subsampling expects *global internal face ordering* (top case neighbour).
- We therefore:
  (1) Save per-rank face-wise time series without modifying length/order.
  (2) Assemble into global internal ordering using faceProcAddressing (separate script).

IMPORTANT invariants for downstream mapping:
- Never drop rows (faces). No masking that changes N.
- Never trim to minN. If a timestep is inconsistent, skip the whole timestep.
- Keep rank ordering exactly as in OpenFOAM processor mesh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_this_dir = Path(__file__).resolve().parent
_parent_dir = _this_dir.parent
for _d in (_this_dir, _parent_dir):
    _s = str(_d)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from of_common import connect, get_field_name


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    """Ensure tensors are (N, C)."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    # Flatten everything but first dim
    return a.reshape(a.shape[0], -1)


def _get_int_meta(meta, key: str, default: int | None = None) -> int | None:
    try:
        return int(meta.get_meta_strings(key)[0])
    except Exception:
        return default


def _time_range_from_metadata(meta) -> tuple[int, int]:
    """Return (start, end) inclusive time indices from Redis metadata."""
    start = _get_int_meta(meta, "StartTimeIndex", 0)
    end = _get_int_meta(meta, "EndTimeIndex", None)
    if end is None:
        end = _get_int_meta(meta, "EndIndex", None)
    if end is None:
        raise RuntimeError(
            "Could not find EndTimeIndex (or EndIndex) in metadata. "
            "Inspect fn_name+'_metadata' in Redis."
        )
    return start, end


def _nan_to_num_inplace(a: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0 without changing shape."""
    # np.nan_to_num returns a new array unless copy=False and dtype allows
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _iter_time_indices(start: int, end: int, stride: int) -> Iterable[int]:
    return range(start, end + 1, stride)


def fetch_rank_over_time(
    *,
    client,
    fn_name: str,
    x_fields: list[str],
    y_field: str,
    rank: int,
    time_indices: list[int],
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fetch X/Y for ONE rank over specified time indices.

    Returns:
      X_time: (T_keep, Nf_local, Fx)
      Y_time: (T_keep, Nf_local, Fy)
      kept_times: (T_keep,)
    """
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    kept: list[int] = []

    expected_N: int | None = None

    for t in time_indices:
        try:
            x_parts = []
            for xf in x_fields:
                key = get_field_name(client, fn_name, xf, processor=rank, timestep=t)
                x_parts.append(_ensure_2d(client.get_tensor(key)).astype(dtype, copy=False))
            x_r = np.concatenate(x_parts, axis=1)  # (N, Fx)

            y_key = get_field_name(client, fn_name, y_field, processor=rank, timestep=t)
            y_r = _ensure_2d(client.get_tensor(y_key)).astype(dtype, copy=False)  # (N, Fy)

            if x_r.shape[0] != y_r.shape[0]:
                raise RuntimeError(f"N mismatch: X={x_r.shape[0]} Y={y_r.shape[0]}")

            if expected_N is None:
                expected_N = x_r.shape[0]
            elif x_r.shape[0] != expected_N:
                # DO NOT trim. Skip this timestep.
                raise RuntimeError(f"N changed: expected {expected_N} but got {x_r.shape[0]}")

            # Replace NaN/inf but keep rows
            x_r = _nan_to_num_inplace(x_r)
            y_r = _nan_to_num_inplace(y_r)

            X_list.append(x_r)
            Y_list.append(y_r)
            kept.append(t)

        except Exception as e:
            print(f"[WARN] rank={rank} skip t={t} due to: {e}")
            continue

    if not X_list:
        raise RuntimeError(
            f"Rank {rank}: no valid (X,Y) fetched. "
            "Check that fields exist for this rank/time range."
        )

    X_time = np.stack(X_list, axis=0)
    Y_time = np.stack(Y_list, axis=0)
    kept_times = np.asarray(kept, dtype=np.int32)
    return X_time, Y_time, kept_times


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="data/of-raw-ranks", help="output directory for per-rank npz")
    p.add_argument(
        "--x-fields",
        nargs="+",
        default=["rAUf_ml", "phiHbyA_ml", "divPhiHbyA_f_ml"],
        help="X field names (face-wise)",
    )
    p.add_argument("--y-field", default="pEqnFlux_ml", help="Y field name (face-wise)")
    p.add_argument("--nproc", type=int, required=True, help="number of MPI ranks (processors)")
    p.add_argument("--stride", type=int, default=1, help="time stride")
    p.add_argument("--start", type=int, default=None, help="override StartTimeIndex")
    p.add_argument("--end", type=int, default=None, help="override EndTimeIndex")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="output dtype")
    args = p.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client, _exp, _of_case_name, fn_name = connect()

    # read metadata to define time range
    client.poll_dataset(fn_name + "_metadata", 10, 1000)
    meta = client.get_dataset(fn_name + "_metadata")
    meta_start, meta_end = _time_range_from_metadata(meta)

    start = meta_start if args.start is None else args.start
    end = meta_end if args.end is None else args.end
    if start > end:
        raise ValueError(f"start({start}) must be <= end({end})")

    time_indices = list(_iter_time_indices(start, end, args.stride))
    print(f"[INFO] time indices: {time_indices[0]}..{time_indices[-1]} stride={args.stride} (count={len(time_indices)})")
    print(f"[INFO] ranks: 0..{args.nproc-1}")

    # Fetch each rank independently and write
    for r in range(args.nproc):
        Xr, Yr, kept_times = fetch_rank_over_time(
            client=client,
            fn_name=fn_name,
            x_fields=args.x_fields,
            y_field=args.y_field,
            rank=r,
            time_indices=time_indices,
            dtype=dtype,
        )

        info = {
            "fn_name": fn_name,
            "rank": np.int32(r),
            "nproc": np.int32(args.nproc),
            "x_fields": np.array(args.x_fields, dtype=object),
            "y_field": np.array([args.y_field], dtype=object),
            "start": np.int32(start),
            "end": np.int32(end),
            "stride": np.int32(args.stride),
            "kept_times": kept_times,
            "X_shape": np.array(Xr.shape, dtype=np.int64),
            "Y_shape": np.array(Yr.shape, dtype=np.int64),
        }

        out_path = outdir / f"cfd_xy_rank{r:04d}.npz"
        np.savez_compressed(out_path, X=Xr, Y=Yr, **info)
        print(f"[INFO] wrote {out_path}  X={Xr.shape}  Y={Yr.shape}  kept_times={len(kept_times)}")

    print("[INFO] done (per-rank NPZ written)")


if __name__ == "__main__":
    main()

