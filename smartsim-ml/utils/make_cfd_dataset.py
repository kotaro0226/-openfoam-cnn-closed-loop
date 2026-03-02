#!/usr/bin/env python3
"""
Fetch face-wise tensors from Redis (via SmartRedis) and write per-rank NPZ files.

Revision note:
- This _rev variant keeps fetch_rank_over_time() and _fetch_and_save_rank() signatures aligned
  for multiprocessing calls (prevents unexpected keyword argument 'client' regressions).

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
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Any

import numpy as np
from smartredis import Client
import jinja2 as jj


env = jj.Environment()


def _connect_client() -> Client:
    addr = os.environ["SSDB"]
    cluster = (os.environ.get("SSDB_CLUSTER", "0") == "1")
    return Client(address=addr, cluster=cluster)


def _get_fn_name(args_fn_name: str | None) -> str:
    return args_fn_name or os.environ.get("SS_FN_NAME", "pEqnInnerTest")


def _safe_meta_string(meta: Any, key: str, default: str | None = None) -> str | None:
    try:
        vals = meta.get_meta_strings(key)
        if vals:
            return str(vals[0])
    except Exception:
        return default
    return default


def _fetch_metadata_templates(client: Client, fn_name: str) -> tuple[object, str, str]:
    client.poll_dataset(fn_name + "_metadata", 10, 1000)
    meta = client.get_dataset(fn_name + "_metadata")
    ds_tmpl = _safe_meta_string(meta, "dataset")
    field_tmpl = _safe_meta_string(meta, "field")
    if not ds_tmpl or not field_tmpl:
        raise RuntimeError(
            "metadata must contain 'dataset' and 'field' templates in "
            f"{fn_name}_metadata"
        )
    return meta, ds_tmpl, field_tmpl


def _safe_render(template: str, **ctx: Any) -> str:
    # 1) Jinja2 template path
    if "{{" in template or "{%" in template:
        return env.from_string(template).render(**ctx)

    # 2) Python format path
    try:
        return template.format(**ctx)
    except Exception:
        pass

    # 3) Simple token replacement fallback
    out = template
    for k, v in ctx.items():
        out = out.replace(f"{{{k}}}", str(v))
    return out


def _render_field_key(
    *,
    ds_template: str,
    field_template: str,
    field_name: str,
    processor: int,
    timestep: int | None,
) -> str:
    ds_name = _safe_render(ds_template, time_index=timestep, mpi_rank=processor)
    f_name = _safe_render(field_template, name=field_name, patch="internal")
    return f"{{{ds_name}}}.{f_name}"




def get_field_name(
    ds_template: str,
    field_template: str,
    field_name: str,
    processor: int = 0,
    timestep: int | None = None,
) -> str:
    """Backward-compatible key builder kept to avoid NameError in mixed environments."""
    return _render_field_key(
        ds_template=ds_template,
        field_template=field_template,
        field_name=field_name,
        processor=processor,
        timestep=timestep,
    )



def _is_transient_io_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    transient_tokens = (
        "resource temporarily unavailable",
        "errno: 11",
        "errno 11",
        "redis io error",
        "failed to get reply",
        "timeout",
    )
    return any(tok in msg for tok in transient_tokens)


def _get_tensor_with_retry(
    client: Client,
    key: str,
    *,
    max_retries: int,
    retry_base_sec: float,
    retry_cap_sec: float,
    rank: int,
    timestep: int,
) -> np.ndarray:
    attempt = 0
    while True:
        try:
            return client.get_tensor(key)
        except Exception as e:
            if (not _is_transient_io_error(e)) or attempt >= max_retries:
                raise
            sleep_s = min(retry_cap_sec, retry_base_sec * (2 ** attempt))
            print(
                f"[WARN] transient Redis I/O error rank={rank} t={timestep} key={key} "
                f"attempt={attempt + 1}/{max_retries}; retry in {sleep_s:.2f}s: {e}"
            )
            time.sleep(sleep_s)
            attempt += 1

def _ensure_2d(a: np.ndarray) -> np.ndarray:
    """Ensure tensors are (N, C)."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    return a.reshape(a.shape[0], -1)


def _get_int_meta(meta, key: str, default: int | None = None) -> int | None:
    try:
        return int(meta.get_meta_strings(key)[0])
    except Exception:
        return default


def _resolve_time_range(meta, start_override: int | None, end_override: int | None) -> tuple[int, int]:
    """Resolve (start, end) while minimizing metadata lookups that spam SmartRedis errors."""
    if start_override is not None and end_override is not None:
        return start_override, end_override

    if end_override is None:
        end = _get_int_meta(meta, "EndTimeIndex", None)
        if end is None:
            end = _get_int_meta(meta, "EndIndex", None)
        if end is None:
            raise RuntimeError(
                "Could not find EndTimeIndex (or EndIndex) in metadata. "
                "Specify --end explicitly if your metadata schema differs."
            )
    else:
        end = end_override

    if start_override is None:
        # Avoid hard failure/log spam when StartTimeIndex is absent on some FO builds.
        start = _get_int_meta(meta, "StartTimeIndex", 0)
    else:
        start = start_override

    return start, end


def _nan_to_num_inplace(a: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0 without changing shape."""
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _iter_time_indices(start: int, end: int, stride: int) -> Iterable[int]:
    return range(start, end + 1, stride)


def fetch_rank_over_time(
    *,
    client: Client,
    fn_name: str,
    ds_template: str,
    field_template: str,
    x_fields: list[str],
    y_field: str,
    rank: int,
    time_indices: list[int],
    dtype: np.dtype,
    max_retries: int,
    retry_base_sec: float,
    retry_cap_sec: float,
    fail_on_missing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch X/Y for ONE rank over specified time indices."""
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    kept: list[int] = []

    expected_N: int | None = None

    for t in time_indices:
        try:
            x_parts = []
            for xf in x_fields:
                key = get_field_name(
                    ds_template=ds_template,
                    field_template=field_template,
                    field_name=xf,
                    processor=rank,
                    timestep=t,
                )
                x_parts.append(_ensure_2d(_get_tensor_with_retry(client, key, max_retries=max_retries, retry_base_sec=retry_base_sec, retry_cap_sec=retry_cap_sec, rank=rank, timestep=t)).astype(dtype, copy=False))
            x_r = np.concatenate(x_parts, axis=1)

            y_key = get_field_name(
                ds_template=ds_template,
                field_template=field_template,
                field_name=y_field,
                processor=rank,
                timestep=t,
            )
            y_r = _ensure_2d(_get_tensor_with_retry(client, y_key, max_retries=max_retries, retry_base_sec=retry_base_sec, retry_cap_sec=retry_cap_sec, rank=rank, timestep=t)).astype(dtype, copy=False)

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
            kept.append(t)

        except Exception as e:
            if fail_on_missing:
                raise RuntimeError(f"rank={rank} t={t} fetch failed after retries: {e}") from e
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


def _fetch_and_save_rank(
    *,
    outdir: str,
    fn_name: str,
    ds_template: str,
    field_template: str,
    x_fields: list[str],
    y_field: str,
    rank: int,
    nproc: int,
    start: int,
    end: int,
    stride: int,
    time_indices: list[int],
    dtype_name: str,
    compressed: bool,
    max_retries: int,
    retry_base_sec: float,
    retry_cap_sec: float,
    fail_on_missing: bool,
) -> tuple[int, tuple[int, ...], tuple[int, ...], int]:
    dtype = np.float32 if dtype_name == "float32" else np.float64
    client = _connect_client()

    Xr, Yr, kept_times = fetch_rank_over_time(
        client=client,
        fn_name=fn_name,
        ds_template=ds_template,
        field_template=field_template,
        x_fields=x_fields,
        y_field=y_field,
        rank=rank,
        time_indices=time_indices,
        dtype=dtype,
        max_retries=max_retries,
        retry_base_sec=retry_base_sec,
        retry_cap_sec=retry_cap_sec,
        fail_on_missing=fail_on_missing,
    )

    info = {
        "fn_name": fn_name,
        "rank": np.int32(rank),
        "nproc": np.int32(nproc),
        "x_fields": np.array(x_fields, dtype=object),
        "y_field": np.array([y_field], dtype=object),
        "start": np.int32(start),
        "end": np.int32(end),
        "stride": np.int32(stride),
        "kept_times": kept_times,
        "X_shape": np.array(Xr.shape, dtype=np.int64),
        "Y_shape": np.array(Yr.shape, dtype=np.int64),
    }

    out_path = Path(outdir) / f"cfd_xy_rank{rank:04d}.npz"
    if compressed:
        np.savez_compressed(out_path, X=Xr, Y=Yr, **info)
    else:
        np.savez(out_path, X=Xr, Y=Yr, **info)

    return rank, Xr.shape, Yr.shape, len(kept_times)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="data/of-raw-ranks", help="output directory for per-rank npz")
    p.add_argument("--fn-name", default=None, help="functionObject name in Redis metadata")
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
    p.add_argument("--compressed", action="store_true", help="write compressed .npz output")
    p.add_argument("--workers", type=int, default=4, help="number of worker processes for rank-wise fetch")
    p.add_argument("--max-retries", type=int, default=12, help="max retries for transient Redis I/O per tensor")
    p.add_argument("--retry-base-sec", type=float, default=0.05, help="initial retry backoff seconds")
    p.add_argument("--retry-cap-sec", type=float, default=2.0, help="maximum retry backoff seconds")
    p.add_argument("--skip-on-error", action="store_true", help="allow skipping failed timesteps (default: fail to avoid data loss)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fn_name = _get_fn_name(args.fn_name)

    # Read metadata once and cache naming templates
    meta_client = _connect_client()
    meta, ds_template, field_template = _fetch_metadata_templates(meta_client, fn_name)
    start, end = _resolve_time_range(meta, args.start, args.end)
    if start > end:
        raise ValueError(f"start({start}) must be <= end({end})")

    time_indices = list(_iter_time_indices(start, end, args.stride))
    print(f"[INFO] fn_name: {fn_name}")
    print(f"[INFO] time indices: {time_indices[0]}..{time_indices[-1]} stride={args.stride} (count={len(time_indices)})")
    print(f"[INFO] ranks: 0..{args.nproc-1}")
    print(f"[INFO] workers: {args.workers}")
    print(f"[INFO] compression: {'on' if args.compressed else 'off'}")
    print(f"[INFO] max_retries: {args.max_retries} base={args.retry_base_sec}s cap={args.retry_cap_sec}s")
    print(f"[INFO] fail_on_missing: {not args.skip_on_error}")

    max_workers = max(1, min(args.workers, args.nproc))
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for r in range(args.nproc):
            futures.append(
                ex.submit(
                    _fetch_and_save_rank,
                    outdir=str(outdir),
                    fn_name=fn_name,
                    ds_template=ds_template,
                    field_template=field_template,
                    x_fields=args.x_fields,
                    y_field=args.y_field,
                    rank=r,
                    nproc=args.nproc,
                    start=start,
                    end=end,
                    stride=args.stride,
                    time_indices=time_indices,
                    dtype_name=args.dtype,
                    compressed=args.compressed,
                    max_retries=args.max_retries,
                    retry_base_sec=args.retry_base_sec,
                    retry_cap_sec=args.retry_cap_sec,
                    fail_on_missing=(not args.skip_on_error),
                )
            )

        for fut in as_completed(futures):
            rank, xshape, yshape, n_kept = fut.result()
            out_path = outdir / f"cfd_xy_rank{rank:04d}.npz"
            print(f"[INFO] wrote {out_path}  X={xshape}  Y={yshape}  kept_times={n_kept}")

    print("[INFO] done (per-rank NPZ written)")


if __name__ == "__main__":
    main()


