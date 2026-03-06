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

IMPORTANT:
- openfoam-smartsim metadata is assumed to contain naming templates only
  ('dataset' and 'field').
- This script does NOT read StartTimeIndex / EndTimeIndex / EndIndex from metadata.
- Time range is determined by:
    1) --start and --end if both are given
    2) otherwise probing Redis datasets by time_index
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


env = jj.Environment(undefined=jj.StrictUndefined)


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
    # 1) Jinja2 template
    if "{{" in template or "{%" in template:
        return env.from_string(template).render(**ctx)

    # 2) Python format
    try:
        return template.format(**ctx)
    except Exception:
        pass

    # 3) Simple fallback
    out = template
    for k, v in ctx.items():
        out = out.replace(f"{{{k}}}", str(v))
    return out


def _render_context(*, timestep: int | None, processor: int, field_name: str) -> dict[str, Any]:
    """Support multiple metadata naming conventions across functionObject versions."""
    return {
        "time_index": timestep,
        "timeIndex": timestep,
        "timestep": timestep,
        "mpi_rank": processor,
        "mpiRank": processor,
        "rank": processor,
        "processor": processor,
        "name": field_name,
        "patch": "internal",
    }


def _validate_resolved_name(
    name: str, what: str, template: str, *, rank: int, timestep: int | None
) -> None:
    if not name or not str(name).strip():
        raise RuntimeError(
            f"Resolved {what} is empty (rank={rank}, t={timestep}). "
            f"template='{template}'"
        )
    if "{" in name or "}" in name:
        raise RuntimeError(
            f"Resolved {what} still has braces (rank={rank}, t={timestep}). "
            f"template='{template}', resolved='{name}'"
        )


def _render_field_key(
    *,
    ds_template: str,
    field_template: str,
    field_name: str,
    processor: int,
    timestep: int | None,
) -> str:
    ctx = _render_context(timestep=timestep, processor=processor, field_name=field_name)
    ds_name = _safe_render(ds_template, **ctx)
    f_name = _safe_render(field_template, **ctx)

    _validate_resolved_name(ds_name, "dataset name", ds_template, rank=processor, timestep=timestep)
    _validate_resolved_name(f_name, "field name", field_template, rank=processor, timestep=timestep)
    return f"{{{ds_name}}}.{f_name}"


def get_field_name(
    ds_template: str,
    field_template: str,
    field_name: str,
    processor: int = 0,
    timestep: int | None = None,
) -> str:
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


def _nan_to_num_inplace(a: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0 without changing shape."""
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _iter_time_indices(start: int, end: int, stride: int) -> Iterable[int]:
    return range(start, end + 1, stride)


def _dataset_name_for_probe(ds_template: str, *, rank: int, t: int) -> str:
    """
    Build the dataset name only for dataset_exists() probing.
    """
    ctx = _render_context(timestep=t, processor=rank, field_name="__dummy__")
    ds_name = _safe_render(ds_template, **ctx)
    _validate_resolved_name(ds_name, "dataset name", ds_template, rank=rank, timestep=t)
    return ds_name


def _discover_time_range_by_probe(
    client: Client,
    *,
    ds_template: str,
    probe_rank: int,
    stride: int,
    max_time_index: int,
    start_hint: int = 0,
) -> tuple[int, int]:
    """
    Discover [start, end] time_index range by checking dataset existence on one rank.

    Rules:
    - We never read start/end from metadata.
    - We assume time_index is a non-negative integer.
    - We probe only one rank (default rank 0) because naming convention is common.
    """
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    if max_time_index < 0:
        raise ValueError("max_time_index must be >= 0")
    if start_hint < 0:
        start_hint = 0

    def exists(t: int) -> bool:
        ds = _dataset_name_for_probe(ds_template, rank=probe_rank, t=t)
        return bool(client.dataset_exists(ds))

    # 1) find first existing index
    first: int | None = None
    t = start_hint
    while t <= max_time_index:
        if exists(t):
            first = t
            break
        t += stride

    if first is None:
        raise RuntimeError(
            f"Could not discover any dataset for probe_rank={probe_rank} "
            f"in [{start_hint}, {max_time_index}] with stride={stride}. "
            "Specify --start and --end explicitly."
        )

    # 2) exponential search for an upper bracket of the last existing index
    hi_exist = first
    step = stride
    while True:
        cand = hi_exist + step
        if cand > max_time_index:
            break
        if exists(cand):
            hi_exist = cand
            step *= 2
        else:
            break

    # 3) binary search on stride-aligned grid
    left = hi_exist
    right = min(max_time_index, hi_exist + step)

    while left + stride <= right:
        span = right - left
        mid = left + ((span // (2 * stride)) * stride)
        if mid == left:
            mid = left + stride

        if exists(mid):
            left = mid
        else:
            right = mid - stride

    last = left
    return first, last


def _resolve_time_range(
    client: Client,
    *,
    ds_template: str,
    start_override: int | None,
    end_override: int | None,
    probe_rank: int,
    probe_step: int,
    stride: int,
    max_time_index: int,
) -> tuple[int, int]:
    """
    Resolve (start, end) WITHOUT reading StartTimeIndex/EndTimeIndex metadata.

    Priority:
    1) If both --start and --end are given, use them directly.
    2) Otherwise auto-discover by dataset probing.
       - If only --start is given, keep it and probe end.
       - If only --end is given, probe start and keep end.
       - If neither is given, probe both.
    """
    if start_override is not None and end_override is not None:
        return start_override, end_override

    probed_start, probed_end = _discover_time_range_by_probe(
        client,
        ds_template=ds_template,
        probe_rank=probe_rank,
        stride=probe_step,
        max_time_index=max_time_index,
        start_hint=(start_override if start_override is not None else 0),
    )

    start = start_override if start_override is not None else probed_start
    end = end_override if end_override is not None else probed_end
    return start, end


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
                x_parts.append(
                    _ensure_2d(
                        _get_tensor_with_retry(
                            client,
                            key,
                            max_retries=max_retries,
                            retry_base_sec=retry_base_sec,
                            retry_cap_sec=retry_cap_sec,
                            rank=rank,
                            timestep=t,
                        )
                    ).astype(dtype, copy=False)
                )
            x_r = np.concatenate(x_parts, axis=1)

            y_key = get_field_name(
                ds_template=ds_template,
                field_template=field_template,
                field_name=y_field,
                processor=rank,
                timestep=t,
            )
            y_r = _ensure_2d(
                _get_tensor_with_retry(
                    client,
                    y_key,
                    max_retries=max_retries,
                    retry_base_sec=retry_base_sec,
                    retry_cap_sec=retry_cap_sec,
                    rank=rank,
                    timestep=t,
                )
            ).astype(dtype, copy=False)

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
    p.add_argument("--start", type=int, default=None, help="override start time_index")
    p.add_argument("--end", type=int, default=None, help="override end time_index")
    p.add_argument("--probe-rank", type=int, default=0, help="rank used for auto-discovering time_index range")
    p.add_argument(
        "--probe-step",
        type=int,
        default=1,
        help="time_index probing step for auto-discovery (independent of --stride)"
    )
    p.add_argument("--max-time-index", type=int, default=100000, help="upper bound for probing when start/end are not both given")
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

    # Read metadata once and cache naming templates only
    meta_client = _connect_client()
    _meta, ds_template, field_template = _fetch_metadata_templates(meta_client, fn_name)

    # Resolve time range without reading StartTimeIndex / EndTimeIndex metadata
    start, end = _resolve_time_range(
        meta_client,
        ds_template=ds_template,
        start_override=args.start,
        end_override=args.end,
        probe_rank=args.probe_rank,
        probe_step=args.probe_step,
        stride=args.stride,
        max_time_index=args.max_time_index,
    )

    if start > end:
        raise ValueError(f"start({start}) must be <= end({end})")

    time_indices = list(_iter_time_indices(start, end, args.stride))
    if not time_indices:
        raise RuntimeError("No time indices resolved.")

    print(f"[INFO] fn_name: {fn_name}")
    print(f"[INFO] dataset template: {ds_template}")
    print(f"[INFO] field template:   {field_template}")
    print(f"[INFO] time indices: {time_indices[0]}..{time_indices[-1]} stride={args.stride} (count={len(time_indices)})")
    print(f"[INFO] ranks: 0..{args.nproc - 1}")
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


