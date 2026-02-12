"""Build supervised learning arrays (X, Y) from OpenFOAM -> Redis tensors.

This project stores OpenFOAM fields/tensors into Redis via a functionObject.
The naming is described by the dataset metadata dataset template and field template.

Goal :
  X = [rAUf_ml, phiHbyA_ml divPhiHbyA_f_ml]
  Y = [pEqnFlux_ml]  # == p_rghEqn.flux()

This script fetches those tensors over multiple time indices and outputs an .npz file.

Typical usage:
  python make_cfd_dataset.py --out cfd_xy.npz
  python make_cfd_dataset.py --out cfd_xy.npz --stride 5 --as-samples

Notes:
  - By default we assume mpi_rank=0 only. If you ran OpenFOAM in parallel and you
    want to include all ranks, pass --ranks 0 1 2 ...
  - If you include multiple ranks, arrays are concatenated in rank order.
    This is *not* the same as reconstructing a global field ordering; it's fine
    for ML training if you don't require exact geometric adjacency.
"""

from __future__ import annotations

import argparse
import numpy as np

from of_common import connect, get_field_name


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    """Make sure tensors are (N, C)."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    # For safety: flatten everything but last dim
    return a.reshape(a.shape[0], -1)


def _get_int_meta(meta, key: str, default: int | None = None) -> int | None:
    try:
        return int(meta.get_meta_strings(key)[0])
    except Exception:
        return default


def _time_range_from_metadata(meta) -> tuple[int, int]:
    """Return (start, end) time indices inclusive.

    openfoam-smartsim metadata usually contains StartTimeIndex/EndTimeIndex.
    If StartTimeIndex is missing we assume 0.
    """
    start = _get_int_meta(meta, "StartTimeIndex", 0)
    end = _get_int_meta(meta, "EndTimeIndex", None)
    if end is None:
        # Fallback keys seen in some examples
        end = _get_int_meta(meta, "EndIndex", None)
    if end is None:
        raise RuntimeError(
            "Could not find EndTimeIndex (or EndIndex) in metadata. "
            "Inspect fn_name+'_metadata' in Redis to see available keys."
        )
    return start, end


def fetch_xy_over_time(
    *,
    x_fields: list[str],
    y_field: str,
    ranks: list[int],
    stride: int,
    start: int | None,
    end: int | None,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fetch X/Y tensors over time indices.

    Returns:
      X_time: (T, N, Fx) float32
      Y_time: (T, N, Fy) float32
      info: dict with metadata and shapes
    """
    client, _exp, _db, _of_case_name, fn_name = connect()

    client.poll_dataset(fn_name + "_metadata", 10, 1000)
    meta = client.get_dataset(fn_name + "_metadata")

    meta_start, meta_end = _time_range_from_metadata(meta)
    if start is None:
        start = meta_start
    if end is None:
        end = meta_end
    if start > end:
        raise ValueError(f"start({start}) must be <= end({end})")

    time_indices = list(range(start, end + 1, stride))

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    kept_times: list[int] = []

    for t in time_indices:
        x_rank_chunks: list[np.ndarray] = []
        y_rank_chunks: list[np.ndarray] = []

        ok = True
        for r in ranks:
            try:
                x_parts = []
                for xf in x_fields:
                    key = get_field_name(client, fn_name, xf, processor=r, timestep=t)
                    x_parts.append(_ensure_2d(client.get_tensor(key)).astype(dtype, copy=False))
                x_r = np.concatenate(x_parts, axis=1)  # (N, Fx)

                y_key = get_field_name(client, fn_name, y_field, processor=r, timestep=t)
                y_r = _ensure_2d(client.get_tensor(y_key)).astype(dtype, copy=False)  # (N, Fy)

                if x_r.shape[0] != y_r.shape[0]:
                    raise RuntimeError(
                        f"Shape mismatch at t={t}, rank={r}: X has N={x_r.shape[0]} but Y has N={y_r.shape[0]}"
                    )

                x_rank_chunks.append(x_r)
                y_rank_chunks.append(y_r)

            except Exception as e:
                # Some time indices might not exist depending on when the functionObject fires.
                # We skip such t.
                print(f"[WARN] skip t={t} (rank={r}) due to: {e}")
                ok = False
                break

        if not ok:
            continue

        X_t = np.concatenate(x_rank_chunks, axis=0) if len(x_rank_chunks) > 1 else x_rank_chunks[0]
        Y_t = np.concatenate(y_rank_chunks, axis=0) if len(y_rank_chunks) > 1 else y_rank_chunks[0]

        # Basic sanitization
        if not np.isfinite(X_t).all() or not np.isfinite(Y_t).all():
            print(f"[WARN] non-finite values at t={t}; filtering those rows")
            mask = np.isfinite(X_t).all(axis=1) & np.isfinite(Y_t).all(axis=1)
            X_t = X_t[mask]
            Y_t = Y_t[mask]

        X_list.append(X_t)
        Y_list.append(Y_t)
        kept_times.append(t)

    if not X_list:
        raise RuntimeError(
            "No (X,Y) pairs were fetched. "
            "Check that your functionObject is writing the requested fields, "
            "and that fn_name and field names match."
        )

    # Enforce constant N across time by trimming to min N if needed
    Ns = [x.shape[0] for x in X_list]
    minN = min(Ns)
    if len(set(Ns)) != 1:
        print(f"[WARN] N differs across time steps: {sorted(set(Ns))}. Trimming to minN={minN}.")
        X_list = [x[:minN] for x in X_list]
        Y_list = [y[:minN] for y in Y_list]

    X_time = np.stack(X_list, axis=0)  # (T, N, Fx)
    Y_time = np.stack(Y_list, axis=0)  # (T, N, Fy)

    info = {
        "fn_name": fn_name,
        "x_fields": x_fields,
        "y_field": y_field,
        "ranks": ranks,
        "start": start,
        "end": end,
        "stride": stride,
        "kept_times": np.array(kept_times, dtype=np.int32),
        "X_shape": np.array(X_time.shape, dtype=np.int32),
        "Y_shape": np.array(Y_time.shape, dtype=np.int32),
    }
    return X_time, Y_time, info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cfd_xy.npz", help="output .npz")
    p.add_argument("--x-fields", nargs="+", default=["rAUf_ml", "phiHbyA_ml","divPhiHbyA_f_ml"], help="X field names")
    p.add_argument("--y-field", default="pEqnFlux_ml", help="Y field name")
    p.add_argument("--ranks", nargs="+", type=int, default=[0], help="mpi ranks to include")
    p.add_argument("--stride", type=int, default=1, help="time stride")
    p.add_argument("--start", type=int, default=None, help="override StartTimeIndex")
    p.add_argument("--end", type=int, default=None, help="override EndTimeIndex")
    p.add_argument("--as-samples", action="store_true", help="flatten (T,N,*) -> (T*N,*)")
    args = p.parse_args()

    X_time, Y_time, info = fetch_xy_over_time(
        x_fields=args.x_fields,
        y_field=args.y_field,
        ranks=args.ranks,
        stride=args.stride,
        start=args.start,
        end=args.end,
        dtype=np.float32,
    )

    if args.as_samples:
        T, N, Fx = X_time.shape
        Fy = Y_time.shape[-1]
        X = X_time.reshape(T * N, Fx)
        Y = Y_time.reshape(T * N, Fy)
    else:
        X, Y = X_time, Y_time

    np.savez_compressed(args.out, X=X, Y=Y, **info)
    print(f"[INFO] wrote {args.out}")
    print(f"[INFO] X.shape={X.shape} Y.shape={Y.shape} (saved)")


if __name__ == "__main__":
    main()

