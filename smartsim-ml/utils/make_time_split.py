#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from split import save_split


def parse_range(spec: str) -> tuple[int, int]:
    lo, hi = (int(v) for v in spec.split(":", 1))
    if hi < lo:
        raise ValueError(f"invalid range {spec}")
    return lo, hi


def indices_for_times(kept_times: np.ndarray, blocks_per_time: int, lo: int, hi: int) -> np.ndarray:
    out = []
    for time_idx, t in enumerate(kept_times):
        if lo <= int(t) <= hi:
            start = time_idx * blocks_per_time
            out.extend(range(start, start + blocks_per_time))
    return np.asarray(out, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-time", default="100:3000")
    p.add_argument("--val-time", default="3100:4000")
    p.add_argument("--test-time", default="4100:5000")
    args = p.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "kept_times" not in z:
        raise KeyError("dataset must include kept_times")
    kept_times = np.asarray(z["kept_times"], dtype=np.int32)
    n_samples = int(z["block_starts"].shape[0])
    if n_samples % len(kept_times) != 0:
        raise ValueError(f"sample count {n_samples} is not divisible by kept_times count {len(kept_times)}")
    blocks_per_time = n_samples // len(kept_times)

    tr = indices_for_times(kept_times, blocks_per_time, *parse_range(args.train_time))
    va = indices_for_times(kept_times, blocks_per_time, *parse_range(args.val_time))
    te = indices_for_times(kept_times, blocks_per_time, *parse_range(args.test_time))

    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        raise ValueError(f"empty split: train={len(tr)} val={len(va)} test={len(te)}")

    save_split(
        args.out,
        tr,
        va,
        te,
        meta={
            "split_type": "time",
            "train_time": args.train_time,
            "val_time": args.val_time,
            "test_time": args.test_time,
            "blocks_per_time": blocks_per_time,
        },
    )
    print(f"[INFO] wrote {args.out}")
    print(f"[INFO] kept_times={len(kept_times)} blocks_per_time={blocks_per_time}")
    print(f"[INFO] train={len(tr)} val={len(va)} test={len(te)}")


if __name__ == "__main__":
    main()
