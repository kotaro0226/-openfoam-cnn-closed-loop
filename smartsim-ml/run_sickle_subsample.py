#!/usr/bin/env python3
"""
Minimal wrapper to run SICKLE subsample.py for directions x/y/z.

Usage (example; pass-through style):
  python run_sickle_subsample.py \
    --sickle_root /path/to/sickle_main \
    -- \
    --dtype of-face-npz \
    --xy_npz /path/to/cfd_xy.npz \
    --maps_npz /path/to/face_maps_rank0_internal.npz \
    --cv_index 2 \
    --hypercubes random \
    --method random \
    --num_hypercubes 32 \
    --num_samples 5000 \
    --nxsl 16 --nysl 16 --nzsl 16 \
    --window 1 \
    --output_dir ./snapshots \
    --fileprefix ofFace_{direction}_{hypercubes}_{method}
"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()

    # only this wrapper's arg
    p.add_argument("--sickle_root", required=True, help="directory that contains subsample.py (sickle_main)")
    # everything after -- is passed to subsample.py (same for x/y/z)
    p.add_argument("passthrough", nargs=argparse.REMAINDER,
                   help="args for subsample.py after '--'")

    args = p.parse_args()
    sickle_root = Path(args.sickle_root).resolve()

    if not sickle_root.exists():
        raise FileNotFoundError(f"--sickle_root not found: {sickle_root}")
    if not (sickle_root / "subsample.py").exists():
        raise FileNotFoundError(f"subsample.py not found under: {sickle_root}")

    passthrough = args.passthrough
    if not passthrough or passthrough[0] != "--":
        raise SystemExit("Need '--' before subsample.py args. Example: python run_sickle_subsample.py --sickle_root ... -- <subsample args...>")

    base_args = passthrough[1:]  # drop the leading "--"

    for d in ["x", "y", "z"]:
        cmd = ["python", "-u", "subsample.py", "--direction", d] + base_args
        print("[INFO] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(sickle_root))

    print("[INFO] done (x/y/z).")


if __name__ == "__main__":
    main()



