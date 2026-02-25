#!/usr/bin/env python3
"""Merge multiple subsampled NPZ files into one NPZ for training.

This script is intended for workflows where CFD/subsampling data is generated in
multiple continuation runs and later combined before training.

Example:
    python utils/marge-npz.py \
      --inputs "data/sampled/subsampled_part*.npz" \
      --output data/sampled/subsampled_merged.npz
"""

import argparse
import glob
import os
from typing import Dict, List

import numpy as np


CONCAT_KEYS = {
    "X",
    "Y",
    "indices_list",
    "V_sub_list",
    "cell_face_sign_sub_list",
}

META_KEYS = {
    "x",
    "y",
    "z",
    "num_hypercubes",
    "num_samples",
    "T_effective",
    "dims_cells",
}


def _resolve_inputs(inputs: List[str]) -> List[str]:
    resolved: List[str] = []
    for pattern in inputs:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matched: {pattern}")
        resolved.extend(matches)

    # Keep order but deduplicate
    unique = list(dict.fromkeys(resolved))
    if len(unique) < 2:
        raise ValueError("Need at least 2 input NPZ files to merge.")
    return unique


def _check_meta_equal(key: str, arrays: List[np.ndarray], strict: bool) -> np.ndarray:
    ref = arrays[0]
    for i, arr in enumerate(arrays[1:], start=1):
        if ref.shape != arr.shape or not np.array_equal(ref, arr):
            msg = (
                f"Metadata mismatch for key '{key}' between file[0] and file[{i}] "
                f"(shape/value differ)."
            )
            if strict:
                raise ValueError(msg)
            print(f"[WARN] {msg} Using value from first file.")
            return ref
    return ref


def merge_npz(files: List[str], strict_meta: bool = True) -> Dict[str, np.ndarray]:
    loaded = [np.load(f, allow_pickle=True) for f in files]

    # Union of keys present across files
    all_keys = sorted({k for z in loaded for k in z.files})
    merged: Dict[str, np.ndarray] = {}

    for key in all_keys:
        present = [key in z.files for z in loaded]
        if not all(present):
            missing = [files[i] for i, ok in enumerate(present) if not ok]
            raise KeyError(f"Key '{key}' missing in: {missing}")

        arrays = [z[key] for z in loaded]

        if key in CONCAT_KEYS:
            # Validate all dims except axis=0
            base_shape = arrays[0].shape[1:]
            for i, arr in enumerate(arrays[1:], start=1):
                if arr.shape[1:] != base_shape:
                    raise ValueError(
                        f"Concat shape mismatch in key '{key}': "
                        f"file[0]={arrays[0].shape}, file[{i}]={arr.shape}"
                    )
            merged[key] = np.concatenate(arrays, axis=0)
        elif key in META_KEYS:
            merged[key] = _check_meta_equal(key, arrays, strict=strict_meta)
        else:
            # Fallback: require equality and keep first
            merged[key] = _check_meta_equal(key, arrays, strict=True)

    # Update T_effective to reflect merged length if key exists.
    if "T_effective" in merged and "X" in merged:
        merged["T_effective"] = np.int32(merged["X"].shape[0])

    return merged


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge multiple subsampled NPZ files.")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input NPZ files or glob patterns. Example: 'data/sampled/part*.npz'",
    )
    p.add_argument("--output", required=True, help="Output merged NPZ file path.")
    p.add_argument(
        "--no-strict-meta",
        action="store_true",
        help="Allow metadata differences (x/y/z etc.) and keep values from first file.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    files = _resolve_inputs(args.inputs)
    print("[INFO] Input files:")
    for f in files:
        print(f"  - {f}")

    merged = merge_npz(files, strict_meta=not args.no_strict_meta)

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    np.savez(args.output, **merged)
    print(f"[INFO] Saved merged NPZ: {args.output}")
    if "X" in merged:
        print(f"[INFO] X shape: {merged['X'].shape}")
    if "Y" in merged:
        print(f"[INFO] Y shape: {merged['Y'].shape}")


if __name__ == "__main__":
    main()

