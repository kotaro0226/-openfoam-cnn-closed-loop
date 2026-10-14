#!/usr/bin/env python3
"""Compare OpenFOAM internalField values between two case directories."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case-a", required=True, help="reference OpenFOAM case directory")
    p.add_argument("--case-b", required=True, help="comparison OpenFOAM case directory")
    p.add_argument("--time", required=True, help="time directory to compare, e.g. 0.1")
    p.add_argument("--fields", nargs="+", default=["U", "phi", "p_rgh", "p"])
    return p.parse_args()


def read_internal_field(path: Path) -> np.ndarray:
    text = path.read_text(errors="ignore")
    marker = "internalField"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"{path}: internalField not found")
    tail = text[start:]
    if "uniform" in tail.split(";", 1)[0] and "nonuniform" not in tail.split(";", 1)[0]:
        first_stmt = tail.split(";", 1)[0]
        vals = [float(x) for x in _NUM_RE.findall(first_stmt)]
        return np.asarray(vals, dtype=np.float64)

    m = re.search(r"nonuniform\s+List<(\w+)>\s*(\d+)\s*\(\s*", tail)
    if not m:
        raise RuntimeError(f"{path}: unsupported internalField format")
    field_type = m.group(1)
    n = int(m.group(2))
    body = tail[m.end() :]

    if field_type == "vector":
        vals = [float(x) for x in _NUM_RE.findall(body)]
        if len(vals) < 3 * n:
            raise RuntimeError(f"{path}: parsed {len(vals) // 3} vectors, expected {n}")
        return np.asarray(vals[: 3 * n], dtype=np.float64).reshape(n, 3)

    vals = [float(x) for x in _NUM_RE.findall(body)]
    if len(vals) < n:
        raise RuntimeError(f"{path}: parsed {len(vals)} scalars, expected {n}")
    return np.asarray(vals[:n], dtype=np.float64)


def main() -> None:
    args = parse_args()
    case_a = Path(args.case_a)
    case_b = Path(args.case_b)

    print("field,n,rmse,mae,max_abs,rel_l2,mean_a,mean_b")
    for field in args.fields:
        a = read_internal_field(case_a / args.time / field)
        b = read_internal_field(case_b / args.time / field)
        if a.shape != b.shape:
            raise RuntimeError(f"{field}: shape mismatch {a.shape} != {b.shape}")
        diff = b - a
        flat_diff = diff.reshape(-1)
        flat_a = a.reshape(-1)
        rmse = float(np.sqrt(np.mean(flat_diff * flat_diff)))
        mae = float(np.mean(np.abs(flat_diff)))
        max_abs = float(np.max(np.abs(flat_diff)))
        denom = float(np.linalg.norm(flat_a))
        rel_l2 = float(np.linalg.norm(flat_diff) / denom) if denom > 0 else float("nan")
        print(
            f"{field},{a.shape[0]},{rmse:.10e},{mae:.10e},"
            f"{max_abs:.10e},{rel_l2:.10e},{np.mean(flat_a):.10e},{np.mean(b.reshape(-1)):.10e}"
        )


if __name__ == "__main__":
    main()
