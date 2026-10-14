#!/usr/bin/env python3
"""Small OpenFOAM ASCII field readers for closed-loop diagnostics."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def numeric_time_dirs(case_dir: Path) -> list[tuple[float, Path]]:
    out: list[tuple[float, Path]] = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            out.append((float(child.name), child))
        except ValueError:
            continue
    return sorted(out, key=lambda item: item[0])


def time_name(t: float) -> str:
    return f"{t:.12g}"


def read_internal_field(path: Path) -> np.ndarray:
    text = path.read_text(errors="ignore")
    marker = "internalField"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"{path}: internalField not found")
    tail = text[start:]
    first_stmt = tail.split(";", 1)[0]

    if "uniform" in first_stmt and "nonuniform" not in first_stmt:
        vals = np.asarray([float(x) for x in NUM_RE.findall(first_stmt)], dtype=np.float64)
        if "(" in first_stmt and ")" in first_stmt:
            return vals.reshape(1, -1)
        return vals

    m = re.search(r"nonuniform\s+List<(\w+)>\s*(\d+)\s*\(\s*", tail)
    if not m:
        raise RuntimeError(f"{path}: unsupported internalField format")
    field_type = m.group(1)
    n = int(m.group(2))
    values = np.fromiter((float(x) for x in NUM_RE.findall(tail[m.end() :])), dtype=np.float64)

    if field_type == "vector":
        if values.size < 3 * n:
            raise RuntimeError(f"{path}: parsed {values.size // 3} vectors, expected {n}")
        return values[: 3 * n].reshape(n, 3)
    if values.size < n:
        raise RuntimeError(f"{path}: parsed {values.size} scalars, expected {n}")
    return values[:n]


def read_label_list(path: Path) -> np.ndarray:
    text = path.read_text(errors="ignore")
    m = re.search(r"\n\s*(\d+)\s*\(\s*", text)
    if not m:
        raise RuntimeError(f"could not parse labelList: {path}")
    n = int(m.group(1))
    values = np.fromiter((int(x) for x in NUM_RE.findall(text[m.end() :])), dtype=np.int64, count=n)
    if values.size != n:
        raise RuntimeError(f"{path}: parsed {values.size}, expected {n}")
    return values


def read_vector_internal(path: Path) -> np.ndarray:
    arr = read_internal_field(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise RuntimeError(f"{path}: expected vector internalField, got shape {arr.shape}")
    return arr


def read_scalar_internal(path: Path) -> np.ndarray:
    arr = read_internal_field(path)
    if arr.ndim != 1:
        raise RuntimeError(f"{path}: expected scalar internalField, got shape {arr.shape}")
    return arr


def l2_relative(reference: np.ndarray, comparison: np.ndarray) -> float:
    if reference.shape != comparison.shape:
        raise RuntimeError(f"shape mismatch: {reference.shape} != {comparison.shape}")
    ref_flat = reference.reshape(-1)
    diff_flat = (comparison - reference).reshape(-1)
    denom = float(np.linalg.norm(ref_flat))
    return float(np.linalg.norm(diff_flat) / denom) if denom > 0.0 else float("nan")


def infer_cell_ijk(centres: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    n = int(round(centres.shape[0] ** (1.0 / 3.0)))
    if n**3 == centres.shape[0]:
        dims = (n, n, n)
    else:
        unique = [np.unique(np.round(centres[:, axis], decimals=10)) for axis in range(3)]
        dims = tuple(int(v.size) for v in unique)
    mins = centres.min(axis=0)
    maxs = centres.max(axis=0)
    spacing = np.where(np.asarray(dims) > 1, (maxs - mins) / (np.asarray(dims) - 1), 1.0)
    ijk = np.rint((centres - mins) / spacing).astype(np.int32)
    linear = ijk[:, 0] * dims[1] * dims[2] + ijk[:, 1] * dims[2] + ijk[:, 2]
    if ijk.min() < 0 or np.unique(linear).size != centres.shape[0]:
        raise RuntimeError("failed to infer unique structured cell ijk coordinates")
    return ijk, dims


def face_axis(owner_ijk: np.ndarray, neigh_ijk: np.ndarray) -> str:
    delta = np.abs(neigh_ijk - owner_ijk)
    axis = int(np.argmax(delta))
    return "xyz"[axis] if delta[axis] == 1 and delta.sum() == 1 else "unknown"
