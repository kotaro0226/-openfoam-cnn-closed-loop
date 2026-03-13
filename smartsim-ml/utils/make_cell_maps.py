#!/usr/bin/env python3
"""
make_cell_maps.py

Build cell-wise connectivity for continuity/divergence loss from:
  - face_maps_*.npz (map_x/map_y/map_z, dims_cells, optional compact_to_global)
  - OpenFOAM polyMesh owner/neighbour (TOP/global case)
  - OpenFOAM fields C (cell centres) and V (cell volumes)

Outputs (npz):
  - cell_faces_idx : (Nc_valid, 6) int32   face ids for each cell (INDEX SPACE OF face maps / Xf axis)
  - cell_face_sign : (Nc_valid, 6) int8    +1 if cell is owner of face, -1 if neighbour
  - cell_ids       : (Nc_valid,) int32     OpenFOAM global cell id for each row
  - cell_ijk       : (Nc_valid, 3) int32   structured indices (i,j,k)
  - V              : (Nc_valid,) float32   cell volumes (by global cell id)
  - dims_cells     : (3,) int32            (Nx,Ny,Nz)
  - face_order     : (6,) <U2             ["xm","xp","ym","yp","zm","zp"]

Key design for "no processor-patch faces":
- face maps can be "compact" (exclude missing global internal faces on processor boundaries).
- In that case, map_* store COMPACT face ids, and missing faces remain -1.
- We keep only cells with all 6 faces present (>=0), so processor-boundary-adjacent cells are excluded.
- For sign computation we use global owner/neighbour:
    faces6_global = compact_to_global[faces6_compact]
  If compact_to_global is absent, faces6 are assumed to be global internal ids.

IMPORTANT FIX:
- cell centre -> (i,j,k) mapping must match the logic used in make_face_maps_from_C.py.
  We therefore use the same "round + unique + dict" mapping with tol_decimals=12 by default.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np


# -------------------------------
# OpenFOAM ASCII field parsers
# -------------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _find_latest_time_dir(case_dir: Path) -> Path | None:
    times = []
    for p in case_dir.iterdir():
        if p.is_dir():
            try:
                t = float(p.name)
                times.append((t, p))
            except ValueError:
                continue
    if not times:
        return None
    return sorted(times, key=lambda x: x[0])[-1][1]


def _read_label_list(path: Path) -> np.ndarray:
    txt = path.read_text(errors="ignore")
    m = re.search(r"\n\s*(\d+)\s*\(\s*", txt)
    if not m:
        raise RuntimeError(f"Could not parse labelList header/count in: {path}")
    n = int(m.group(1))
    body = txt[m.end():]
    vals = re.findall(r"[-+]?\d+", body)
    if len(vals) < n:
        raise RuntimeError(f"Parsed only {len(vals)} entries but expected {n} in {path}")
    return np.asarray(vals[:n], dtype=np.int32)


def _read_nonuniform_scalar_field(path: Path) -> np.ndarray:
    txt = path.read_text(errors="ignore")
    m = re.search(r"nonuniform\s+List<scalar>\s*(\d+)\s*\(\s*", txt)
    if not m:
        m = re.search(r"nonuniform\s+List<\s*scalar\s*>\s*(\d+)\s*\(\s*", txt)
    if not m:
        raise RuntimeError(f"Could not find nonuniform List<scalar> in: {path}")
    n = int(m.group(1))
    body = txt[m.end():]
    vals = [float(x) for x in _NUM_RE.findall(body)]
    if len(vals) < n:
        raise RuntimeError(f"Parsed only {len(vals)} scalars but expected {n} in {path}")
    return np.asarray(vals[:n], dtype=np.float32)


def _read_nonuniform_vector_field(path: Path) -> np.ndarray:
    txt = path.read_text(errors="ignore")
    m = re.search(r"nonuniform\s+List<vector>\s*(\d+)\s*\(\s*", txt)
    if not m:
        m = re.search(r"nonuniform\s+List<\s*vector\s*>\s*(\d+)\s*\(\s*", txt)
    if not m:
        raise RuntimeError(f"Could not find nonuniform List<vector> in: {path}")
    n = int(m.group(1))
    body = txt[m.end():]
    vals = [float(x) for x in _NUM_RE.findall(body)]
    if len(vals) < 3 * n:
        raise RuntimeError(f"Parsed only {len(vals)//3} vectors but expected {n} in {path}")
    arr = np.asarray(vals[: 3 * n], dtype=np.float64).reshape(n, 3)
    return arr.astype(np.float32, copy=False)


# -------------------------------
# C -> (i,j,k) mapping (MATCH face_maps_from_C.py)
# -------------------------------

def _infer_cell_ijk_from_C_dict(C: np.ndarray, dims_cells: tuple[int, int, int], tol_decimals: int = 12) -> np.ndarray:
    """
    Infer mapping from cell center coordinates to structured (i,j,k) using the SAME logic
    as make_face_maps_from_C.py: round -> unique -> dict mapping.

    Returns:
      cell_ijk: (Nc,3) int32
    """
    Nx, Ny, Nz = dims_cells

    x = np.round(C[:, 0], tol_decimals)
    y = np.round(C[:, 1], tol_decimals)
    z = np.round(C[:, 2], tol_decimals)

    xu = np.unique(x); xu.sort()
    yu = np.unique(y); yu.sort()
    zu = np.unique(z); zu.sort()

    if (len(xu), len(yu), len(zu)) != (Nx, Ny, Nz):
        raise RuntimeError(
            f"Cell-centre unique counts do not match dims_cells:\n"
            f"  unique=({len(xu)},{len(yu)},{len(zu)}) vs dims=({Nx},{Ny},{Nz})\n"
            f"Try changing --tol-decimals (current={tol_decimals})."
        )

    xmap = {v: i for i, v in enumerate(xu)}
    ymap = {v: i for i, v in enumerate(yu)}
    zmap = {v: i for i, v in enumerate(zu)}

    ix = np.fromiter((xmap[v] for v in x), dtype=np.int32, count=len(x))
    iy = np.fromiter((ymap[v] for v in y), dtype=np.int32, count=len(y))
    iz = np.fromiter((zmap[v] for v in z), dtype=np.int32, count=len(z))

    cell_ijk = np.stack([ix, iy, iz], axis=1).astype(np.int32, copy=False)

    # bounds check
    if (
        cell_ijk[:, 0].min() < 0 or cell_ijk[:, 0].max() >= Nx or
        cell_ijk[:, 1].min() < 0 or cell_ijk[:, 1].max() >= Ny or
        cell_ijk[:, 2].min() < 0 or cell_ijk[:, 2].max() >= Nz
    ):
        raise RuntimeError("Computed cell_ijk out of bounds; check coordinate parsing / tol_decimals.")
    return cell_ijk


# -------------------------------
# Core mapping construction
# -------------------------------

def build_cell_maps(
    *,
    face_maps_npz: Path,
    owner: np.ndarray,
    neighbour: np.ndarray,
    C: np.ndarray,
    V: np.ndarray,
    exclude_boundary_cells: bool = True,
    tol_decimals: int = 12,
) -> dict[str, np.ndarray]:
    fm = np.load(face_maps_npz, allow_pickle=True)

    map_x = fm["map_x"].astype(np.int32, copy=False)
    map_y = fm["map_y"].astype(np.int32, copy=False)
    map_z = fm["map_z"].astype(np.int32, copy=False)
    dims_cells = tuple(int(x) for x in fm["dims_cells"].tolist())
    Nx, Ny, Nz = dims_cells

    compact_to_global = None
    if "compact_to_global" in fm.files:
        compact_to_global = fm["compact_to_global"].astype(np.int64, copy=False)
        if compact_to_global.ndim != 1:
            raise RuntimeError("compact_to_global must be 1D")
        print(f"[INFO] compact face maps detected: compact_to_global length={compact_to_global.shape[0]}")

    Nc = C.shape[0]
    if V.shape[0] != Nc:
        raise RuntimeError(f"V length mismatch: V={V.shape[0]} vs C={Nc}")

    # Cell (i,j,k) per OpenFOAM cell id (MATCH face map logic)
    cell_ijk_all = _infer_cell_ijk_from_C_dict(C, dims_cells, tol_decimals=tol_decimals)

    # Inverse mapping: ijk -> cellId
    ijk_to_cell = np.full((Nx, Ny, Nz), -1, dtype=np.int32)
    ijk_to_cell[cell_ijk_all[:, 0], cell_ijk_all[:, 1], cell_ijk_all[:, 2]] = np.arange(Nc, dtype=np.int32)
    if (ijk_to_cell < 0).any():
        missing = np.argwhere(ijk_to_cell < 0)
        raise RuntimeError(f"ijk_to_cell has missing entries ({missing.shape[0]}). Check C parsing/tol_decimals.")

    # Structured interior exclusion (geometric boundary)
    if exclude_boundary_cells:
        i0, i1 = 1, Nx - 2
        j0, j1 = 1, Ny - 2
        k0, k1 = 1, Nz - 2
    else:
        i0, i1 = 0, Nx - 1
        j0, j1 = 0, Ny - 1
        k0, k1 = 0, Nz - 1

    ii, jj, kk = np.meshgrid(
        np.arange(i0, i1 + 1, dtype=np.int32),
        np.arange(j0, j1 + 1, dtype=np.int32),
        np.arange(k0, k1 + 1, dtype=np.int32),
        indexing="ij",
    )
    ijk_valid = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    cell_ids = ijk_to_cell[ijk_valid[:, 0], ijk_valid[:, 1], ijk_valid[:, 2]]

    # Faces for each cell: (xm,xp,ym,yp,zm,zp) in FACE-ID SPACE of map_*
    fxm = map_x[ijk_valid[:, 0],     ijk_valid[:, 1],     ijk_valid[:, 2]]
    fxp = map_x[ijk_valid[:, 0] + 1, ijk_valid[:, 1],     ijk_valid[:, 2]]
    fym = map_y[ijk_valid[:, 0],     ijk_valid[:, 1],     ijk_valid[:, 2]]
    fyp = map_y[ijk_valid[:, 0],     ijk_valid[:, 1] + 1, ijk_valid[:, 2]]
    fzm = map_z[ijk_valid[:, 0],     ijk_valid[:, 1],     ijk_valid[:, 2]]
    fzp = map_z[ijk_valid[:, 0],     ijk_valid[:, 1],     ijk_valid[:, 2] + 1]

    faces6 = np.stack([fxm, fxp, fym, fyp, fzm, fzp], axis=1).astype(np.int32, copy=False)

    # Keep only rows where all 6 faces exist (>=0).
    # This removes physical boundary cells AND processor-boundary-adjacent cells (if compact maps used).
    mask_all6 = (faces6 >= 0).all(axis=1)
    faces6 = faces6[mask_all6]
    cell_ids = cell_ids[mask_all6]
    ijk_valid = ijk_valid[mask_all6]

    if faces6.size == 0:
        raise RuntimeError("No valid cells remain after faces6>=0 filtering. Check face maps / missing faces.")

    # Convert to GLOBAL internal face ids for owner/neighbour lookup
    if compact_to_global is not None:
        if faces6.max() >= compact_to_global.shape[0]:
            raise RuntimeError(
                f"faces6 contains compact id {faces6.max()} >= compact_to_global length {compact_to_global.shape[0]}"
            )
        faces6_global = compact_to_global[faces6].astype(np.int32, copy=False)
        if (faces6_global < 0).any():
            bad = np.argwhere(faces6_global < 0)
            raise RuntimeError(f"compact_to_global returned -1 for some kept faces (count={bad.shape[0]}).")
    else:
        faces6_global = faces6

    # owner/neighbour are GLOBAL internal indexing for neighbour; owner length may be >= internal (total faces).
    nInternalFaces_global = neighbour.shape[0]
    if faces6_global.max() >= nInternalFaces_global:
        raise RuntimeError(
            f"faces6_global contains face id {faces6_global.max()} >= nInternalFaces_global={nInternalFaces_global}. "
            "Check that compact_to_global and top polyMesh are consistent."
        )

    owners = owner[faces6_global]
    neighs = neighbour[faces6_global]

    sign = np.zeros_like(faces6, dtype=np.int8)
    sign[owners == cell_ids[:, None]] = 1
    sign[neighs == cell_ids[:, None]] = -1

    bad = (sign == 0)
    if bad.any():
        bi, bj = np.where(bad)
        example = (
            int(cell_ids[bi[0]]),
            int(faces6[bi[0], bj[0]]),
            int(faces6_global[bi[0], bj[0]]),
            int(owners[bi[0], bj[0]]),
            int(neighs[bi[0], bj[0]]),
        )
        raise RuntimeError(
            "Owner/neighbour sign assignment failed for some (cell, face) pairs.\n"
            f"Example (cellId, faceId_compact, faceId_global, owner, neighbour) = {example}\n"
            "This usually indicates C->ijk mapping mismatch vs face maps.\n"
            f"Try adjusting --tol-decimals (current={tol_decimals})."
        )

    V_valid = V[cell_ids].astype(np.float32, copy=False)

    return {
        "cell_faces_idx": faces6.astype(np.int32, copy=False),   # index space for Xf axis
        "cell_face_sign": sign.astype(np.int8, copy=False),
        "cell_ids": cell_ids.astype(np.int32, copy=False),
        "cell_ijk": ijk_valid.astype(np.int32, copy=False),
        "V": V_valid,
        "dims_cells": np.asarray([Nx, Ny, Nz], dtype=np.int32),
        "face_order": np.asarray(["xm", "xp", "ym", "yp", "zm", "zp"]),
    }


# -------------------------------
# CLI
# -------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--case", required=True, help="OpenFOAM TOP case directory (contains constant/polyMesh)")
    ap.add_argument("--face-maps", required=True, help="Path to face_maps*.npz")
    ap.add_argument("--out", default="cell_maps_internal.npz", help="Output npz path")
    ap.add_argument("--time", default=None, help="Time directory to read C and V from (e.g., '0'). If not set, uses latest numeric time.")
    ap.add_argument("--tol-decimals", type=int, default=12, help="Rounding decimals for C->ijk mapping (must match face map logic).")

    ap.add_argument("--exclude-boundary-cells", action="store_true", default=True,
                    help="Keep only structured interior cells (i=1..Nx-2 etc).")
    ap.add_argument("--include-boundary-cells", action="store_true",
                    help="Include structured boundary cells (still filtered by faces6>=0).")

    args = ap.parse_args()

    case_dir = Path(args.case).resolve()
    poly = case_dir / "constant" / "polyMesh"
    owner_path = poly / "owner"
    neigh_path = poly / "neighbour"
    if not owner_path.exists() or not neigh_path.exists():
        raise SystemExit(f"owner/neighbour not found under: {poly}")

    exclude_boundary = args.exclude_boundary_cells and not args.include_boundary_cells

    # Find time directory
    if args.time is not None:
        time_dir = case_dir / args.time
    else:
        latest = _find_latest_time_dir(case_dir)
        if latest is None:
            raise SystemExit(f"No numeric time directories found under: {case_dir}")
        time_dir = latest

    C_path = time_dir / "C"
    V_path = time_dir / "V"
    if not C_path.exists():
        raise SystemExit(f"Cell centres field not found: {C_path}\nRun: postProcess -func writeCellCentres -case <case>")
    if not V_path.exists():
        raise SystemExit(f"Cell volumes field not found: {V_path}\nRun: postProcess -func writeCellVolumes -case <case>")

    print(f"[INFO] case        : {case_dir}")
    print(f"[INFO] face_maps   : {Path(args.face_maps).resolve()}")
    print(f"[INFO] time_dir    : {time_dir.name}")
    print(f"[INFO] exclude bdy : {exclude_boundary}")
    print(f"[INFO] tol_decimals: {args.tol_decimals}")

    owner = _read_label_list(owner_path)
    neigh = _read_label_list(neigh_path)
    print(f"[INFO] owner len (faces total)   : {owner.shape[0]}")
    print(f"[INFO] neighbour len (internal)  : {neigh.shape[0]}")

    C = _read_nonuniform_vector_field(C_path)
    V = _read_nonuniform_scalar_field(V_path)
    print(f"[INFO] read C: {C.shape}  V: {V.shape}")

    out = build_cell_maps(
        face_maps_npz=Path(args.face_maps).resolve(),
        owner=owner,
        neighbour=neigh,
        C=C,
        V=V,
        exclude_boundary_cells=exclude_boundary,
        tol_decimals=args.tol_decimals,
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"[INFO] wrote {out_path}")
    print(f"[INFO] cell_faces_idx: {out['cell_faces_idx'].shape} (order={out['face_order'].tolist()})")


if __name__ == "__main__":
    main()

