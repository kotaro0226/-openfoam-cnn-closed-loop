#!/usr/bin/env python3
"""
make_cell_maps.py

Build cell-wise connectivity for continuity/divergence loss from:
  - face_maps_rank0_internal.npz (map_x/map_y/map_z, dims_cells)
  - OpenFOAM polyMesh owner/neighbour
  - OpenFOAM fields C (cell centres) and V (cell volumes) written by:
      postProcess -func writeCellCentres
      postProcess -func writeCellVolumes

Outputs (npz):
  - cell_faces_idx : (Nc_valid, 6) int32   face ids for each cell (internal faces only)
  - cell_face_sign : (Nc_valid, 6) int8    +1 if cell is owner of face, -1 if neighbour
  - cell_ids       : (Nc_valid,) int32     OpenFOAM cell id for each row
  - cell_ijk       : (Nc_valid, 3) int32   structured indices (i,j,k)
  - V              : (Nc_valid,) float32   cell volumes
  - dims_cells     : (3,) int32            (Nx,Ny,Nz)
  - face_order     : (6,) <U2             ["xm","xp","ym","yp","zm","zp"]

Notes:
  - We keep only cells that have all 6 faces INTERNAL (i.e., boundary-adjacent cells are excluded by default).
  - This makes continuity loss cheap and well-defined without reconstructing full fields.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import numpy as np


# -------------------------------
# OpenFOAM ASCII field parsers
# -------------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _find_latest_time_dir(case_dir: Path) -> Path | None:
    """Return latest numeric time directory under case_dir, or None."""
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
    """Read OpenFOAM labelList file (owner/neighbour) in ASCII.

    Robustly parses only the integer list body after: "<n>( ... )"
    """
    txt = path.read_text(errors="ignore")

    # Find: "\n  <n>  (\n"
    m = re.search(r"\n\s*(\d+)\s*\(\s*", txt)
    if not m:
        raise RuntimeError(f"Could not parse labelList header/count in: {path}")
    n = int(m.group(1))

    body = txt[m.end():]  # start right after the opening '('
    # Parse integers only (avoid floats like '2.0' in headers)
    vals = re.findall(r"[-+]?\d+", body)
    if len(vals) < n:
        raise RuntimeError(f"Parsed only {len(vals)} entries but expected {n} in {path}")

    return np.asarray(vals[:n], dtype=np.int32)


def _read_nonuniform_scalar_field(path: Path) -> np.ndarray:
    """Read volScalarField values (nonuniform List<scalar>) in ASCII."""
    txt = path.read_text(errors="ignore")
    m = re.search(r"nonuniform\s+List<scalar>\s*(\d+)\s*\(\s*", txt)
    if not m:
        # Sometimes it is "nonuniform List<Scalar>" (rare) or formatting differs
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
    """Read volVectorField values (nonuniform List<vector>) in ASCII."""
    txt = path.read_text(errors="ignore")
    m = re.search(r"nonuniform\s+List<vector>\s*(\d+)\s*\(\s*", txt)
    if not m:
        m = re.search(r"nonuniform\s+List<\s*vector\s*>\s*(\d+)\s*\(\s*", txt)
    if not m:
        raise RuntimeError(f"Could not find nonuniform List<vector> in: {path}")
    n = int(m.group(1))
    body = txt[m.end():]

    # Extract all floats and reshape into (n,3)
    vals = [float(x) for x in _NUM_RE.findall(body)]
    if len(vals) < 3 * n:
        raise RuntimeError(f"Parsed only {len(vals)//3} vectors but expected {n} in {path}")
    arr = np.asarray(vals[: 3 * n], dtype=np.float64).reshape(n, 3)
    return arr.astype(np.float32, copy=False)


def _infer_cell_ijk_from_C(C: np.ndarray, dims_cells: tuple[int, int, int]) -> np.ndarray:
    """
    Infer mapping from cell center coordinates to structured (i,j,k).
    We do this by:
      - finding unique sorted coords along each axis (within tolerance)
      - mapping each cell center to nearest bin index.

    Returns:
      cell_ijk: (Nc,3) int32
    """
    Nx, Ny, Nz = dims_cells
    x = np.unique(np.round(C[:, 0], 10))
    y = np.unique(np.round(C[:, 1], 10))
    z = np.unique(np.round(C[:, 2], 10))

    if (len(x), len(y), len(z)) != (Nx, Ny, Nz):
        raise RuntimeError(
            f"Cell-centre unique counts do not match dims_cells: "
            f"unique=({len(x)},{len(y)},{len(z)}) vs dims=({Nx},{Ny},{Nz}). "
            f"Check that C is written for the same mesh and that dims_cells is correct."
        )

    # Map coords to indices with searchsorted (robust because we rounded)
    ix = np.searchsorted(x, np.round(C[:, 0], 10))
    iy = np.searchsorted(y, np.round(C[:, 1], 10))
    iz = np.searchsorted(z, np.round(C[:, 2], 10))

    cell_ijk = np.stack([ix, iy, iz], axis=1).astype(np.int32)
    # basic sanity
    if (cell_ijk[:, 0].min() < 0 or cell_ijk[:, 0].max() >= Nx or
        cell_ijk[:, 1].min() < 0 or cell_ijk[:, 1].max() >= Ny or
        cell_ijk[:, 2].min() < 0 or cell_ijk[:, 2].max() >= Nz):
        raise RuntimeError("Computed cell_ijk out of bounds; check coordinate parsing.")
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
) -> dict[str, np.ndarray]:
    """
    Build (cell -> 6 faces) mapping and sign matrix using face maps + owner/neighbour + C + V.
    """
    fm = np.load(face_maps_npz, allow_pickle=True)
    map_x = fm["map_x"].astype(np.int32, copy=False)
    map_y = fm["map_y"].astype(np.int32, copy=False)
    map_z = fm["map_z"].astype(np.int32, copy=False)
    dims_cells = tuple(int(x) for x in fm["dims_cells"].tolist())
    Nx, Ny, Nz = dims_cells

    Nc = C.shape[0]
    if V.shape[0] != Nc:
        raise RuntimeError(f"V length mismatch: V={V.shape[0]} vs C={Nc}")

    # Cell (i,j,k) per OpenFOAM cell id
    cell_ijk = _infer_cell_ijk_from_C(C, dims_cells)

    # Build inverse mapping: ijk -> cellId
    ijk_to_cell = np.full((Nx, Ny, Nz), -1, dtype=np.int32)
    ijk_to_cell[cell_ijk[:, 0], cell_ijk[:, 1], cell_ijk[:, 2]] = np.arange(Nc, dtype=np.int32)
    if (ijk_to_cell < 0).any():
        # Some ijk bins not filled
        missing = np.argwhere(ijk_to_cell < 0)
        raise RuntimeError(f"ijk_to_cell has missing entries ({missing.shape[0]}). Check C parsing/uniques.")

    # Determine which cells to include
    if exclude_boundary_cells:
        # internal cells only: i=1..Nx-2 etc (so that all 6 faces exist and are internal)
        i0, i1 = 1, Nx - 2
        j0, j1 = 1, Ny - 2
        k0, k1 = 1, Nz - 2
    else:
        i0, i1 = 0, Nx - 1
        j0, j1 = 0, Ny - 1
        k0, k1 = 0, Nz - 1

    # Build lists (vectorized over flattened ijk range)
    ii, jj, kk = np.meshgrid(
        np.arange(i0, i1 + 1, dtype=np.int32),
        np.arange(j0, j1 + 1, dtype=np.int32),
        np.arange(k0, k1 + 1, dtype=np.int32),
        indexing="ij",
    )
    ijk_valid = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)  # (Nc_valid_candidate, 3)
    cell_ids = ijk_to_cell[ijk_valid[:, 0], ijk_valid[:, 1], ijk_valid[:, 2]]  # (Nc_valid_candidate,)

    # Faces for each cell: (xm,xp,ym,yp,zm,zp)
    fxm = map_x[ijk_valid[:, 0],     ijk_valid[:, 1],     ijk_valid[:, 2]]
    fxp = map_x[ijk_valid[:, 0] + 1, ijk_valid[:, 1],     ijk_valid[:, 2]]
    fym = map_y[ijk_valid[:, 0],     ijk_valid[:, 1],     ijk_valid[:, 2]]
    fyp = map_y[ijk_valid[:, 0],     ijk_valid[:, 1] + 1, ijk_valid[:, 2]]
    fzm = map_z[ijk_valid[:, 0],     ijk_valid[:, 1],     ijk_valid[:, 2]]
    fzp = map_z[ijk_valid[:, 0],     ijk_valid[:, 1],     ijk_valid[:, 2] + 1]

    faces6 = np.stack([fxm, fxp, fym, fyp, fzm, fzp], axis=1).astype(np.int32, copy=False)

    # Keep only rows where all 6 faces are internal (>=0)
    mask_internal = (faces6 >= 0).all(axis=1)
    faces6 = faces6[mask_internal]
    cell_ids = cell_ids[mask_internal]
    ijk_valid = ijk_valid[mask_internal]

    # Validate face id range for owner/neighbour arrays
    nInternalFaces = owner.shape[0]
    if faces6.max() >= nInternalFaces:
        raise RuntimeError(
            f"faces6 contains face id {faces6.max()} >= nInternalFaces={nInternalFaces}. "
            f"Check that face maps are internal-face indexed."
        )

    # Compute sign for each (cell, face): +1 if cell is owner, -1 if neighbour
    # We'll do this efficiently by vectorized lookup.
    owners = owner[faces6]      # (Nc_valid,6)
    neighs = neighbour[faces6]  # (Nc_valid,6)

    sign = np.zeros_like(faces6, dtype=np.int8)
    sign[owners == cell_ids[:, None]] = 1
    sign[neighs == cell_ids[:, None]] = -1

    bad = (sign == 0)
    if bad.any():
        # This indicates (cell,face) pairing doesn't match owner/neighbour, which shouldn't happen.
        bi, bj = np.where(bad)
        example = (int(cell_ids[bi[0]]), int(faces6[bi[0], bj[0]]),
                   int(owners[bi[0], bj[0]]), int(neighs[bi[0], bj[0]]))
        raise RuntimeError(
            "Owner/neighbour sign assignment failed for some (cell, face) pairs.\n"
            f"Example (cellId, faceId, owner, neighbour) = {example}\n"
            "Check that ijk_to_cell mapping matches OpenFOAM cell numbering and face maps."
        )

    V_valid = V[cell_ids].astype(np.float32, copy=False)

    return {
        "cell_faces_idx": faces6.astype(np.int32, copy=False),
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
    ap.add_argument("--case", required=True, help="OpenFOAM case directory (contains constant/polyMesh)")
    ap.add_argument("--face-maps", required=True, help="Path to face_maps_rank0_internal.npz")
    ap.add_argument("--out", default="cell_maps_rank0_internal.npz", help="Output npz path")

    ap.add_argument(
        "--time",
        default=None,
        help="Time directory to read C and V from (e.g., '0'). If not set, uses latest numeric time.",
    )
    ap.add_argument(
        "--exclude-boundary-cells",
        action="store_true",
        default=True,
        help="Keep only cells that have all 6 INTERNAL faces (recommended for clean div loss).",
    )
    ap.add_argument(
        "--include-boundary-cells",
        action="store_true",
        help="Include boundary-adjacent cells (requires handling boundary faces separately).",
    )

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
    V_path = time_dir / "V"  # writeCellVolumes typically writes 'V'

    if not C_path.exists():
        raise SystemExit(
            f"Cell centres field not found: {C_path}\n"
            "Run: postProcess -func writeCellCentres -case <case>"
        )
    if not V_path.exists():
        raise SystemExit(
            f"Cell volumes field not found: {V_path}\n"
            "Run: postProcess -func writeCellVolumes -case <case>"
        )

    print(f"[INFO] case        : {case_dir}")
    print(f"[INFO] face_maps   : {Path(args.face_maps).resolve()}")
    print(f"[INFO] time_dir    : {time_dir.name}")
    print(f"[INFO] exclude bdy : {exclude_boundary}")

    owner = _read_label_list(owner_path)
    neigh = _read_label_list(neigh_path)
    print(f"[INFO] nInternalFaces from neighbour/owner: {owner.shape[0]}")

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
    )

    out_path = Path(args.out).resolve()
    np.savez_compressed(out_path, **out)
    print(f"[INFO] wrote {out_path}")
    print(f"[INFO] cell_faces_idx: {out['cell_faces_idx'].shape} (order={out['face_order'].tolist()})")


if __name__ == "__main__":
    main()



