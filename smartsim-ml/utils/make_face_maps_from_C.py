from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import re

def parse_openfoam_list(path: Path) -> np.ndarray:
    """Parse OpenFOAM labelList/scalarList etc. Returns 1D int array."""
    txt = path.read_text()
    m = re.search(r"^\s*(\d+)\s*$", txt, re.M)
    if not m:
        raise RuntimeError(f"Could not find list length in {path}")
    n = int(m.group(1))
    # Extract all integers after that header
    nums = list(map(int, re.findall(r"[-]?\d+", txt[m.end():])))
    if len(nums) < n:
        raise RuntimeError(f"Parsed only {len(nums)} ints but expected {n} from {path}")
    return np.asarray(nums[:n], dtype=np.int64)

def parse_openfoam_vector_field(path: Path) -> np.ndarray:
    """Parse OpenFOAM vectorField file like 0/C. Returns (N,3) float array."""
    txt = path.read_text()
    m = re.search(r"^\s*(\d+)\s*$", txt, re.M)
    if not m:
        raise RuntimeError(f"Could not find field length in {path}")
    n = int(m.group(1))
    # vectors appear as (x y z)
    vecs = re.findall(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", txt[m.end():])
    if len(vecs) < n:
        raise RuntimeError(f"Parsed only {len(vecs)} vectors but expected {n} from {path}")
    arr = np.asarray(vecs[:n], dtype=np.float64)
    return arr

def build_index_from_centres(C: np.ndarray, tol_decimals: int = 12):
    """Map each cell to (ix,iy,iz) using unique coords of cell centres."""
    x = np.round(C[:,0], tol_decimals)
    y = np.round(C[:,1], tol_decimals)
    z = np.round(C[:,2], tol_decimals)

    xu = np.unique(x); xu.sort()
    yu = np.unique(y); yu.sort()
    zu = np.unique(z); zu.sort()

    # dict: coordinate -> index
    xmap = {v:i for i,v in enumerate(xu)}
    ymap = {v:i for i,v in enumerate(yu)}
    zmap = {v:i for i,v in enumerate(zu)}

    ix = np.fromiter((xmap[v] for v in x), dtype=np.int64, count=len(x))
    iy = np.fromiter((ymap[v] for v in y), dtype=np.int64, count=len(y))
    iz = np.fromiter((zmap[v] for v in z), dtype=np.int64, count=len(z))

    Nx, Ny, Nz = len(xu), len(yu), len(zu)  # cells counts
    return ix, iy, iz, (Nx, Ny, Nz)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="OpenFOAM case dir")
    ap.add_argument("--npz", required=True, help="cfd_xy*.npz (X: T,Ninternal,F)")
    ap.add_argument("--C", default=None, help="path to cell centres file (e.g. case/0/C). If omitted, tries case/0/C")
    ap.add_argument("--out", default="face_maps_rank0_internal.npz")
    args = ap.parse_args()

    case = Path(args.case)
    pm = case / "constant" / "polyMesh"
    owner_p = pm / "owner"
    neigh_p = pm / "neighbour"

    # internal faces list length = neighbour length
    owner = parse_openfoam_list(owner_p)
    neigh = parse_openfoam_list(neigh_p)
    nInt = len(neigh)

    npz = np.load(args.npz)
    X = npz["X"]
    _, N, _ = X.shape
    if N != nInt:
        raise RuntimeError(f"Dataset Ninternal={N} but neighbour length={nInt}. Check inputs.")

    # cell centres
    C_path = Path(args.C) if args.C else (case / "0" / "C")
    C = parse_openfoam_vector_field(C_path)

    # cell -> (ix,iy,iz)
    ix, iy, iz, (Nx, Ny, Nz) = build_index_from_centres(C)

    # allocate maps (faces live on cell-face grids)
    map_x = -np.ones((Nx+1, Ny,   Nz),   dtype=np.int64)  # i=1..Nx-1 internal
    map_y = -np.ones((Nx,   Ny+1, Nz),   dtype=np.int64)  # j=1..Ny-1 internal
    map_z = -np.ones((Nx,   Ny,   Nz+1), dtype=np.int64)  # k=1..Nz-1 internal

    cx = cy = cz = 0

    for f in range(nInt):
        co = owner[f]
        cn = neigh[f]

        i0, j0, k0 = ix[co], iy[co], iz[co]
        i1, j1, k1 = ix[cn], iy[cn], iz[cn]

        di = i1 - i0
        dj = j1 - j0
        dk = k1 - k0

        # structured mesh: exactly one of (di,dj,dk) is +/-1, others 0
        if di != 0 and dj == 0 and dk == 0:
            # x-normal face between i0 and i1 -> plane index is max(i0,i1)
            ii = max(i0, i1)
            jj = j0
            kk = k0
            if map_x[ii, jj, kk] != -1:
                raise RuntimeError(f"Duplicate in map_x at ({ii},{jj},{kk})")
            map_x[ii, jj, kk] = f
            cx += 1
        elif dj != 0 and di == 0 and dk == 0:
            jj = max(j0, j1)
            ii = i0
            kk = k0
            if map_y[ii, jj, kk] != -1:
                raise RuntimeError(f"Duplicate in map_y at ({ii},{jj},{kk})")
            map_y[ii, jj, kk] = f
            cy += 1
        elif dk != 0 and di == 0 and dj == 0:
            kk = max(k0, k1)
            ii = i0
            jj = j0
            if map_z[ii, jj, kk] != -1:
                raise RuntimeError(f"Duplicate in map_z at ({ii},{jj},{kk})")
            map_z[ii, jj, kk] = f
            cz += 1
        else:
            # if this triggers, mesh isn't purely structured or indexing isn't consistent
            raise RuntimeError(f"Non-structured adjacency at face {f}: (di,dj,dk)=({di},{dj},{dk})")

    np.savez_compressed(
        args.out,
        map_x=map_x, map_y=map_y, map_z=map_z,
        dims_cells=np.array([Nx,Ny,Nz], dtype=np.int32),
        counts=np.array([cx,cy,cz], dtype=np.int64),
    )

    print(f"[INFO] wrote {args.out}")
    print("[INFO] dims_cells (Nx,Ny,Nz) =", (Nx,Ny,Nz))
    print("[INFO] counts x/y/z         =", (cx,cy,cz))
    print("[INFO] nInternalFaces        =", nInt)

if __name__ == "__main__":
    main()



