from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import re

def parse_openfoam_list(path: Path) -> np.ndarray:
    txt = path.read_text(errors="ignore")
    m = re.search(r"^\s*(\d+)\s*$", txt, re.M)
    if not m:
        raise RuntimeError(f"Could not find list length in {path}")
    n = int(m.group(1))
    nums = list(map(int, re.findall(r"[-]?\d+", txt[m.end():])))
    if len(nums) < n:
        raise RuntimeError(f"Parsed only {len(nums)} ints but expected {n} from {path}")
    return np.asarray(nums[:n], dtype=np.int64)

def parse_openfoam_vector_field(path: Path) -> np.ndarray:
    txt = path.read_text(errors="ignore")
    m = re.search(r"^\s*(\d+)\s*$", txt, re.M)
    if not m:
        raise RuntimeError(f"Could not find field length in {path}")
    n = int(m.group(1))
    vecs = re.findall(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", txt[m.end():])
    if len(vecs) < n:
        raise RuntimeError(f"Parsed only {len(vecs)} vectors but expected {n} from {path}")
    arr = np.asarray(vecs[:n], dtype=np.float64)
    return arr

def build_index_from_centres(C: np.ndarray, tol_decimals: int = 12):
    x = np.round(C[:,0], tol_decimals)
    y = np.round(C[:,1], tol_decimals)
    z = np.round(C[:,2], tol_decimals)

    xu = np.unique(x); xu.sort()
    yu = np.unique(y); yu.sort()
    zu = np.unique(z); zu.sort()

    xmap = {v:i for i,v in enumerate(xu)}
    ymap = {v:i for i,v in enumerate(yu)}
    zmap = {v:i for i,v in enumerate(zu)}

    ix = np.fromiter((xmap[v] for v in x), dtype=np.int64, count=len(x))
    iy = np.fromiter((ymap[v] for v in y), dtype=np.int64, count=len(y))
    iz = np.fromiter((zmap[v] for v in z), dtype=np.int64, count=len(z))

    Nx, Ny, Nz = len(xu), len(yu), len(zu)
    return ix, iy, iz, (Nx, Ny, Nz)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="OpenFOAM case dir (TOP/global)")
    ap.add_argument("--npz", required=True, help="face-wise NPZ. Either (T,Nint,F) or compact with global_to_compact")
    ap.add_argument("--C", default=None, help="path to cell centres file (e.g. case/0/C). If omitted, tries case/0/C")
    ap.add_argument("--out", default="face_maps_internal.npz")
    args = ap.parse_args()

    case = Path(args.case)
    pm = case / "constant" / "polyMesh"
    owner = parse_openfoam_list(pm / "owner")
    neigh  = parse_openfoam_list(pm / "neighbour")
    nInt_global = len(neigh)

    npz = np.load(args.npz, allow_pickle=True)
    X = npz["X"]
    _, Nnpz, _ = X.shape

    # If compact mapping exists, use it. Otherwise require exact match.
    global_to_compact = None
    if "global_to_compact" in npz.files:
        global_to_compact = npz["global_to_compact"].astype(np.int64, copy=False)
        if global_to_compact.shape[0] != nInt_global:
            raise RuntimeError(
                f"global_to_compact length {global_to_compact.shape[0]} != nInternalFaces {nInt_global}"
            )
        Navail = int((global_to_compact >= 0).sum())
        if Navail != Nnpz:
            raise RuntimeError(
                f"NPZ N={Nnpz} but count(global_to_compact>=0)={Navail}. "
                "Use assemble_compact output NPZ."
            )
        print(f"[INFO] using compact mapping: Navailable={Navail}, NglobalInternal={nInt_global}")
    else:
        if Nnpz != nInt_global:
            raise RuntimeError(f"Dataset Ninternal={Nnpz} but neighbour length={nInt_global}. Check inputs.")
        print(f"[INFO] using full internal ordering: Ninternal={Nnpz}")

    # cell centres
    C_path = Path(args.C) if args.C else (case / "0" / "C")
    C = parse_openfoam_vector_field(C_path)

    ix, iy, iz, (Nx, Ny, Nz) = build_index_from_centres(C)

    map_x = -np.ones((Nx+1, Ny,   Nz),   dtype=np.int64)
    map_y = -np.ones((Nx,   Ny+1, Nz),   dtype=np.int64)
    map_z = -np.ones((Nx,   Ny,   Nz+1), dtype=np.int64)

    cx = cy = cz = 0

    # Loop over GLOBAL internal faces, place only those that exist in NPZ
    for f_global in range(nInt_global):
        if global_to_compact is None:
            f = f_global
        else:
            f = int(global_to_compact[f_global])
            if f < 0:
                continue  # missing (processor boundary face) -> keep -1

        co = owner[f_global]
        cn = neigh[f_global]

        i0, j0, k0 = ix[co], iy[co], iz[co]
        i1, j1, k1 = ix[cn], iy[cn], iz[cn]

        di = i1 - i0
        dj = j1 - j0
        dk = k1 - k0

        if di != 0 and dj == 0 and dk == 0:
            ii = max(i0, i1); jj = j0; kk = k0
            if map_x[ii, jj, kk] != -1:
                raise RuntimeError(f"Duplicate in map_x at ({ii},{jj},{kk})")
            map_x[ii, jj, kk] = f
            cx += 1
        elif dj != 0 and di == 0 and dk == 0:
            jj = max(j0, j1); ii = i0; kk = k0
            if map_y[ii, jj, kk] != -1:
                raise RuntimeError(f"Duplicate in map_y at ({ii},{jj},{kk})")
            map_y[ii, jj, kk] = f
            cy += 1
        elif dk != 0 and di == 0 and dj == 0:
            kk = max(k0, k1); ii = i0; jj = j0
            if map_z[ii, jj, kk] != -1:
                raise RuntimeError(f"Duplicate in map_z at ({ii},{jj},{kk})")
            map_z[ii, jj, kk] = f
            cz += 1
        else:
            raise RuntimeError(f"Non-structured adjacency at global face {f_global}: (di,dj,dk)=({di},{dj},{dk})")

    # --- add these before saving ---
    compact_to_global = None
    global_to_compact = None
    n_global_internal = None
    
    if "compact_to_global" in npz.files and "global_to_compact" in npz.files:
        compact_to_global = npz["compact_to_global"].astype(np.int64, copy=False)
        global_to_compact = npz["global_to_compact"].astype(np.int64, copy=False)
        n_global_internal = np.int64(global_to_compact.shape[0])
    
    save_kwargs = dict(
        map_x=map_x, map_y=map_y, map_z=map_z,
        dims_cells=np.array([Nx,Ny,Nz], dtype=np.int32),
        counts=np.array([cx,cy,cz], dtype=np.int64),
        available_faces=np.int64(np.count_nonzero(map_x >= 0) + np.count_nonzero(map_y >= 0) + np.count_nonzero(map_z >= 0)),
    )
    
    if compact_to_global is not None:
        save_kwargs.update(
            compact_to_global=compact_to_global,
            global_to_compact=global_to_compact,
            Nf_global_internal=n_global_internal,
        )
    
    np.savez_compressed(args.out, **save_kwargs)


    print(f"[INFO] wrote {args.out}")
    print("[INFO] dims_cells (Nx,Ny,Nz) =", (Nx,Ny,Nz))
    print("[INFO] counts x/y/z         =", (cx,cy,cz))
    print("[INFO] available faces       =", Nnpz)
    print("[INFO] global internal faces =", nInt_global)

if __name__ == "__main__":
    main()

