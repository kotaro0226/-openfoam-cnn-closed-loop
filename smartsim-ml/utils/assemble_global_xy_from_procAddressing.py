#!/usr/bin/env python3
"""
assemble_global_xy_from_procAddressing.py  (COMPACT / internal-only version)

This is a drop-in replacement for the original "assemble_global_xy_from_procAddressing.py",
but adapted for the workflow where you DO NOT export processor-patch faces.

Problem this solves:
- Per-rank NPZs (from make_cfd_dataset.py) contain ONLY processor-local INTERNAL faces
  (i.e., faces counted by processor*/polyMesh/neighbour length).
- Global (top case) internal faces include faces that lie on processor boundaries.
  Those faces are NOT present in per-rank internal-only exports.
- Therefore, we cannot assemble a full global internal-face array without holes.
- Instead, we build a COMPACT global internal-face axis containing only the global faces
  for which we have data, and we provide mappings:
    global_to_compact[f_global] = compact_id or -1 if missing
    compact_to_global[compact_id] = f_global

Downstream impact (your intent):
- make_face_maps_from_C.py should use global_to_compact to place compact face ids onto maps
  and leave missing faces as -1.
- make_cell_maps.py should use compact_to_global to compute sign via owner/neighbour and
  output cell_faces_idx in compact ids (so subsample/train remain unchanged).
- Cells touching missing faces (processor boundary faces) get excluded by (faces6>=0).

Input:
- --case TOP_CASE_DIR (contains constant/polyMesh and processor*/constant/polyMesh)
- --rank-dir directory with per-rank NPZ files: cfd_xy_rank####.npz
- --nproc number of ranks
- --out output NPZ path (default keeps original script name behavior)

Output NPZ:
- X: (T, N_compact, Fx)
- Y: (T, N_compact, Fy)
- kept_times: (T,) int32
- global_to_compact: (N_global_internal,) int64 (-1 for missing)
- compact_to_global: (N_compact,) int64
- Nf_global_internal: int64
- plus small metadata fields

IMPORTANT invariants:
- We do NOT drop rows within each rank NPZ.
- We require aligned time steps across ranks (kept_times identical) by default.
  If you truly have staggered writes per rank, you need a more complex intersection logic.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


# -----------------------------
# OpenFOAM list parser (ASCII)
# -----------------------------
def parse_openfoam_list(path: Path) -> np.ndarray:
    """Parse OpenFOAM labelList/scalarList etc. Returns 1D int array."""
    txt = path.read_text(errors="ignore")
    m = re.search(r"^\s*(\d+)\s*$", txt, re.M)
    if not m:
        raise RuntimeError(f"Could not find list length in {path}")
    n = int(m.group(1))
    nums = list(map(int, re.findall(r"[-]?\d+", txt[m.end():])))
    if len(nums) < n:
        raise RuntimeError(f"Parsed only {len(nums)} ints but expected {n} from {path}")
    return np.asarray(nums[:n], dtype=np.int64)


def load_rank_npz(rank_dir: Path, r: int) -> dict:
    p = rank_dir / f"cfd_xy_rank{r:04d}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--case", required=True, help="Top OpenFOAM case directory")
    ap.add_argument("--rank-dir", required=True, help="Directory with per-rank NPZs (cfd_xy_rank####.npz)")
    ap.add_argument("--nproc", type=int, required=True, help="Number of MPI ranks (processors)")
    ap.add_argument("--processor-prefix", default="processor", help="Processor dir prefix (default: processor)")
    ap.add_argument("--out", default="cfd_xy_compact_internal_only.npz", help="Output NPZ path")
    ap.add_argument(
        "--require-same-times",
        action="store_true",
        default=True,
        help="Require kept_times identical across all ranks (recommended).",
    )
    ap.add_argument(
        "--allow-different-times",
        action="store_true",
        help="Disable strict kept_times check (NOT recommended unless you know what you're doing).",
    )
    args = ap.parse_args()

    if args.allow_different_times:
        args.require_same_times = False

    case = Path(args.case).resolve()
    rank_dir = Path(args.rank_dir).resolve()
    out_path = Path(args.out).resolve()

    # Global internal faces count from TOP neighbour
    neigh_top = parse_openfoam_list(case / "constant" / "polyMesh" / "neighbour")
    Nf_global = int(len(neigh_top))
    print(f"[INFO] Nf_global_internal = {Nf_global}")

    # Load rank0 to get T/F dims and reference kept_times
    d0 = load_rank_npz(rank_dir, 0)
    X0 = d0["X"]
    Y0 = d0["Y"]
    kept0 = d0.get("kept_times", None)
    if kept0 is None:
        raise RuntimeError("Rank0 NPZ missing 'kept_times'. Recreate per-rank NPZs with updated make_cfd_dataset.py.")
    T, N0, Fx = X0.shape
    Fy = Y0.shape[2]
    print(f"[INFO] rank0: T={T}, Nint_local={N0}, Fx={Fx}, Fy={Fy}, kept_times={len(kept0)}")

    # Pass 1: determine which global internal faces are available (internal-only contributions)
    provided = np.zeros(Nf_global, dtype=bool)

    for r in range(args.nproc):
        dr = load_rank_npz(rank_dir, r)
        Xr = dr["X"]
        Yr = dr["Y"]
        kept = dr.get("kept_times", None)
        if kept is None:
            raise RuntimeError(f"Rank {r} NPZ missing 'kept_times'.")

        if args.require_same_times and (kept.shape != kept0.shape or not np.array_equal(kept, kept0)):
            raise RuntimeError(
                f"Rank {r}: kept_times mismatch vs rank0. "
                "Align time steps in make_cfd_dataset.py or use --allow-different-times (not recommended)."
            )

        if Xr.shape[0] != T or Yr.shape[0] != T:
            raise RuntimeError(f"Rank {r}: T mismatch vs rank0. Got X.T={Xr.shape[0]} Y.T={Yr.shape[0]} expected {T}.")
        if Xr.shape[2] != Fx or Yr.shape[2] != Fy:
            raise RuntimeError(f"Rank {r}: feature dim mismatch vs rank0.")

        # Processor mesh paths
        proc_poly = case / f"{args.processor_prefix}{r}" / "constant" / "polyMesh"
        neigh_r = parse_openfoam_list(proc_poly / "neighbour")
        Nint_r = int(len(neigh_r))

        if Xr.shape[1] != Nint_r or Yr.shape[1] != Nint_r:
            raise RuntimeError(
                f"Rank {r}: NPZ N={Xr.shape[1]} but processor neighbour length={Nint_r}. "
                "This assembler assumes per-rank NPZ contains INTERNAL faces only."
            )

        addr = parse_openfoam_list(proc_poly / "faceProcAddressing")
        if len(addr) < Nint_r:
            raise RuntimeError(f"Rank {r}: faceProcAddressing length {len(addr)} < Nint_r {Nint_r}.")

        gids = addr[:Nint_r]

        # IMPORTANT: filter in-range BEFORE any indexing (avoid IndexError on sentinel values like Nf_global)
        mask_in = (gids >= 0) & (gids < Nf_global)
        gids_in = gids[mask_in].astype(np.int64, copy=False)

        provided[gids_in] = True
        print(f"[INFO] rank {r:04d}: internal faces={Nint_r}, in-range global internal contributed={gids_in.size}")

    Navail = int(provided.sum())
    missing = int((~provided).sum())
    print(f"[INFO] Available global internal faces = {Navail}")
    print(f"[INFO] Missing  global internal faces = {missing}  (expected ~processor-boundary faces)")

    if Navail == 0:
        raise RuntimeError("No global internal faces were provided. Check addressing and per-rank NPZ coverage.")

    # Build mappings
    compact_to_global = np.nonzero(provided)[0].astype(np.int64)
    global_to_compact = np.full(Nf_global, -1, dtype=np.int64)
    global_to_compact[compact_to_global] = np.arange(Navail, dtype=np.int64)

    # Allocate compact arrays
    Xc = np.full((T, Navail, Fx), np.nan, dtype=X0.dtype)
    Yc = np.full((T, Navail, Fy), np.nan, dtype=Y0.dtype)

    # Pass 2: scatter into compact arrays
    for r in range(args.nproc):
        dr = load_rank_npz(rank_dir, r)
        Xr = dr["X"]
        Yr = dr["Y"]

        proc_poly = case / f"{args.processor_prefix}{r}" / "constant" / "polyMesh"
        neigh_r = parse_openfoam_list(proc_poly / "neighbour")
        Nint_r = int(len(neigh_r))
        addr = parse_openfoam_list(proc_poly / "faceProcAddressing")
        gids = addr[:Nint_r]

        # Filter in-range first
        mask_in = (gids >= 0) & (gids < Nf_global)
        gids_in = gids[mask_in].astype(np.int64, copy=False)

        # Now safe to index global_to_compact
        cids_in = global_to_compact[gids_in]
        mask_have = cids_in >= 0
        gids_use = gids_in[mask_have]
        cids = cids_in[mask_have].astype(np.int64, copy=False)

        # Build local mask corresponding to Xr[:, :, :]
        mask_local = mask_in.copy()
        tmp = np.zeros(mask_in.sum(), dtype=bool)
        tmp[mask_have] = True
        mask_local[mask_in] = tmp

        # Sanity: shapes must match
        if Xr[:, mask_local, :].shape[1] != cids.shape[0]:
            raise RuntimeError(
                f"Rank {r}: local mask selected {Xr[:, mask_local, :].shape[1]} faces "
                f"but cids has {cids.shape[0]} entries."
            )

        # Duplicate detection (optional but helpful)
        already = np.isfinite(Xc[0, cids, 0])
        if already.any():
            # This should not happen for proper internal face mapping; but we guard anyway.
            dup = int(already.sum())
            raise RuntimeError(f"Rank {r}: duplicate writes detected for {dup} compact faces (unexpected).")

        Xc[:, cids, :] = Xr[:, mask_local, :]
        Yc[:, cids, :] = Yr[:, mask_local, :]
        print(f"[INFO] rank {r:04d}: scattered {cids.size} faces")

    # Verify fully filled in compact axis
    filled = np.isfinite(Xc[..., 0])
    if not filled.all():
        # Should not happen because compact axis built from provided
        miss = int((~filled).sum())
        raise RuntimeError(
            f"Internal error: compact arrays still have {miss} unfilled entries. "
            "Check addressing consistency."
        )

    # Replace any remaining NaN/inf defensively (should already be filled)
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    Yc = np.nan_to_num(Yc, nan=0.0, posinf=0.0, neginf=0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=Xc,
        Y=Yc,
        kept_times=kept0.astype(np.int32, copy=False),
        Nf_global_internal=np.int64(Nf_global),
        global_to_compact=global_to_compact,
        compact_to_global=compact_to_global,
        nproc=np.int32(args.nproc),
        note=np.array(
            ["COMPACT internal-face dataset: excludes global internal faces that lack data "
             "(typically processor-boundary faces). Missing faces map to -1 in global_to_compact."],
            dtype=object,
        ),
    )

    print(f"[INFO] wrote {out_path}")
    print(f"[INFO] X.shape={Xc.shape}  Y.shape={Yc.shape}")
    print(f"[INFO] global_to_compact: {global_to_compact.shape}  compact_to_global: {compact_to_global.shape}")


if __name__ == "__main__":
    main()
