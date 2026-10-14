#!/usr/bin/env python3
"""
assemble_global_xy_from_procAddressing.py  (DEBUG + cached addressing)

Purpose:
- Build a COMPACT global internal-face axis containing only global internal faces
  for which per-rank INTERNAL-face data exists (no processor-patch exports).
- Provide mappings:
    global_to_compact[f_global] = compact_id or -1 if missing
    compact_to_global[compact_id] = f_global

Additions in this version:
- DEBUG progress prints (flush=True) to identify where it "hangs"
- Cache per-rank Nint_r and gids (faceProcAddressing[:Nint_r]) in Pass1
  to avoid re-reading those big files in Pass2 (reduces FS pressure)

Input:
- --case TOP_CASE_DIR (contains constant/polyMesh and processor*/constant/polyMesh)
- --rank-dir directory with per-rank NPZ files: cfd_xy_rank####.npz
- --nproc number of ranks
- --out output NPZ path

Output NPZ:
- X: (T, N_compact, Fx)
- Y: (T, N_compact, Fy)
- kept_times: (T,) int32
- global_to_compact: (N_global_internal,) int64 (-1 for missing)
- compact_to_global: (N_compact,) int64
- Nf_global_internal: int64
- plus small metadata fields
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np


# -----------------------------
# OpenFOAM list parser (ASCII)
# -----------------------------
def parse_openfoam_list(path: Path) -> np.ndarray:
    """Parse OpenFOAM labelList/scalarList etc. Returns 1D int array."""
    # NOTE: This reads the whole file and regex-parses it. On large files / slow FS,
    # this can be the slowest part. We add debug around calls so we can locate hangs.
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


def dbg(msg: str) -> None:
    print(msg, flush=True)


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
    # Optional: reduce debug verbosity if needed
    ap.add_argument(
        "--dbg-every",
        type=int,
        default=1,
        help="Print DBG headers every N ranks (default: 1 prints every rank).",
    )
    args = ap.parse_args()

    if args.allow_different_times:
        args.require_same_times = False

    case = Path(args.case).resolve()
    rank_dir = Path(args.rank_dir).resolve()
    out_path = Path(args.out).resolve()

    dbg(f"[DBG] case={case}")
    dbg(f"[DBG] rank_dir={rank_dir}")
    dbg(f"[DBG] nproc={args.nproc}")
    dbg(f"[DBG] out={out_path}")

    # -----------------------------
    # Global internal faces count from TOP neighbour
    # -----------------------------
    dbg("[DBG] reading TOP constant/polyMesh/neighbour ...")
    t0 = time.time()
    neigh_top = parse_openfoam_list(case / "constant" / "polyMesh" / "neighbour")
    Nf_global = int(len(neigh_top))
    dbg(f"[INFO] Nf_global_internal = {Nf_global}  (dt={time.time()-t0:.2f}s)")

    # -----------------------------
    # Load rank0 to get dims and reference kept_times
    # -----------------------------
    dbg("[DBG] loading rank0 npz ...")
    t0 = time.time()
    d0 = load_rank_npz(rank_dir, 0)
    X0 = d0["X"]
    Y0 = d0["Y"]
    kept0 = d0.get("kept_times", None)
    if kept0 is None:
        raise RuntimeError("Rank0 NPZ missing 'kept_times'. Recreate per-rank NPZs with updated make_cfd_dataset.py.")
    T, N0, Fx = X0.shape
    Fy = Y0.shape[2]
    dbg(f"[INFO] rank0: T={T}, Nint_local={N0}, Fx={Fx}, Fy={Fy}, kept_times={len(kept0)}  (dt={time.time()-t0:.2f}s)")

    # -----------------------------
    # Fast path: serial/local case without processor0 decomposition
    # In a 1-rank run, rank0 face ordering already matches the top-case internal face axis.
    # -----------------------------
    proc0_poly = case / f"{args.processor_prefix}0" / "constant" / "polyMesh"
    if args.nproc == 1 and not proc0_poly.exists():
        dbg("[INFO] processor0 polyMesh not found for nproc=1; using rank0 NPZ as global internal ordering.")
        if N0 != Nf_global:
            raise RuntimeError(
                f"Serial fast-path requires rank0 NPZ N={N0} to match Nf_global_internal={Nf_global}."
            )

        global_to_compact = np.arange(Nf_global, dtype=np.int64)
        compact_to_global = np.arange(Nf_global, dtype=np.int64)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            X=np.nan_to_num(X0, nan=0.0, posinf=0.0, neginf=0.0),
            Y=np.nan_to_num(Y0, nan=0.0, posinf=0.0, neginf=0.0),
            kept_times=kept0.astype(np.int32, copy=False),
            Nf_global_internal=np.int64(Nf_global),
            global_to_compact=global_to_compact,
            compact_to_global=compact_to_global,
            nproc=np.int32(args.nproc),
            note=np.array(
                ["Serial fast-path: rank0 NPZ already uses the global internal-face ordering."],
                dtype=object,
            ),
        )
        dbg(f"[INFO] wrote {out_path} via serial fast-path")
        dbg(f"[INFO] X.shape={X0.shape}  Y.shape={Y0.shape}")
        return

    # -----------------------------
    # Pass 1: determine which global internal faces are available
    # Also cache per-rank addressing (Nint_r, gids[:Nint_r]) to reuse in Pass2.
    # -----------------------------
    provided = np.zeros(Nf_global, dtype=bool)

    # Cache: lists of arrays
    nint_list: list[int] = [0] * args.nproc
    gids_list: list[np.ndarray] = [np.empty((0,), dtype=np.int64) for _ in range(args.nproc)]

    dbg("[DBG] PASS1 start: scanning ranks, building provided[] and caching gids ...")

    for r in range(args.nproc):
        if args.dbg_every > 0 and (r % args.dbg_every == 0):
            dbg(f"[DBG] PASS1 r={r} start")

        tr0 = time.time()

        # 1) load npz
        dr = load_rank_npz(rank_dir, r)
        Xr = dr["X"]
        Yr = dr["Y"]
        kept = dr.get("kept_times", None)
        if kept is None:
            raise RuntimeError(f"Rank {r} NPZ missing 'kept_times'.")
        dbg(f"[DBG] PASS1 r={r} npz_loaded dt={time.time()-tr0:.2f}s")

        # 2) time alignment check
        if args.require_same_times and (kept.shape != kept0.shape or not np.array_equal(kept, kept0)):
            raise RuntimeError(
                f"Rank {r}: kept_times mismatch vs rank0. "
                "Align time steps in make_cfd_dataset.py or use --allow-different-times (not recommended)."
            )

        # 3) shape checks
        if Xr.shape[0] != T or Yr.shape[0] != T:
            raise RuntimeError(f"Rank {r}: T mismatch vs rank0. Got X.T={Xr.shape[0]} Y.T={Yr.shape[0]} expected {T}.")
        if Xr.shape[2] != Fx or Yr.shape[2] != Fy:
            raise RuntimeError(f"Rank {r}: feature dim mismatch vs rank0.")

        # 4) read processor neighbour to get Nint_r
        proc_poly = case / f"{args.processor_prefix}{r}" / "constant" / "polyMesh"

        dbg(f"[DBG] PASS1 r={r} reading neighbour ...")
        t1 = time.time()
        neigh_r = parse_openfoam_list(proc_poly / "neighbour")
        Nint_r = int(len(neigh_r))
        dbg(f"[DBG] PASS1 r={r} neighbour_loaded Nint_r={Nint_r} dt={time.time()-t1:.2f}s")

        if Xr.shape[1] != Nint_r or Yr.shape[1] != Nint_r:
            raise RuntimeError(
                f"Rank {r}: NPZ N={Xr.shape[1]} but processor neighbour length={Nint_r}. "
                "This assembler assumes per-rank NPZ contains INTERNAL faces only."
            )

        # 5) read faceProcAddressing (cache gids[:Nint_r])
        dbg(f"[DBG] PASS1 r={r} reading faceProcAddressing ...")
        t2 = time.time()
        addr = parse_openfoam_list(proc_poly / "faceProcAddressing")
        dbg(f"[DBG] PASS1 r={r} faceProcAddressing_loaded len={len(addr)} dt={time.time()-t2:.2f}s")

        if len(addr) < Nint_r:
            raise RuntimeError(f"Rank {r}: faceProcAddressing length {len(addr)} < Nint_r {Nint_r}.")

        gids = addr[:Nint_r].astype(np.int64, copy=False)

        # Cache
        nint_list[r] = Nint_r
        gids_list[r] = gids

        # Filter in-range BEFORE any indexing
        mask_in = (gids >= 0) & (gids < Nf_global)
        gids_in = gids[mask_in].astype(np.int64, copy=False)

        provided[gids_in] = True
        dbg(f"[INFO] rank {r:04d}: internal faces={Nint_r}, in-range global internal contributed={gids_in.size} (PASS1 dt={time.time()-tr0:.2f}s)")

    Navail = int(provided.sum())
    missing = int((~provided).sum())
    dbg(f"[INFO] Available global internal faces = {Navail}")
    dbg(f"[INFO] Missing  global internal faces = {missing}  (expected ~processor-boundary faces)")

    if Navail == 0:
        raise RuntimeError("No global internal faces were provided. Check addressing and per-rank NPZ coverage.")

    # -----------------------------
    # Build mappings
    # -----------------------------
    compact_to_global = np.nonzero(provided)[0].astype(np.int64)
    global_to_compact = np.full(Nf_global, -1, dtype=np.int64)
    global_to_compact[compact_to_global] = np.arange(Navail, dtype=np.int64)

    # -----------------------------
    # Allocate compact arrays
    # -----------------------------
    dbg("[DBG] allocating compact arrays ...")
    t0 = time.time()
    Xc = np.full((T, Navail, Fx), np.nan, dtype=X0.dtype)
    Yc = np.full((T, Navail, Fy), np.nan, dtype=Y0.dtype)

    # Track which compact face ids have been written (O(Navail) verification later)
    written = np.zeros(Navail, dtype=bool)

    dbg(f"[DBG] allocated Xc/Yc dt={time.time()-t0:.2f}s  Xc.shape={Xc.shape} Yc.shape={Yc.shape}")

    # -----------------------------
    # Pass 2: scatter into compact arrays (reuse cached gids)
    # -----------------------------
    dbg("[DBG] PASS2 start: scattering data into compact arrays ...")

    for r in range(args.nproc):
        if args.dbg_every > 0 and (r % args.dbg_every == 0):
            dbg(f"[DBG] PASS2 r={r} start")

        tr0 = time.time()

        dr = load_rank_npz(rank_dir, r)
        Xr = dr["X"]
        Yr = dr["Y"]
        dbg(f"[DBG] PASS2 r={r} npz_loaded dt={time.time()-tr0:.2f}s")

        Nint_r = nint_list[r]
        gids = gids_list[r]
        if gids.size != Nint_r:
            raise RuntimeError(f"Internal error: cached gids.size={gids.size} != cached Nint_r={Nint_r} for rank {r}")

        # Filter in-range first
        mask_in = (gids >= 0) & (gids < Nf_global)
        gids_in = gids[mask_in].astype(np.int64, copy=False)

        # Now safe to index global_to_compact
        cids_in = global_to_compact[gids_in]
        mask_have = cids_in >= 0
        cids = cids_in[mask_have].astype(np.int64, copy=False)

        # Build local mask corresponding to Xr[:, :, :]
        # mask_local is length Nint_r
        mask_local = mask_in.copy()
        tmp = np.zeros(mask_in.sum(), dtype=bool)
        tmp[mask_have] = True
        mask_local[mask_in] = tmp

        # Sanity: shapes must match
        selN = Xr[:, mask_local, :].shape[1]
        if selN != cids.shape[0]:
            raise RuntimeError(
                f"Rank {r}: local mask selected {selN} faces but cids has {cids.shape[0]} entries."
            )

        # Duplicate detection
        already = np.isfinite(Xc[0, cids, 0])
        if already.any():
            dup = int(already.sum())
            raise RuntimeError(f"Rank {r}: duplicate writes detected for {dup} compact faces (unexpected).")

        Xc[:, cids, :] = Xr[:, mask_local, :]
        Yc[:, cids, :] = Yr[:, mask_local, :]

        # Mark written compact faces (for cheap verification later)
        written[cids] = True

        dbg(f"[INFO] rank {r:04d}: scattered {cids.size} faces (PASS2 dt={time.time()-tr0:.2f}s)")

    # -----------------------------
    # Verify fully filled in compact axis
    # -----------------------------
    dbg("[DBG] verifying compact arrays filled (via written mask) ...")
    t0 = time.time()
    
    if not written.all():
        miss = int((~written).sum())
        sample = np.flatnonzero(~written)[:10]
        raise RuntimeError(
            f"Internal error: compact arrays still have {miss} unwritten faces. "
            f"Sample compact ids: {sample.tolist()}"
        )
    
    dbg(f"[DBG] verification OK (dt={time.time()-t0:.2f}s)")

    # Replace any remaining NaN/inf defensively
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    Yc = np.nan_to_num(Yc, nan=0.0, posinf=0.0, neginf=0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    dbg(f"[DBG] writing output npz: {out_path} ...")
    t0 = time.time()
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
    dbg(f"[INFO] wrote {out_path} (dt={time.time()-t0:.2f}s)")
    dbg(f"[INFO] X.shape={Xc.shape}  Y.shape={Yc.shape}")
    dbg(f"[INFO] global_to_compact: {global_to_compact.shape}  compact_to_global: {compact_to_global.shape}")


if __name__ == "__main__":
    main()


