#!/usr/bin/env python3
"""Compare native/CNN/used pressure-flux corrections on the same state.

The solver writes the final corrected phi and pEqnFlux_used.  Since

    phi = phiHbyA - pEqnFlux_used

we reconstruct phiHbyA as phi + pEqnFlux_used, then form alternative fluxes:

    phi_native  = phiHbyA - pEqnFlux_native
    phi_cnn_pre = phiHbyA - pEqnFlux_cnn_masked
    phi_used    = phiHbyA - pEqnFlux_used

The CNN alternative uses pEqnFlux_cnn only on faces selected by
pEqnFlux_cnn_mask and native flux elsewhere, matching the solver's masked
replacement behavior.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from openfoam_field_utils import (
    l2_relative,
    numeric_time_dirs,
    read_label_list,
    read_scalar_internal,
    time_name,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, help="OpenFOAM case with pEqnFlux_native/cnn/used fields")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--stop", type=float, default=None)
    return p.parse_args()


def load_volumes(case_dir: Path) -> np.ndarray | None:
    for name in ("0", "1", "0.01"):
        path = case_dir / name / "V"
        if path.exists():
            return read_scalar_internal(path)
    return None


def divergence(case_dir: Path, phi: np.ndarray, volumes: np.ndarray | None) -> np.ndarray:
    owner_all = read_label_list(case_dir / "constant/polyMesh/owner")
    neighbour = read_label_list(case_dir / "constant/polyMesh/neighbour")
    n_internal = neighbour.size
    owner = owner_all[:n_internal]
    if phi.shape[0] < n_internal:
        raise RuntimeError(f"phi has {phi.shape[0]} values but mesh has {n_internal} internal faces")

    n_cells = int(max(owner.max(), neighbour.max())) + 1
    div = np.zeros(n_cells, dtype=np.float64)
    internal_phi = phi[:n_internal]
    np.add.at(div, owner, internal_phi)
    np.add.at(div, neighbour, -internal_phi)
    if volumes is not None:
        if volumes.shape[0] != div.shape[0]:
            raise RuntimeError(f"V shape {volumes.shape} does not match nCells={div.shape[0]}")
        div = div / np.maximum(volumes, 1e-300)
    return div


def div_stats(div: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_div_mean_abs": float(np.mean(np.abs(div))),
        f"{prefix}_div_max_abs": float(np.max(np.abs(div))),
        f"{prefix}_div_l2": float(np.linalg.norm(div)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote {path}")


def plot_rows(rows: list[dict], keys: list[tuple[str, str]], out_path: Path, ylabel: str, logy: bool = False) -> None:
    if plt is None:
        print(f"[WARN] matplotlib is not available; skipped {out_path}")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    t = [float(r["time"]) for r in rows]
    for key, label in keys:
        ax.plot(t, [float(r[key]) for r in rows], marker="o", linewidth=1.8, markersize=4, label=label)
    ax.set_xlabel("simulation time")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] wrote {out_path}")


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    volumes = load_volumes(case_dir)

    rows: list[dict] = []
    for t, tdir in numeric_time_dirs(case_dir):
        if args.start is not None and t < args.start:
            continue
        if args.stop is not None and t > args.stop:
            continue

        required = ["phi", "pEqnFlux_native", "pEqnFlux_cnn", "pEqnFlux_used"]
        if not all((tdir / name).exists() for name in required):
            continue

        phi_used_saved = read_scalar_internal(tdir / "phi")
        native = read_scalar_internal(tdir / "pEqnFlux_native")
        cnn = read_scalar_internal(tdir / "pEqnFlux_cnn")
        used = read_scalar_internal(tdir / "pEqnFlux_used")
        if not (native.shape == cnn.shape == used.shape == phi_used_saved.shape):
            raise RuntimeError(f"{tdir}: field shape mismatch")

        mask_path = tdir / "pEqnFlux_cnn_mask"
        if mask_path.exists():
            mask = read_scalar_internal(mask_path) > 0.5
        else:
            mask = np.ones_like(native, dtype=bool)

        cnn_masked = native.copy()
        cnn_masked[mask] = cnn[mask]

        phi_hbya = phi_used_saved + used
        phi_native = phi_hbya - native
        phi_cnn = phi_hbya - cnn_masked
        phi_used = phi_hbya - used

        div_hbya = divergence(case_dir, phi_hbya, volumes)
        div_native = divergence(case_dir, phi_native, volumes)
        div_cnn = divergence(case_dir, phi_cnn, volumes)
        div_used = divergence(case_dir, phi_used, volumes)

        active = mask
        row = {
            "time": time_name(t),
            "active_faces": int(np.count_nonzero(active)),
            "all_faces": int(active.size),
            "pEqnFlux_cnn_minus_native_rel_l2_active": l2_relative(native[active], cnn[active]),
            "pEqnFlux_used_minus_native_rel_l2_all": l2_relative(native, used),
            "phi_cnn_minus_native_rel_l2_all": l2_relative(phi_native, phi_cnn),
            "phi_used_minus_native_rel_l2_all": l2_relative(phi_native, phi_used),
            "phi_used_saved_reconstruction_rel_l2": l2_relative(phi_used_saved, phi_used),
        }
        row.update(div_stats(div_native, "native"))
        row.update(div_stats(div_cnn, "cnn_masked"))
        row.update(div_stats(div_used, "used"))
        row.update(div_stats(div_hbya, "hbya"))
        row["cnn_div_l2_over_native"] = row["cnn_masked_div_l2"] / max(row["native_div_l2"], 1e-300)
        row["used_div_l2_over_native"] = row["used_div_l2"] / max(row["native_div_l2"], 1e-300)
        row["cnn_div_l2_over_hbya"] = row["cnn_masked_div_l2"] / max(row["hbya_div_l2"], 1e-300)
        row["native_div_l2_over_hbya"] = row["native_div_l2"] / max(row["hbya_div_l2"], 1e-300)
        row["cnn_reduces_hbya_div_l2"] = int(row["cnn_masked_div_l2"] < row["hbya_div_l2"])
        row["cnn_between_hbya_and_native_div_l2"] = int(
            row["native_div_l2"] < row["cnn_masked_div_l2"] < row["hbya_div_l2"]
        )
        rows.append(row)

    if not rows:
        raise SystemExit("no complete phi/pEqnFlux_native/cnn/used time directories found")

    write_csv(out_dir / "flux_divergence_summary.csv", rows)
    plot_rows(
        rows,
        [
            ("hbya_div_l2", "uncorrected phiHbyA"),
            ("native_div_l2", "native"),
            ("cnn_masked_div_l2", "cnn pre-corrected"),
            ("used_div_l2", "used"),
        ],
        out_dir / "flux_divergence_l2.png",
        "Internal-face div(phi) L2 proxy",
        logy=True,
    )
    plot_rows(
        rows,
        [
            ("pEqnFlux_cnn_minus_native_rel_l2_active", "pEqnFlux cnn-native"),
            ("phi_cnn_minus_native_rel_l2_all", "phi cnn-native"),
        ],
        out_dir / "flux_difference_rel_l2.png",
        "Relative L2 difference",
    )


if __name__ == "__main__":
    main()
