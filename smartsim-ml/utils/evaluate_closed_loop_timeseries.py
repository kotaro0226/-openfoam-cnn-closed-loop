#!/usr/bin/env python3
"""Evaluate closed-loop OpenFOAM CNN runs against an alpha=0 baseline."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openfoam_field_utils import (
    l2_relative,
    numeric_time_dirs,
    read_label_list,
    read_scalar_internal,
    read_vector_internal,
    time_name,
)


def parse_case_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--case must be label=/path/to/case")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute E_U, E_phi, and E_mean time series for CNN closed-loop OpenFOAM runs."
    )
    p.add_argument("--baseline", required=True, help="alpha=0 baseline case directory")
    p.add_argument("--case", action="append", default=[], help="label=/path/to/cnn_case; repeatable")
    p.add_argument("--out-dir", default="outputs/closed_loop_timeseries")
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--stop", type=float, default=None)
    p.add_argument("--continuity", action="store_true", help="also compute internal-face div(phi) proxy metrics")
    return p.parse_args()


def available_field_times(case_dir: Path, fields: tuple[str, ...]) -> dict[float, Path]:
    out = {}
    for t, d in numeric_time_dirs(case_dir):
        if all((d / field).exists() for field in fields):
            out[t] = d
    return out


def load_volumes(case_dir: Path) -> np.ndarray | None:
    candidates = [case_dir / "0" / "V", case_dir / "0.01" / "V"]
    for path in candidates:
        if path.exists():
            return read_scalar_internal(path)
    return None


def continuity_metrics(case_dir: Path, phi: np.ndarray, volumes: np.ndarray | None) -> dict[str, float]:
    owner = read_label_list(case_dir / "constant/polyMesh/owner")
    neighbour = read_label_list(case_dir / "constant/polyMesh/neighbour")
    n_internal = neighbour.size
    if phi.shape[0] < n_internal:
        raise RuntimeError(f"phi has {phi.shape[0]} values but mesh has {n_internal} internal faces")

    n_cells = int(max(owner[:n_internal].max(), neighbour.max())) + 1
    div = np.zeros(n_cells, dtype=np.float64)
    internal_phi = phi[:n_internal]
    np.add.at(div, owner[:n_internal], internal_phi)
    np.add.at(div, neighbour, -internal_phi)
    if volumes is not None:
        if volumes.shape[0] != div.shape[0]:
            raise RuntimeError(f"V shape {volumes.shape} does not match nCells={div.shape[0]}")
        div = div / np.maximum(volumes, 1e-300)
    return {
        "div_phi_internal_mean_abs": float(np.mean(np.abs(div))),
        "div_phi_internal_max_abs": float(np.max(np.abs(div))),
        "div_phi_internal_l2": float(np.linalg.norm(div)),
    }


def evaluate_case(
    label: str,
    case_dir: Path,
    baseline_dir: Path,
    out_dir: Path,
    start: float | None,
    stop: float | None,
    do_continuity: bool,
) -> list[dict]:
    fields = ("U", "phi")
    base_times = available_field_times(baseline_dir, fields)
    case_times = available_field_times(case_dir, fields)
    common = sorted(set(base_times) & set(case_times))
    if start is not None:
        common = [t for t in common if t >= start]
    if stop is not None:
        common = [t for t in common if t <= stop]
    if not common:
        raise RuntimeError(f"{label}: no common U/phi time directories with baseline")

    volumes = load_volumes(case_dir) if do_continuity else None
    rows: list[dict] = []
    for t in common:
        bdir = base_times[t]
        cdir = case_times[t]
        u_base = read_vector_internal(bdir / "U")
        u_case = read_vector_internal(cdir / "U")
        phi_base = read_scalar_internal(bdir / "phi")
        phi_case = read_scalar_internal(cdir / "phi")

        e_u = l2_relative(u_base, u_case)
        e_phi = l2_relative(phi_base, phi_case)
        row = {
            "alpha": label,
            "time": time_name(t),
            "E_U": e_u,
            "E_phi": e_phi,
            "E_mean": 0.5 * (e_u + e_phi),
            "E_U_percent": 100.0 * e_u,
            "E_phi_percent": 100.0 * e_phi,
            "E_mean_percent": 100.0 * 0.5 * (e_u + e_phi),
        }
        if do_continuity:
            row.update(continuity_metrics(case_dir, phi_case, volumes))
        rows.append(row)

    out_csv = out_dir / f"closed_loop_timeseries_{label}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote {out_csv}")
    return rows


def plot_metric(all_rows: dict[str, list[dict]], metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, rows in all_rows.items():
        t = [float(r["time"]) for r in rows]
        y = [float(r[metric]) for r in rows]
        ax.plot(t, y, marker="o", linewidth=1.8, markersize=4, label=f"alpha={label}")
    ax.set_xlabel("simulation time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] wrote {out_path}")


def main() -> None:
    args = parse_args()
    if not args.case:
        raise SystemExit("provide at least one --case label=/path")

    baseline = Path(args.baseline).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict]] = {}
    for spec in args.case:
        label, case_dir = parse_case_spec(spec)
        all_rows[label] = evaluate_case(
            label, case_dir, baseline, out_dir, args.start, args.stop, args.continuity
        )

    combined = [row for rows in all_rows.values() for row in rows]
    combined_csv = out_dir / "closed_loop_timeseries_all.csv"
    with combined_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"[INFO] wrote {combined_csv}")

    plot_metric(all_rows, "E_mean_percent", "Mean Relative L2 Error [%]", out_dir / "closed_loop_E_mean.png")
    plot_metric(all_rows, "E_U_percent", "Relative L2 Error of U [%]", out_dir / "closed_loop_E_U.png")
    plot_metric(all_rows, "E_phi_percent", "Relative L2 Error of phi [%]", out_dir / "closed_loop_E_phi.png")

    if args.continuity and "div_phi_internal_l2" in combined[0]:
        plot_metric(
            all_rows,
            "div_phi_internal_l2",
            "Internal-face div(phi) L2 proxy",
            out_dir / "closed_loop_div_phi_internal_l2.png",
        )


if __name__ == "__main__":
    main()
