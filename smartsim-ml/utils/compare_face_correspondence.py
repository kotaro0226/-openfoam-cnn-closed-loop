#!/usr/bin/env python3
"""
Compare solver-side CNN face-slot diagnostics with Python cell_maps.

The solver writes the diagnostic when:

    ML_WRITE_FACE_CORRESPONDENCE=1

Expected solver CSV:

    postProcessing/cnnFaceCorrespondence/<time>/face_correspondence.csv

The comparison checks, for each sampled cell and slot:

  - face id in global OpenFOAM internal-face indexing
  - owner/neighbour sign for that cell
  - face slot order xm,xp,ym,yp,zm,zp
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FACE_ORDER = ("xm", "xp", "ym", "yp", "zm", "zp")


def _load_solver_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in solver diagnostic CSV: {path}")
    required = {
        "cell_id",
        "i",
        "j",
        "k",
        "slot",
        "slot_name",
        "face_id",
        "cell_owner_neighbour_sign",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
    return rows


def _load_cell_maps(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    required = {"cell_ids", "cell_ijk", "cell_faces_idx", "cell_face_sign", "face_order"}
    missing = required.difference(z.files)
    if missing:
        raise RuntimeError(f"Missing keys in {path}: {sorted(missing)}. keys={z.files}")

    faces = np.asarray(z["cell_faces_idx"], dtype=np.int64)
    if "compact_to_global" in z.files:
        compact_to_global = np.asarray(z["compact_to_global"], dtype=np.int64)
        faces_global = compact_to_global[faces]
    else:
        faces_global = faces

    return {
        "cell_ids": np.asarray(z["cell_ids"], dtype=np.int64),
        "cell_ijk": np.asarray(z["cell_ijk"], dtype=np.int64),
        "cell_faces_global": np.asarray(faces_global, dtype=np.int64),
        "cell_face_sign": np.asarray(z["cell_face_sign"], dtype=np.int64),
        "face_order": np.asarray(z["face_order"]).astype(str),
    }


def compare(solver_csv: Path, cell_maps_npz: Path) -> tuple[int, list[str]]:
    rows = _load_solver_csv(solver_csv)
    maps = _load_cell_maps(cell_maps_npz)

    face_order = tuple(maps["face_order"].tolist())
    if face_order != FACE_ORDER:
        raise RuntimeError(f"Unexpected cell_maps face_order={face_order}, expected={FACE_ORDER}")

    row_by_cell = {int(cell_id): idx for idx, cell_id in enumerate(maps["cell_ids"])}
    errors: list[str] = []
    checked = 0

    for row in rows:
        cell_id = int(row["cell_id"])
        slot = int(row["slot"])
        slot_name = row["slot_name"]
        solver_face = int(row["face_id"])
        solver_sign = int(row["cell_owner_neighbour_sign"])
        solver_ijk = (int(row["i"]), int(row["j"]), int(row["k"]))

        if slot < 0 or slot >= len(FACE_ORDER):
            errors.append(f"cell={cell_id}: invalid slot={slot}")
            continue
        if slot_name != FACE_ORDER[slot]:
            errors.append(
                f"cell={cell_id} slot={slot}: slot_name={slot_name} expected={FACE_ORDER[slot]}"
            )
            continue
        if cell_id not in row_by_cell:
            errors.append(f"cell={cell_id}: missing from cell_maps cell_ids")
            continue

        idx = row_by_cell[cell_id]
        map_ijk = tuple(int(x) for x in maps["cell_ijk"][idx])
        map_face = int(maps["cell_faces_global"][idx, slot])
        map_sign = int(maps["cell_face_sign"][idx, slot])

        if solver_ijk != map_ijk:
            errors.append(f"cell={cell_id} slot={slot_name}: ijk solver={solver_ijk} map={map_ijk}")
        if solver_face != map_face:
            errors.append(
                f"cell={cell_id} slot={slot_name}: face solver={solver_face} map={map_face}"
            )
        if solver_sign != map_sign:
            errors.append(
                f"cell={cell_id} slot={slot_name}: sign solver={solver_sign} map={map_sign}"
            )

        checked += 1

    return checked, errors


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--solver-csv", required=True, type=Path)
    ap.add_argument("--cell-maps", required=True, type=Path)
    ap.add_argument("--max-errors", type=int, default=20)
    args = ap.parse_args()

    checked, errors = compare(args.solver_csv, args.cell_maps)
    print(f"[INFO] checked rows: {checked}")
    if errors:
        print(f"[ERROR] mismatches: {len(errors)}")
        for err in errors[: args.max_errors]:
            print(f"  - {err}")
        if len(errors) > args.max_errors:
            print(f"  ... {len(errors) - args.max_errors} more")
        raise SystemExit(1)

    print("[OK] solver face slots match cell_maps for all checked rows")


if __name__ == "__main__":
    main()
