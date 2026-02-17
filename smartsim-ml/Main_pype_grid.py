#!/usr/bin/env python3
"""
Main_pype_grid.py

Grid-search pipeline focused on:
- train.py (cell-wise MLP)
- eval_cell.py

Assumption:
- Subsampling step has already been executed by Main_pype.py.
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List

# Shared default root for input/output artifacts
DATAD = os.environ.get("DATAD", "./data")


@dataclass(frozen=True)
class TrialConfig:
    hidden: List[int]
    lr: float
    dropout: float
    lambda_cont: float

    @property
    def hidden_label(self) -> str:
        return "-".join(str(h) for h in self.hidden)

    @property
    def run_name(self) -> str:
        lr_tag = f"{self.lr:.3e}".replace("+", "")
        return (
            f"h{self.hidden_label}"
            f"_lr{lr_tag}"
            f"_do{self.dropout:g}"
            f"_lc{self.lambda_cont:g}"
        )


def parse_hidden_grid(spec: str) -> List[List[int]]:
    # e.g. "32,32;64,64;128,64,32"
    grids: List[List[int]] = []
    for block in (x.strip() for x in spec.split(";") if x.strip()):
        vals = [int(v.strip()) for v in block.split(",") if v.strip()]
        if not vals:
            continue
        grids.append(vals)
    if not grids:
        raise ValueError("--hidden-grid must include at least one setting")
    return grids


def parse_float_grid(spec: str) -> List[float]:
    vals = [float(v.strip()) for v in spec.split(",") if v.strip()]
    if not vals:
        raise ValueError("Grid spec is empty")
    return vals


def build_trials(args) -> List[TrialConfig]:
    hidden_grid = parse_hidden_grid(args.hidden_grid)
    lr_grid = parse_float_grid(args.lr_grid)
    dropout_grid = parse_float_grid(args.dropout_grid)
    lambda_grid = parse_float_grid(args.lambda_cont_grid)

    trials: List[TrialConfig] = []
    for hidden, lr, dropout, lambda_cont in itertools.product(
        hidden_grid, lr_grid, dropout_grid, lambda_grid
    ):
        trials.append(
            TrialConfig(
                hidden=list(hidden),
                lr=lr,
                dropout=dropout,
                lambda_cont=lambda_cont,
            )
        )
    return trials


def run_cmd(cmd: Iterable[str], cwd: str, log_path: str = ""):
    cmd = list(cmd)
    print("[RUN]", " ".join(cmd))

    if not log_path:
        subprocess.run(cmd, check=True, cwd=cwd)
        return

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def read_single_metrics_row(metrics_csv_path: str) -> dict:
    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(f"metrics CSV not found: {metrics_csv_path}")

    with open(metrics_csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"metrics CSV has no rows: {metrics_csv_path}")
    return rows[-1]


def append_summary_row(summary_csv: str, row: dict):
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    write_header = (not os.path.exists(summary_csv)) or (os.path.getsize(summary_csv) == 0)
    with open(summary_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(description="Grid-search for cell-wise MLP training/evaluation")

    # Inputs from completed subsampling stage
    p.add_argument(
        "--cell-npz",
        default=f"{DATAD}/sampled/subsampled_ofCell_random_random.npz",
        help="subsampled_ofCell_*.npz (default: existing Main_pype output under DATAD)",
    )

    # Shared split across all trials
    p.add_argument(
        "--shared-split-npz",
        default=f"{DATAD}/grid/splits/split_shared_seed1234.npz",
        help="split file shared across all trials",
    )

    # Grid definitions
    p.add_argument("--hidden-grid", default="32,32;64,64", help="e.g. '32,32;64,64;128,64,32'")
    p.add_argument("--lr-grid", default="1e-3,5e-4")
    p.add_argument("--dropout-grid", default="0.0,0.05")
    p.add_argument("--lambda-cont-grid", default="0.0,1.0")

    # Common train/eval options
    p.add_argument("--act", default="tanh")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--split", default="0.8,0.1,0.1")
    p.add_argument("--which", choices=["train", "val", "test"], default="test")
    p.add_argument("--fields-per-face", type=int, default=3)
    p.add_argument("--phi-in-face", type=int, default=1)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    # Outputs
    p.add_argument("--outdir", default=f"{DATAD}/grid")
    p.add_argument(
        "--summary-csv",
        default="",
        help="summary csv path (default: <outdir>/results/grid_metrics.csv)",
    )
    p.add_argument("--save-pred", action="store_true", help="save predictions npz per trial")
    p.add_argument("--run-vis", action="store_true", help="run utils/vis-plot.py at the end")

    return p.parse_args()


def main():
    args = parse_args()
    if "DATAD" not in os.environ:
        print("[WARN] DATAD is not set. Falling back to relative ./data defaults.")
    repo_dir = os.path.dirname(os.path.abspath(__file__))

    summary_csv = args.summary_csv or os.path.join(args.outdir, "results", "grid_metrics.csv")
    trials = build_trials(args)
    print(f"[INFO] Number of trials: {len(trials)}")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.shared_split_npz) or ".", exist_ok=True)

    split_initialized = os.path.exists(args.shared_split_npz)

    for i, trial in enumerate(trials, start=1):
        run_dir = os.path.join(args.outdir, "runs", trial.run_name)
        hidden_tag = trial.hidden_label
        jit_path = os.path.join(run_dir, f"model_h{hidden_tag}.jit")
        norm_path = os.path.join(run_dir, f"norm_h{hidden_tag}.npz")
        eval_metrics_csv = os.path.join(run_dir, f"eval_metrics_h{hidden_tag}.csv")
        pred_npz = os.path.join(run_dir, f"pred_h{hidden_tag}.npz")
        train_log = os.path.join(run_dir, "train.log")

        os.makedirs(run_dir, exist_ok=True)

        print("=" * 80)
        print(f"[INFO] Trial {i}/{len(trials)}: {trial.run_name}")

        train_cmd = [
            sys.executable,
            "-u",
            "train.py",
            "mlp",
            "--src",
            "sickle",
            "--cell_npz",
            args.cell_npz,
            "--epochs",
            str(args.epochs),
            "--lr",
            str(trial.lr),
            "--batch",
            str(args.batch),
            "--lambda-cont",
            str(trial.lambda_cont),
            "--phi-in-face",
            str(args.phi_in_face),
            "--fields-per-face",
            str(args.fields_per_face),
            "--hidden",
            *[str(h) for h in trial.hidden],
            "--act",
            args.act,
            "--dropout",
            str(trial.dropout),
            "--split",
            args.split,
            "--seed",
            str(args.seed),
            "--norm-path",
            norm_path,
            "--fit-norm",
            "--save-jit",
            jit_path,
        ]

        if split_initialized:
            train_cmd += ["--split-load", args.shared_split_npz]
        else:
            train_cmd += ["--split-save", args.shared_split_npz]

        run_cmd(train_cmd, cwd=repo_dir, log_path=train_log)
        split_initialized = True

        eval_cmd = [
            sys.executable,
            "-u",
            "eval_cell.py",
            "--jit",
            jit_path,
            "--cell_npz",
            args.cell_npz,
            "--split_npz",
            args.shared_split_npz,
            "--which",
            args.which,
            "--norm_path",
            norm_path,
            "--batch_cells",
            str(args.batch),
            "--fields_per_face",
            str(args.fields_per_face),
            "--phi_in_face",
            str(args.phi_in_face),
            "--device",
            args.device,
            "--metrics_csv",
            eval_metrics_csv,
        ]
        if args.save_pred:
            eval_cmd += ["--save_pred_npz", pred_npz]

        run_cmd(eval_cmd, cwd=repo_dir)

        m = read_single_metrics_row(eval_metrics_csv)
        summary_row = {
            "run_name": trial.run_name,
            "hidden": trial.hidden_label,
            "lr": trial.lr,
            "dropout": trial.dropout,
            "lambda_cont": trial.lambda_cont,
            "epochs": args.epochs,
            "batch": args.batch,
            "split": args.which,
            "jit": m.get("jit", ""),
            "Nsel": m.get("Nsel", ""),
            "mse": m.get("mse", ""),
            "rmse": m.get("rmse", ""),
            "mae": m.get("mae", ""),
            "r2": m.get("r2", ""),
            "rel_rmse": m.get("rel_rmse", ""),
            "cont_mean": m.get("cont_mean", ""),
            "norm_path": m.get("norm_path", ""),
            "run_dir": run_dir,
        }
        append_summary_row(summary_csv, summary_row)
        print(f"[INFO] appended summary to: {summary_csv}")

    if args.run_vis:
        vis_cmd = [
            sys.executable,
            "-u",
            os.path.join("utils", "vis-plot.py"),
            "--csv",
            summary_csv,
            "--outdir",
            os.path.join(args.outdir, "figures"),
            "--split",
            args.which,
            "--agg",
            "min",
        ]
        run_cmd(vis_cmd, cwd=repo_dir)

    print("[INFO] Grid-search done.")
    print(f"[INFO] Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()

