#!/usr/bin/env python3
"""Unified grid search for cell-wise MLP training + evaluation.

DDP support:
- Launch with srun on Frontier/Slurm.
- Heavy split/norm preparation is done BEFORE init_process_group().
- Training is distributed with DistributedSampler + DDP.
- Final evaluation/CSV/JIT writing is done on rank0.
"""

import argparse
import csv
import itertools
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from train import _build_model, _fit_norm_on_train, _parse_split, sickle_xy_cell
from utils.losses import continuity_loss_cell6
from utils.normalization import apply_zscore, invert_zscore, load_norm, save_norm
from utils.split import load_split, make_split_indices, save_split

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
    grids: List[List[int]] = []
    for block in (x.strip() for x in spec.split(";") if x.strip()):
        vals = [int(v.strip()) for v in block.split(",") if v.strip()]
        if vals:
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
    trials: List[TrialConfig] = []
    for hidden, lr, dropout, lambda_cont in itertools.product(
        parse_hidden_grid(args.hidden_grid),
        parse_float_grid(args.lr_grid),
        parse_float_grid(args.dropout_grid),
        parse_float_grid(args.lambda_cont_grid),
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


def compute_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    err = pred - y
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    y_mean = float(np.mean(y))
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    y_std = float(np.std(y))
    rel_rmse = float(rmse / y_std) if y_std > 0 else float("nan")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "rel_rmse": rel_rmse}


def append_summary_row(summary_csv: str, row: dict):
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    write_header = (not os.path.exists(summary_csv)) or (os.path.getsize(summary_csv) == 0)
    with open(summary_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def resolve_dist_env(args):
    """Resolve rank/world/local_rank without initializing process group yet."""
    rank_env = (
        os.environ.get("RANK")
        or os.environ.get("PMI_RANK")
        or os.environ.get("SLURM_PROCID")
    )
    world_size_env = (
        os.environ.get("WORLD_SIZE")
        or os.environ.get("PMI_SIZE")
        or os.environ.get("SLURM_NTASKS")
    )

    use_ddp = (
        (not args.no_ddp)
        and rank_env is not None
        and world_size_env is not None
        and int(world_size_env) > 1
    )

    if not use_ddp:
        return False, 0, 1, 0

    rank = int(rank_env)
    world_size = int(world_size_env)

    backend = args.ddp_backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    if backend == "nccl":
        n_visible = torch.cuda.device_count()
        if n_visible < 1:
            raise RuntimeError("NCCL backend requested but no visible GPU was found.")
        # Frontier + srun --gpus-per-task=1: each task typically sees exactly one GPU -> cuda:0
        local_rank = 0 if n_visible == 1 else (rank % n_visible)
    else:
        local_rank = 0

    return True, rank, world_size, local_rank


def init_dist(args, rank: int, world_size: int, local_rank: int):
    """Initialize torch.distributed after heavy preprocessing is done."""
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    if not master_addr or not master_port:
        raise RuntimeError("MASTER_ADDR / MASTER_PORT are required for DDP.")

    backend = args.ddp_backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if backend == "nccl":
        torch.cuda.set_device(local_rank)

    print(
        f"[DDP INIT] host={os.uname().nodename} "
        f"rank={rank} world_size={world_size} local_rank={local_rank} "
        f"master={master_addr}:{master_port}",
        flush=True,
    )

    kwargs = dict(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=2),
    )

    if backend == "nccl":
        kwargs["device_id"] = torch.device("cuda", local_rank)

    dist.init_process_group(**kwargs)


def barrier(is_dist: bool, local_rank: int = 0):
    if not is_dist:
        return
    if dist.is_initialized() and torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    elif dist.is_initialized():
        dist.barrier()


def cleanup_dist(is_dist: bool, local_rank: int = 0):
    if is_dist and dist.is_initialized():
        barrier(is_dist, local_rank)
        dist.destroy_process_group()


def wait_for_file(path: str, sleep_sec: float = 2.0, timeout_sec: float = 7200.0):
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"Timed out waiting for file: {path}")
        time.sleep(sleep_sec)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in {"cuda", "rocm"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce_mean(value: float, device: torch.device, is_dist: bool) -> float:
    if not is_dist:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def eval_model_rank0(args, model, loader, x_mean, x_std, y_mean, y_std, device, pin_memory):
    model.eval()
    pred_list, y_list = [], []
    cont_sum = 0.0
    cont_cnt = 0

    with torch.no_grad():
        for Xc, Y6, S6, V in loader:
            Xc = Xc.to(device, non_blocking=pin_memory)
            Y6 = Y6.to(device, non_blocking=pin_memory)
            S6 = S6.to(device, non_blocking=pin_memory)
            V = V.to(device, non_blocking=pin_memory)

            bsz = Xc.shape[0]
            Xfaces = Xc.view(bsz, args.num_faces_per_cell, args.fields_per_face).reshape(
                bsz * args.num_faces_per_cell, args.fields_per_face
            )
            pred_n = model(apply_zscore(Xfaces, x_mean, x_std)).reshape(bsz * args.num_faces_per_cell, 1)
            pred_phys = invert_zscore(pred_n, y_mean, y_std).reshape(bsz, args.num_faces_per_cell)

            c = continuity_loss_cell6(
                pEqnFlux_pred6=pred_phys,
                X_cell=Xc,
                cell_face_sign=S6,
                V=V,
                fields_per_face=args.fields_per_face,
                phi_in_face=args.phi_in_face,
            )
            c_val = float(c.detach().cpu().item())
            cont_sum += c_val * bsz
            cont_cnt += bsz

            pred_list.append(pred_phys.detach().cpu().numpy().astype(np.float32))
            y_list.append(Y6.detach().cpu().numpy().astype(np.float32))

    pred = np.concatenate(pred_list, axis=0)
    ygt = np.concatenate(y_list, axis=0)
    metrics = compute_metrics(pred, ygt)
    metrics["cont_mean"] = float(cont_sum / max(cont_cnt, 1))
    return metrics, pred, ygt


def parse_args():
    p = argparse.ArgumentParser(description="Unified grid-search for cell-wise MLP (DDP-ready)")
    p.add_argument("--cell-npz", default=f"{DATAD}/sampled/subsampled_ofCell_random_random.npz")
    p.add_argument("--shared-split-npz", default=f"{DATAD}/grid/splits/split_shared_seed1234.npz")
    p.add_argument("--shared-norm-npz", default="", help="optional shared norm path (default: <outdir>/shared_norm.npz)")

    p.add_argument("--hidden-grid", default="32,32;64,64")
    p.add_argument("--lr-grid", default="1e-3,5e-4")
    p.add_argument("--dropout-grid", default="0.0,0.05")
    p.add_argument("--lambda-cont-grid", default="0.0,1.0")

    p.add_argument("--arch", default="mlp")
    p.add_argument("--act", default="tanh")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--split", default="0.8,0.1,0.1")
    p.add_argument("--which", choices=["train", "val", "test"], default="test")

    p.add_argument("--fields-per-face", type=int, default=3)
    p.add_argument("--phi-in-face", type=int, default=1)
    p.add_argument("--num-faces-per-cell", type=int, default=6)

    p.add_argument("--device", choices=["auto", "cuda", "rocm", "cpu"], default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")

    p.add_argument("--ddp-backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--no-ddp", action="store_true", help="disable DDP even if distributed env exists")

    p.add_argument("--outdir", default=f"{DATAD}/grid")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--save-pred", action="store_true")
    p.add_argument("--jit-mode", choices=["trace", "script"], default="trace")
    p.add_argument("--run-vis", action="store_true", help="run utils/vis-plot.py at the end (rank0 only)")
    return p.parse_args()


def main():
    args = parse_args()
    is_dist, rank, world_size, local_rank = resolve_dist_env(args)

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_dir)

    trials = build_trials(args)
    summary_csv = args.summary_csv or os.path.join(args.outdir, "results", "grid_metrics.csv")
    shared_norm_npz = args.shared_norm_npz or os.path.join(args.outdir, "shared_norm.npz")

    if rank == 0:
        print(f"[INFO] Number of trials: {len(trials)}")
        print(f"[INFO] DDP requested={is_dist} world_size={world_size}")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.shared_split_npz) or ".", exist_ok=True)

    device = resolve_device(args.device)
    if device.type == "cuda":
        visible_count = torch.cuda.device_count()
        if visible_count < 1:
            raise RuntimeError("CUDA/HIP device was requested, but no visible GPU was found.")
        device = torch.device("cuda", local_rank)
        print(
            f"[DEVICE] rank={rank} local_rank={local_rank} "
            f"visible_cuda_count={visible_count} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')} "
            f"device={device}",
            flush=True,
        )

    pin_memory = bool(args.pin_memory) and device.type == "cuda"

    # ------------------------------------------------------------------
    # Heavy preprocessing BEFORE DDP init
    # ------------------------------------------------------------------
    dataset, input_shape, output_shape = sickle_xy_cell(args.cell_npz)
    n_total = len(dataset)
    print(f"[INFO] rank={rank} Loaded dataset once-per-rank: Ncells={n_total}", flush=True)

    tr, va, te = _parse_split(args.split)

    if rank == 0:
        if os.path.exists(args.shared_split_npz):
            train_idx, val_idx, test_idx = load_split(args.shared_split_npz)
            print(f"[INFO] loaded split: {args.shared_split_npz}", flush=True)
        else:
            train_idx, val_idx, test_idx = make_split_indices(n_total, tr, va, te, args.seed)
            save_split(
                args.shared_split_npz,
                train_idx,
                val_idx,
                test_idx,
                meta={"N": n_total, "split": args.split, "seed": args.seed, "is_cell": 1},
            )
            print(f"[INFO] saved split: {args.shared_split_npz}", flush=True)
    else:
        wait_for_file(args.shared_split_npz)
        train_idx, val_idx, test_idx = load_split(args.shared_split_npz)
        print(f"[INFO] loaded split: {args.shared_split_npz}", flush=True)

    ds_train = Subset(dataset, train_idx)
    ds_val = Subset(dataset, val_idx)
    ds_test = Subset(dataset, test_idx)
    split_map = {"train": ds_train, "val": ds_val, "test": ds_test}
    ds_eval = split_map[args.which]

    if rank == 0:
        x_mean, x_std, y_mean, y_std = _fit_norm_on_train(
            args,
            True,
            ds_train,
            input_shape,
            output_shape,
            str(device),
        )
        save_norm(
            shared_norm_npz,
            x_mean,
            x_std,
            y_mean,
            y_std,
            meta={"fit_on": "train", "split": args.split, "seed": args.seed, "is_cell": 1},
        )
        print(f"[INFO] fitted shared normalization once: {shared_norm_npz}", flush=True)
    else:
        wait_for_file(shared_norm_npz)

    x_mean, x_std, y_mean, y_std = load_norm(shared_norm_npz, device=str(device))

    # ------------------------------------------------------------------
    # Initialize DDP only after split/norm are ready
    # ------------------------------------------------------------------
    if is_dist:
        init_dist(args, rank, world_size, local_rank)
        if rank == 0:
            print(f"[INFO] DDP enabled=True world_size={world_size}", flush=True)

    loader_val = None
    loader_eval_rank0 = None
    train_sampler = None
    val_sampler = None

    if is_dist:
        train_sampler = DistributedSampler(
            ds_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            ds_val,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    loader_val = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    if rank == 0:
        loader_eval_rank0 = DataLoader(
            ds_eval,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    for idx, trial in enumerate(trials, start=1):
        run_dir = os.path.join(args.outdir, "runs", trial.run_name)
        hidden_tag = trial.hidden_label
        jit_path = os.path.join(run_dir, f"model_h{hidden_tag}.jit")
        norm_path = os.path.join(run_dir, f"norm_h{hidden_tag}.npz")
        eval_metrics_csv = os.path.join(run_dir, f"eval_metrics_h{hidden_tag}.csv")
        pred_npz = os.path.join(run_dir, f"pred_h{hidden_tag}.npz")

        if rank == 0:
            os.makedirs(run_dir, exist_ok=True)
            print("=" * 80)
            print(f"[INFO] Trial {idx}/{len(trials)}: {trial.run_name}")
        barrier(is_dist, local_rank)

        run_args = argparse.Namespace(**vars(args))
        run_args.hidden = trial.hidden
        run_args.dropout = trial.dropout
        base_model = _build_model(run_args, (args.fields_per_face,)).to(device)

        if is_dist:
            if device.type == "cuda":
                model = DDP(base_model, device_ids=[local_rank], output_device=local_rank)
            else:
                model = DDP(base_model)
        else:
            model = base_model

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=trial.lr)

        loader_train = DataLoader(
            ds_train,
            batch_size=args.batch,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        for ep in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(ep)

            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for Xc, Y6, S6, V in loader_train:
                Xc = Xc.to(device, non_blocking=pin_memory)
                Y6 = Y6.to(device, non_blocking=pin_memory)
                S6 = S6.to(device, non_blocking=pin_memory)
                V = V.to(device, non_blocking=pin_memory)
                bsz = Xc.shape[0]

                Xfaces = Xc.view(bsz, args.num_faces_per_cell, args.fields_per_face).reshape(
                    bsz * args.num_faces_per_cell,
                    args.fields_per_face,
                )
                Xn = apply_zscore(Xfaces, x_mean, x_std)
                pred_n = model(Xn).view(bsz, args.num_faces_per_cell)
                Yn = apply_zscore(Y6.reshape(bsz * args.num_faces_per_cell, 1), y_mean, y_std).view(
                    bsz,
                    args.num_faces_per_cell,
                )

                loss_data = criterion(pred_n, Yn)
                loss_cont = torch.tensor(0.0, device=device)
                if trial.lambda_cont > 0.0:
                    pred_phys = invert_zscore(
                        pred_n.reshape(bsz * args.num_faces_per_cell, 1),
                        y_mean,
                        y_std,
                    ).view(bsz, args.num_faces_per_cell)
                    loss_cont = continuity_loss_cell6(
                        pEqnFlux_pred6=pred_phys,
                        X_cell=Xc,
                        cell_face_sign=S6,
                        V=V,
                        fields_per_face=args.fields_per_face,
                        phi_in_face=args.phi_in_face,
                    )
                loss = loss_data + trial.lambda_cont * loss_cont

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item()) * bsz
                train_count += bsz

            local_train_loss = train_loss_sum / max(train_count, 1)
            train_loss = reduce_mean(local_train_loss, device, is_dist)

            model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_count = 0
                for Xc, Y6, *_ in loader_val:
                    Xc = Xc.to(device, non_blocking=pin_memory)
                    Y6 = Y6.to(device, non_blocking=pin_memory)
                    bsz = Xc.shape[0]
                    Xfaces = Xc.view(bsz, args.num_faces_per_cell, args.fields_per_face).reshape(
                        bsz * args.num_faces_per_cell,
                        args.fields_per_face,
                    )
                    pred_n = model(apply_zscore(Xfaces, x_mean, x_std)).view(bsz, args.num_faces_per_cell)
                    Yn = apply_zscore(Y6.reshape(bsz * args.num_faces_per_cell, 1), y_mean, y_std).view(
                        bsz,
                        args.num_faces_per_cell,
                    )
                    l = criterion(pred_n, Yn)
                    val_loss_sum += float(l.item()) * bsz
                    val_count += bsz

                local_val_loss = val_loss_sum / max(val_count, 1)
                val_loss = reduce_mean(local_val_loss, device, is_dist)

            if rank == 0:
                print(
                    f"[INFO] epoch {ep + 1}/{args.epochs} "
                    f"train_loss={train_loss:.6e} "
                    f"val_loss={val_loss:.6e}"
                )

        barrier(is_dist, local_rank)

        if rank == 0:
            save_norm(norm_path, x_mean, x_std, y_mean, y_std, meta={"from": "shared_norm", "run": trial.run_name})

            save_model = model.module if isinstance(model, DDP) else model
            save_model.eval()
            example = torch.rand(1, args.fields_per_face, device=device)
            scripted = torch.jit.script(save_model) if args.jit_mode == "script" else torch.jit.trace(save_model, example)
            scripted.save(jit_path)

            metrics, pred, ygt = eval_model_rank0(
                args,
                save_model,
                loader_eval_rank0,
                x_mean,
                x_std,
                y_mean,
                y_std,
                device,
                pin_memory,
            )
            print(
                f"[INFO] split={args.which} metrics: "
                f"MSE={metrics['mse']:.6e} RMSE={metrics['rmse']:.6e} MAE={metrics['mae']:.6e} "
                f"R2={metrics['r2']:.6f} relRMSE={metrics['rel_rmse']:.6e} cont_mean={metrics['cont_mean']:.6e}"
            )

            if args.save_pred:
                np.savez_compressed(
                    pred_npz,
                    split=args.which,
                    pred_pEqnFlux_6=pred,
                    gt_pEqnFlux_6=ygt,
                    **{f"metric_{k}": v for k, v in metrics.items()},
                    cell_npz=os.path.basename(args.cell_npz),
                    split_npz=os.path.basename(args.shared_split_npz),
                    norm_path=os.path.basename(norm_path),
                )

            row = {
                "jit": os.path.basename(jit_path),
                "cell_npz": os.path.basename(args.cell_npz),
                "split_npz": os.path.basename(args.shared_split_npz),
                "split": args.which,
                "Nsel": int(len(ds_eval)),
                "norm_path": os.path.basename(norm_path),
                **metrics,
            }
            with open(eval_metrics_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)

            summary_row = {
                "run_name": trial.run_name,
                "hidden": trial.hidden_label,
                "lr": trial.lr,
                "dropout": trial.dropout,
                "lambda_cont": trial.lambda_cont,
                "epochs": args.epochs,
                "batch": args.batch,
                "split": args.which,
                "jit": row["jit"],
                "Nsel": row["Nsel"],
                "mse": row["mse"],
                "rmse": row["rmse"],
                "mae": row["mae"],
                "r2": row["r2"],
                "rel_rmse": row["rel_rmse"],
                "cont_mean": row["cont_mean"],
                "norm_path": row["norm_path"],
                "run_dir": run_dir,
            }
            append_summary_row(summary_csv, summary_row)
            print(f"[INFO] appended summary to: {summary_csv}")

        barrier(is_dist, local_rank)

    if rank == 0 and args.run_vis:
        vis_cmd = [
            "python",
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
        print("[RUN]", " ".join(vis_cmd))
        subprocess.run(vis_cmd, check=True)

    if rank == 0:
        print("[INFO] Grid-search done.")
        print(f"[INFO] Summary CSV: {summary_csv}")

    cleanup_dist(is_dist, local_rank)


if __name__ == "__main__":
    main()


