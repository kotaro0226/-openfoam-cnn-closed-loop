#!/usr/bin/env python3
"""Distributed/GPU training entrypoint for Frontier-like environments.

Use with torchrun, e.g.:
  torchrun --standalone --nproc_per_node=8 train_frontier.py mlp --src sickle ...
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from train import (
    _build_model,
    _fit_norm_on_train,
    _parse_split,
    sickle_xy_cell,
    sickle_xy_face,
)
from utils.losses import continuity_loss_cell6
from utils.normalization import apply_zscore, invert_zscore, load_norm, save_norm
from utils.split import load_split, make_split_indices, save_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("arch", type=str, help="model arch in archs/")
    p.add_argument("--src", choices=["random", "redis", "sickle"], default="sickle")

    p.add_argument("-b", "--batch", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--sickle_x", type=str, default="")
    p.add_argument("--sickle_y", type=str, default="")
    p.add_argument("--sickle_z", type=str, default="")
    p.add_argument("--add_orient", action="store_true")
    p.add_argument("--cell_npz", type=str, default="")

    p.add_argument("--lambda-cont", type=float, default=0.0)
    p.add_argument("--phi-in-face", type=int, default=1)
    p.add_argument("--fields-per-face", type=int, default=3)
    p.add_argument("--num-faces-per-cell", type=int, default=6)

    p.add_argument("--hidden", nargs="+", type=int, default=[64, 64])
    p.add_argument("--act", type=str, default="tanh")
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--split", type=str, default="0.8,0.1,0.1")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--split-save", type=str, default="")
    p.add_argument("--split-load", type=str, default="")

    p.add_argument("--norm-path", type=str, default="")
    p.add_argument("--fit-norm", action="store_true")
    p.add_argument("--load-norm", action="store_true")

    p.add_argument("--save-jit", type=str, default="mlp_model.jit")
    p.add_argument("--jit-mode", choices=["trace", "script"], default="trace")

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")

    p.add_argument(
        "--device",
        choices=["auto", "cuda", "rocm", "cpu"],
        default="auto",
        help="Execution device. Use 'rocm' as an alias of torch's 'cuda' API on AMD/ROCm builds.",
    )
    p.add_argument("--ddp-backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--no-ddp", action="store_true", help="disable DDP even if torchrun env exists")
    return p.parse_args()


def setup_dist(args):
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    use_ddp = (not args.no_ddp) and rank_env is not None and world_size_env is not None and int(world_size_env) > 1

    if not use_ddp:
        return False, 0, 1, 0

    rank = int(rank_env)
    world_size = int(world_size_env)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = args.ddp_backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def cleanup_dist(is_dist):
    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def reduce_mean(value: float, device: torch.device, is_dist: bool):
    if not is_dist:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def resolve_device(device_arg: str) -> torch.device:
    # NOTE: PyTorch on ROCm still uses the 'cuda' device namespace/API.
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in {"cuda", "rocm"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    is_dist, rank, world_size, local_rank = setup_dist(args)

    device = resolve_device(args.device)

    if device.type == "cuda":
        torch.cuda.set_device(local_rank if is_dist else 0)
        device = torch.device("cuda", local_rank if is_dist else 0)

    if rank == 0:
        hip_ver = getattr(torch.version, "hip", None)
        if hip_ver:
            print(f"[INFO] Detected PyTorch ROCm build (HIP={hip_ver}). Using torch '{device.type}' device API.")

    pin_memory = bool(args.pin_memory) and (device.type == "cuda")

    if args.src != "sickle":
        raise ValueError("This script expects --src sickle for your current pipeline.")

    is_cell = bool(args.cell_npz)
    if is_cell:
        dataset, input_shape, output_shape = sickle_xy_cell(args.cell_npz)
        face_in = (args.fields_per_face,)
    else:
        dataset, input_shape, output_shape = sickle_xy_face(args)
        face_in = input_shape

    tr, va, te = _parse_split(args.split)
    N = len(dataset)
    if args.split_load:
        train_idx, val_idx, test_idx = load_split(args.split_load)
        if rank == 0:
            print(f"[INFO] loaded split: {args.split_load}  (N={N})")
    else:
        train_idx, val_idx, test_idx = make_split_indices(N, tr, va, te, args.seed)
        if args.split_save and rank == 0:
            meta = {"N": N, "split": args.split, "seed": args.seed, "is_cell": int(is_cell)}
            save_split(args.split_save, train_idx, val_idx, test_idx, meta=meta)
            print(f"[INFO] saved split: {args.split_save}  (N={N})")

    ds_train = Subset(dataset, train_idx)
    ds_val = Subset(dataset, val_idx)
    ds_test = Subset(dataset, test_idx)

    train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True) if is_dist else None
    val_sampler = DistributedSampler(ds_val, num_replicas=world_size, rank=rank, shuffle=False) if is_dist else None
    test_sampler = DistributedSampler(ds_test, num_replicas=world_size, rank=rank, shuffle=False) if is_dist else None

    loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    loader_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    loader_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, sampler=test_sampler,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    if (args.fit_norm or args.load_norm) and (not args.norm_path):
        raise ValueError("--fit-norm/--load-norm requires --norm-path")

    if args.fit_norm:
        if rank == 0:
            x_mean, x_std, y_mean, y_std = _fit_norm_on_train(args, is_cell, ds_train, input_shape, output_shape, str(device))
            save_norm(
                args.norm_path,
                x_mean,
                x_std,
                y_mean,
                y_std,
                meta={"fit_on": "train", "split": args.split, "seed": args.seed, "is_cell": int(is_cell)},
            )
            print(f"[INFO] saved norm stats: {args.norm_path}")
        if is_dist:
            dist.barrier()
        x_mean, x_std, y_mean, y_std = load_norm(args.norm_path, device=str(device))
    elif args.load_norm:
        x_mean, x_std, y_mean, y_std = load_norm(args.norm_path, device=str(device))
        if rank == 0:
            print(f"[INFO] loaded norm stats: {args.norm_path}")
    else:
        raise ValueError("Use --fit-norm or --load-norm with --norm-path.")

    model = _build_model(args, face_in).to(device)
    if is_dist:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in loader_train:
            optimizer.zero_grad(set_to_none=True)
            if is_cell:
                Xc, Y6, S6, V = [x.to(device, non_blocking=pin_memory) for x in batch]
                B = Xc.shape[0]
                Xfaces = Xc.view(B, args.num_faces_per_cell, args.fields_per_face).reshape(B * 6, args.fields_per_face)
                Xn = apply_zscore(Xfaces, x_mean, x_std)
                pred_n = model(Xn).view(B, args.num_faces_per_cell)
                Yn = apply_zscore(Y6.reshape(B * 6, 1), y_mean, y_std).view(B, args.num_faces_per_cell)
                loss_data = criterion(pred_n, Yn)
                loss_cont = torch.tensor(0.0, device=device)
                if args.lambda_cont > 0.0:
                    pred_phys = invert_zscore(pred_n.reshape(B * 6, 1), y_mean, y_std).view(B, args.num_faces_per_cell)
                    loss_cont = continuity_loss_cell6(
                        pEqnFlux_pred6=pred_phys,
                        X_cell=Xc,
                        cell_face_sign=S6,
                        V=V,
                        fields_per_face=args.fields_per_face,
                        phi_in_face=args.phi_in_face,
                    )
                loss = loss_data + args.lambda_cont * loss_cont
                bs = B
            else:
                X, Y = [x.to(device, non_blocking=pin_memory) for x in batch]
                pred_n = model(apply_zscore(X, x_mean, x_std))
                Yn = apply_zscore(Y, y_mean, y_std)
                loss = criterion(pred_n, Yn)
                bs = X.shape[0]

            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * bs
            train_count += bs

        local_train_loss = train_loss_sum / max(train_count, 1)
        train_loss = reduce_mean(local_train_loss, device, is_dist)

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_count = 0
            for batch in loader_val:
                if is_cell:
                    Xc, Y6, S6, V = [x.to(device, non_blocking=pin_memory) for x in batch]
                    B = Xc.shape[0]
                    Xfaces = Xc.view(B, args.num_faces_per_cell, args.fields_per_face).reshape(B * 6, args.fields_per_face)
                    pred_n = model(apply_zscore(Xfaces, x_mean, x_std)).view(B, args.num_faces_per_cell)
                    Yn = apply_zscore(Y6.reshape(B * 6, 1), y_mean, y_std).view(B, args.num_faces_per_cell)
                    l = criterion(pred_n, Yn)
                    val_loss_sum += float(l.item()) * B
                    val_count += B
                else:
                    X, Y = [x.to(device, non_blocking=pin_memory) for x in batch]
                    l = criterion(model(apply_zscore(X, x_mean, x_std)), apply_zscore(Y, y_mean, y_std))
                    val_loss_sum += float(l.item()) * X.shape[0]
                    val_count += int(X.shape[0])

            local_val_loss = val_loss_sum / max(val_count, 1)
            val_loss = reduce_mean(local_val_loss, device, is_dist)

        if rank == 0:
            print(f"[INFO] epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.6e} val_loss={val_loss:.6e}")

    model.eval()
    with torch.no_grad():
        test_loss_sum = 0.0
        test_count = 0
        for batch in loader_test:
            if is_cell:
                Xc, Y6, S6, V = [x.to(device, non_blocking=pin_memory) for x in batch]
                B = Xc.shape[0]
                Xfaces = Xc.view(B, args.num_faces_per_cell, args.fields_per_face).reshape(B * 6, args.fields_per_face)
                pred_n = model(apply_zscore(Xfaces, x_mean, x_std)).view(B, args.num_faces_per_cell)
                Yn = apply_zscore(Y6.reshape(B * 6, 1), y_mean, y_std).view(B, args.num_faces_per_cell)
                l = criterion(pred_n, Yn)
                test_loss_sum += float(l.item()) * B
                test_count += B
            else:
                X, Y = [x.to(device, non_blocking=pin_memory) for x in batch]
                l = criterion(model(apply_zscore(X, x_mean, x_std)), apply_zscore(Y, y_mean, y_std))
                test_loss_sum += float(l.item()) * X.shape[0]
                test_count += int(X.shape[0])

    local_test_loss = test_loss_sum / max(test_count, 1)
    test_loss = reduce_mean(local_test_loss, device, is_dist)
    if rank == 0:
        print(f"[INFO] test_loss={test_loss:.6e}")

    if rank == 0:
        save_model = model.module if isinstance(model, DDP) else model
        save_model.eval()
        example = torch.rand(1, *face_in, device=device)
        scripted = torch.jit.script(save_model) if args.jit_mode == "script" else torch.jit.trace(save_model, example)
        os.makedirs(os.path.dirname(args.save_jit) or ".", exist_ok=True)
        scripted.save(args.save_jit)
        print(f"[INFO] saved jit: {args.save_jit}")

    cleanup_dist(is_dist)


if __name__ == "__main__":
    main()



