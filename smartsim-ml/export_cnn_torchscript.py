#!/usr/bin/env python3
"""Export a 3D-block CNN checkpoint for C++/LibTorch inference.

The exported module accepts physical input:
    (B, 18, 20, 20, 20)

and returns physical core flux prediction:
    (B, 6, 16, 16, 16)

It wraps the trained model with the same input normalization, center crop, and
output inverse-normalization used by the Python evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from archs.cnn import build_model
from train_cnn import center_crop_3d


class CnnBlockInference(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        core_shape: tuple[int, int, int],
    ):
        super().__init__()
        self.model = model
        self.register_buffer("x_mean", x_mean.float())
        self.register_buffer("x_std", x_std.float())
        self.register_buffer("y_mean", y_mean.float())
        self.register_buffer("y_std", y_std.float())
        self.core_shape = core_shape

    def forward(self, x_phys: torch.Tensor) -> torch.Tensor:
        x_norm = (x_phys - self.x_mean) / self.x_std
        pred_norm = self.model(x_norm)
        pred_core_norm = center_crop_3d(pred_norm, self.core_shape)
        return pred_core_norm * self.y_std + self.y_mean


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    required = (
        "model_state",
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
        "in_channels",
        "out_channels",
        "input_dims",
        "core_shape",
    )
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise KeyError(f"checkpoint is missing keys: {missing}")
    if ckpt.get("grid_mode") != "3d-block":
        raise ValueError(f"expected grid_mode='3d-block', got {ckpt.get('grid_mode')}")
    return ckpt


def build_wrapped_model(ckpt: dict) -> CnnBlockInference:
    input_dims = tuple(int(v) for v in ckpt["input_dims"])
    core_shape = tuple(int(v) for v in ckpt["core_shape"])
    model = build_model(
        (int(ckpt["in_channels"]), *input_dims),
        out_channels=int(ckpt["out_channels"]),
        width=int(ckpt.get("width", 8)),
        depth=int(ckpt.get("depth", 3)),
        act=str(ckpt.get("act", "gelu")),
        dropout=float(ckpt.get("dropout", 0.0)),
        norm=str(ckpt.get("norm", "none")),
        residual=bool(ckpt.get("residual", False)),
        padding_mode=str(ckpt.get("padding_mode", "zeros")),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return CnnBlockInference(
        model=model,
        x_mean=ckpt["x_mean"],
        x_std=ckpt["x_std"],
        y_mean=ckpt["y_mean"],
        y_std=ckpt["y_std"],
        core_shape=core_shape,
    ).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-out", default="")
    parser.add_argument("--check", action="store_true", help="compare TorchScript output against eager output")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    wrapped = build_wrapped_model(ckpt)

    input_dims = tuple(int(v) for v in ckpt["input_dims"])
    example = torch.zeros((1, int(ckpt["in_channels"]), *input_dims), dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example, strict=True)
        traced = torch.jit.freeze(traced.eval())

        if args.check:
            sample = torch.randn_like(example)
            eager = wrapped(sample)
            scripted = traced(sample)
            max_abs = float((eager - scripted).abs().max().item())
            print(f"[INFO] eager-vs-torchscript max_abs={max_abs:.6e}")
            if max_abs > 1e-5:
                raise RuntimeError(f"TorchScript check failed: max_abs={max_abs}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"[INFO] wrote TorchScript model: {out_path}")

    metadata = {
        "checkpoint": args.checkpoint,
        "torchscript": str(out_path),
        "input_shape": [1, int(ckpt["in_channels"]), *input_dims],
        "output_shape": [1, int(ckpt["out_channels"]), *tuple(int(v) for v in ckpt["core_shape"])],
        "input_units": "physical, unnormalized",
        "output_units": "physical pEqnFlux, center-cropped core",
        "grid_mode": ckpt.get("grid_mode"),
        "width": int(ckpt.get("width", 8)),
        "depth": int(ckpt.get("depth", 3)),
        "norm": str(ckpt.get("norm", "none")),
        "padding_mode": str(ckpt.get("padding_mode", "zeros")),
        "residual": bool(ckpt.get("residual", False)),
        "dropout": float(ckpt.get("dropout", 0.0)),
    }
    metadata_out = args.metadata_out or os.path.splitext(args.out)[0] + ".json"
    meta_path = Path(metadata_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[INFO] wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
