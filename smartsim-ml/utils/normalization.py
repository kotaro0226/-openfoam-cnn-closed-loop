# utils/normalization.py
import os
import numpy as np
import torch

class RunningZScore:
    def __init__(self, dim: int, device="cpu", eps=1e-8):
        self.dim = dim
        self.device = device
        self.eps = eps
        self.n = 0
        self.mean = torch.zeros(dim, device=device)
        self.M2 = torch.zeros(dim, device=device)  # sum of squares of diffs

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        # x: (N, C)
        x = x.to(self.device)
        if x.dim() != 2 or x.size(1) != self.dim:
            raise ValueError(f"Expected (N,{self.dim}), got {tuple(x.shape)}")

        for i in range(x.size(0)):
            self.n += 1
            delta = x[i] - self.mean
            self.mean += delta / self.n
            delta2 = x[i] - self.mean
            self.M2 += delta * delta2

    @torch.no_grad()
    def finalize(self):
        if self.n < 2:
            var = torch.ones_like(self.mean)
        else:
            var = self.M2 / self.n  # population variance
        std = torch.sqrt(var).clamp_min(self.eps)
        return self.mean, std

def apply_zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return (x - mean) / std

def invert_zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return x * std + mean

def save_norm(path: str, x_mean, x_std, y_mean, y_std, meta: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if meta is None:
        meta = {}
    np.savez(path,
             x_mean=x_mean.detach().cpu().numpy(),
             x_std=x_std.detach().cpu().numpy(),
             y_mean=y_mean.detach().cpu().numpy(),
             y_std=y_std.detach().cpu().numpy(),
             **{f"meta_{k}": v for k, v in meta.items()})

def load_norm(path: str, device="cpu"):
    z = np.load(path, allow_pickle=True)
    x_mean = torch.from_numpy(z["x_mean"]).to(device)
    x_std  = torch.from_numpy(z["x_std"]).to(device)
    y_mean = torch.from_numpy(z["y_mean"]).to(device)
    y_std  = torch.from_numpy(z["y_std"]).to(device)
    return x_mean, x_std, y_mean, y_std

