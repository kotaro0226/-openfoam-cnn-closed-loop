import torch
import torch.nn as nn
import math


_ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def _make_norm(ndim: int, channels: int, norm: str) -> nn.Module:
    if norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(channels) if ndim == 2 else nn.BatchNorm3d(channels)
    if norm == "group":
        groups = math.gcd(channels, 8)
        return nn.GroupNorm(groups, channels)
    raise ValueError("Unknown norm={}. choose from ['none', 'batch', 'group']".format(norm))


class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: str = "gelu",
        dropout: float = 0.0,
        norm: str = "none",
        residual: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        Act = _ACTS[act]
        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.norm1 = _make_norm(2, out_channels, norm)
        self.act1 = Act()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.norm2 = _make_norm(2, out_channels, norm)
        self.act2 = Act()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if residual and in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        if self.residual:
            y = y + self.skip(x)
        return self.act2(y)


class ConvBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: str = "gelu",
        dropout: float = 0.0,
        norm: str = "none",
        residual: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        Act = _ACTS[act]
        self.residual = residual
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.norm1 = _make_norm(3, out_channels, norm)
        self.act1 = Act()
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.norm2 = _make_norm(3, out_channels, norm)
        self.act2 = Act()
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if residual and in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        if self.residual:
            y = y + self.skip(x)
        return self.act2(y)


class CNN2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        depth: int = 4,
        act: str = "gelu",
        dropout: float = 0.0,
        norm: str = "none",
        residual: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if act not in _ACTS:
            raise ValueError(f"Unknown act={act}. choose from {list(_ACTS.keys())}")

        layers = [
            ConvBlock2d(
                in_channels,
                width,
                act=act,
                dropout=dropout,
                norm=norm,
                residual=residual,
                padding_mode=padding_mode,
            )
        ]
        for _ in range(max(0, depth - 1)):
            layers.append(
                ConvBlock2d(
                    width,
                    width,
                    act=act,
                    dropout=dropout,
                    norm=norm,
                    residual=residual,
                    padding_mode=padding_mode,
                )
            )
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class CNN3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 8,
        depth: int = 3,
        act: str = "gelu",
        dropout: float = 0.0,
        norm: str = "none",
        residual: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if act not in _ACTS:
            raise ValueError(f"Unknown act={act}. choose from {list(_ACTS.keys())}")

        layers = [
            ConvBlock3d(
                in_channels,
                width,
                act=act,
                dropout=dropout,
                norm=norm,
                residual=residual,
                padding_mode=padding_mode,
            )
        ]
        for _ in range(max(0, depth - 1)):
            layers.append(
                ConvBlock3d(
                    width,
                    width,
                    act=act,
                    dropout=dropout,
                    norm=norm,
                    residual=residual,
                    padding_mode=padding_mode,
                )
            )
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv3d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model(input_shape, **kwargs):
    out_channels = kwargs.pop("out_channels", 6)
    if len(input_shape) == 3:
        in_channels, _, _ = input_shape
        return CNN2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if len(input_shape) == 4:
        in_channels, _, _, _ = input_shape
        return CNN3d(in_channels=in_channels, out_channels=out_channels, **kwargs)
    raise ValueError(f"CNN expects input_shape=(C,H,W) or (C,D,H,W), got {tuple(input_shape)}")
