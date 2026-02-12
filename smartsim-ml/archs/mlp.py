# archs/mlp.py
import torch
import torch.nn as nn


_ACTS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden: list[int] = [64, 64],
        act: str = "tanh",
        dropout: float = 0.0,
    ):
        super().__init__()
        if act not in _ACTS:
            raise ValueError(f"Unknown act={act}. choose from {list(_ACTS.keys())}")
        Act = _ACTS[act]

        layers = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def build_model(input_shape, **kwargs):
    # input_shape expected like (3,)
    in_features = 1
    for s in input_shape:
        in_features *= int(s)
    if in_features != 3:
        raise ValueError(f"MLP expects 3 inputs, got input_shape={tuple(input_shape)} -> {in_features}")

    return MLP(in_features=in_features, out_features=1, **kwargs)



