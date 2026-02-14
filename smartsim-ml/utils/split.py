# utils/split.py
import os
import numpy as np

def make_split_indices(n: int, train: float, val: float, test: float, seed: int):
    assert abs(train + val + test - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train)
    n_val   = int(n * val)
    n_test  = n - n_train - n_val

    i_train = idx[:n_train]
    i_val   = idx[n_train:n_train+n_val]
    i_test  = idx[n_train+n_val:]
    assert len(i_test) == n_test
    return i_train, i_val, i_test

def save_split(path: str, train_idx, val_idx, test_idx, meta: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if meta is None:
        meta = {}
    np.savez(path,
             train_idx=np.asarray(train_idx, dtype=np.int64),
             val_idx=np.asarray(val_idx, dtype=np.int64),
             test_idx=np.asarray(test_idx, dtype=np.int64),
             **{f"meta_{k}": v for k, v in meta.items()})

def load_split(path: str):
    z = np.load(path, allow_pickle=True)
    return z["train_idx"], z["val_idx"], z["test_idx"]

