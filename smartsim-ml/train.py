# smartsim/train.py
import argparse
import importlib
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from utils.losses import continuity_loss_cell6

def parse_args():
    archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
    p = argparse.ArgumentParser()

    p.add_argument('arch', type=str, choices=archs, help='model arch in archs/')
    p.add_argument('--src', choices=['random', 'redis', 'sickle'], default='random')

    p.add_argument('-b', '--batch', type=int, default=4096)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)

    # SICKLE (FACE) datasets
    p.add_argument('--sickle_x', type=str, default='')
    p.add_argument('--sickle_y', type=str, default='')
    p.add_argument('--sickle_z', type=str, default='')
    p.add_argument('--add_orient', action='store_true')

    # SICKLE (CELL) dataset (1 file)
    p.add_argument('--cell_npz', type=str, default='')

    # CELL loss controls
    p.add_argument('--lambda-cont', type=float, default=0.0)
    p.add_argument('--phi-in-face', type=int, default=1, help='0:rAUf 1:phiHbyA 2:div (inside each face triplet)')
    p.add_argument('--fields-per-face', type=int, default=3, help='default 3 (rAUf,phiHbyA,div)')
    p.add_argument('--num-faces-per-cell', type=int, default=6)

    # MLP hyperparams
    p.add_argument('--hidden', nargs='+', type=int, default=[64, 64])
    p.add_argument('--act', type=str, default='tanh')
    p.add_argument('--dropout', type=float, default=0.0)

    # Save / Redis
    p.add_argument('--save-jit', type=str, default='mlp_model.jit')
    p.add_argument('--jit-mode', choices=['trace', 'script'], default='trace')

    return p.parse_args()

# ---------- FACE (unchanged) ----------
def sickle_npz_xy(npz_path: str, orient_id: int | None):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]  # (T, Ns, Cin)
    Y = z["Y"]  # (T, Ns, Cout)
    T, Ns, Cin = X.shape
    Cout = Y.shape[2]
    Xf = X.reshape(T * Ns, Cin).astype(np.float32)
    Yf = Y.reshape(T * Ns, Cout).astype(np.float32)
    if orient_id is not None:
        orient_col = np.full((Xf.shape[0], 1), float(orient_id), dtype=np.float32)
        Xf = np.concatenate([Xf, orient_col], axis=1)
    return Xf, Yf, (Xf.shape[1],), (Cout,)

def sickle_xy_face(args):
    sets = []
    shapes = None

    def add(path, oid):
        nonlocal shapes
        if not path:
            return
        X, Y, in_shape, out_shape = sickle_npz_xy(path, oid if args.add_orient else None)
        ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                           torch.as_tensor(Y, dtype=torch.float32))
        sets.append(ds)
        if shapes is None:
            shapes = (in_shape, out_shape)
        else:
            if shapes[0] != in_shape or shapes[1] != out_shape:
                raise ValueError("Shape mismatch among sickle datasets")

    add(args.sickle_x, 0)
    add(args.sickle_y, 1)
    add(args.sickle_z, 2)

    if not sets:
        raise ValueError("No sickle FACE datasets provided.")
    return ConcatDataset(sets), shapes[0], shapes[1]

# ---------- CELL (new) ----------
def sickle_xy_cell(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]  # (T, Ns_cell, 18)
    Y = z["Y"]  # (T, Ns_cell, 6)
    if "V_sub_list" not in z or "cell_face_sign_sub_list" not in z:
        raise KeyError("CELL npz must include V_sub_list and cell_face_sign_sub_list")

    V = np.asarray(z["V_sub_list"], dtype=np.float32)                 # (T, Ns_cell)
    S = np.asarray(z["cell_face_sign_sub_list"], dtype=np.float32)    # (T, Ns_cell, 6)

    T, Ns, Cin = X.shape
    Cout = Y.shape[2]

    Xf = X.reshape(T * Ns, Cin).astype(np.float32)     # (Ncells,18)
    Yf = Y.reshape(T * Ns, Cout).astype(np.float32)    # (Ncells,6)
    Vf = V.reshape(T * Ns).astype(np.float32)          # (Ncells,)
    Sf = S.reshape(T * Ns, 6).astype(np.float32)       # (Ncells,6)

    ds = TensorDataset(torch.from_numpy(Xf),
                       torch.from_numpy(Yf),
                       torch.from_numpy(Sf),
                       torch.from_numpy(Vf))
    return ds, (Cin,), (Cout,)

def main():
    args = parse_args()

    if args.src != "sickle":
        raise ValueError("This script version expects --src sickle for your pipeline right now.")

    is_cell = bool(args.cell_npz)

    if is_cell:
        dataset, input_shape, output_shape = sickle_xy_cell(args.cell_npz)
        # model input is per-face triplet (3), not 18
        face_in = (args.fields_per_face,)
    else:
        dataset, input_shape, output_shape = sickle_xy_face(args)
        face_in = input_shape  # face pipeline as-is

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)

    model_module = importlib.import_module('archs.' + args.arch)
    # Build model for per-face input:
    model = model_module.build_model(
        face_in,
        hidden=args.hidden,
        act=args.act,
        dropout=args.dropout
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        last = None

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            if is_cell:
                Xc, Y6, S6, V = batch
                Xc = Xc.to(device)   # (B,18)
                Y6 = Y6.to(device)   # (B,6)
                S6 = S6.to(device)   # (B,6)
                V = V.to(device)     # (B,)

                B = Xc.shape[0]
                X6 = Xc.view(B, args.num_faces_per_cell, args.fields_per_face)  # (B,6,3)

                # Apply shared MLP to all faces:
                Xfaces = X6.reshape(B * args.num_faces_per_cell, args.fields_per_face)  # (B*6,3)
                pred = model(Xfaces).view(B, args.num_faces_per_cell)                   # (B,6)

                loss_data = criterion(pred, Y6)

                loss_cont = torch.tensor(0.0, device=device)
                if args.lambda_cont > 0.0:
                    loss_cont = continuity_loss_cell6(
                        pEqnFlux_pred6=pred,
                        X_cell=Xc,
                        cell_face_sign=S6,
                        V=V,
                        fields_per_face=args.fields_per_face,
                        phi_in_face=args.phi_in_face,
                    )
                loss = loss_data + args.lambda_cont * loss_cont
            else:
                X, Y = batch
                X = X.to(device)
                Y = Y.to(device)
                pred = model(X)
                loss = criterion(pred, Y)

            loss.backward()
            optimizer.step()
            last = loss.item()

        print(f"[INFO] epoch {epoch+1}/{args.epochs} loss={last:.6e}")

    # JIT save
    model.eval()
    example = torch.rand(1, *face_in, device=device)
    if args.jit_mode == "script":
        scripted = torch.jit.script(model)
    else:
        scripted = torch.jit.trace(model, example)
    scripted.save(args.save_jit)
    print(f"[INFO] saved jit: {args.save_jit}")

if __name__ == "__main__":
    main()




