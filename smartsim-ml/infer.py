# smartsim/infer.py
import argparse
import os
import csv
import numpy as np
import torch

from utils.losses import continuity_loss_cell6


def parse_timesteps(spec: str, T: int):
    """
    spec examples:
      "all"
      "0,1,2"
      "10:20"   # end exclusive
      ":-1"
    """
    spec = (spec or "all").strip().lower()
    if spec in ("all", ""):
        return list(range(T))

    if ":" in spec:
        a, b = spec.split(":")
        a = int(a) if a != "" else 0
        b = int(b) if b != "" else T
        a = max(a, 0)
        b = min(b, T)
        return list(range(a, b))

    out = []
    for s in spec.split(","):
        s = s.strip()
        if not s:
            continue
        i = int(s)
        if i < 0:
            i = T + i
        if 0 <= i < T:
            out.append(i)

    if not out:
        raise ValueError(f"Invalid --timesteps={spec} for T={T}")
    return out


def stats_line(name: str, a: np.ndarray) -> str:
    a = np.asarray(a)
    return (
        f"{name}: min={float(np.nanmin(a)):.6e} "
        f"max={float(np.nanmax(a)):.6e} "
        f"mean={float(np.nanmean(a)):.6e} "
        f"std={float(np.nanstd(a)):.6e}"
    )


def main():
    ap = argparse.ArgumentParser()

    # model / data
    ap.add_argument("--jit", required=True, help="TorchScript .jit")
    ap.add_argument("--dataset_npz", required=True, help="raw dataset npz containing X and (optionally) Y")
    ap.add_argument("--x_key", default="X")
    ap.add_argument("--y_key", default="Y")

    # inference control
    ap.add_argument("--timesteps", default="all", help='"all", "0,1,2", "10:20"')
    ap.add_argument("--batch_faces", type=int, default=1_048_576)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # continuity diagnostics (optional)
    ap.add_argument("--cont_eval", action="store_true")
    ap.add_argument("--cell_maps_npz", default="")
    ap.add_argument("--fields_per_face", type=int, default=3)
    ap.add_argument("--phi_in_face", type=int, default=1)
    ap.add_argument("--num_faces_per_cell", type=int, default=6)

    # outputs
    ap.add_argument("--out", required=True, help="output npz")
    ap.add_argument("--metrics_csv", default="", help="append one-line metrics to CSV")
    ap.add_argument("--save_float32", action="store_true")

    # debug
    ap.add_argument("--debug_stats", action="store_true", help="print Y/Pred min/max/mean/std")

    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load model
    model = torch.jit.load(args.jit, map_location=device)
    model.eval()

    # Load dataset
    z = np.load(args.dataset_npz, allow_pickle=True)
    if args.x_key not in z.files:
        raise KeyError(f"{args.dataset_npz} has no key '{args.x_key}'. keys={z.files}")

    X = z[args.x_key]  # (T, Nf, 3)
    if X.ndim != 3 or X.shape[2] != args.fields_per_face:
        raise ValueError(f"Expected X(T,Nf,{args.fields_per_face}), got X.shape={X.shape}")

    T, Nf, Cin = X.shape
    t_list = parse_timesteps(args.timesteps, T)

    print(f"[INFO] dataset={os.path.basename(args.dataset_npz)}  X.shape={X.shape}")
    print(f"[INFO] timesteps={args.timesteps} -> Tsel={len(t_list)} (min={min(t_list)} max={max(t_list)})")
    print(f"[INFO] device={device}  batch_faces={args.batch_faces}")

    # Load cell maps if cont_eval
    if args.cont_eval:
        if not args.cell_maps_npz:
            raise ValueError("--cont_eval requires --cell_maps_npz")
        cm = np.load(args.cell_maps_npz, allow_pickle=True)
        for k in ("cell_faces_idx", "cell_face_sign", "V"):
            if k not in cm.files:
                raise KeyError(f"{args.cell_maps_npz} missing '{k}'. keys={cm.files}")

        cell_faces = cm["cell_faces_idx"].astype(np.int64)    # (Nc,6)
        cell_sign  = cm["cell_face_sign"].astype(np.float32)  # (Nc,6)
        V          = cm["V"].astype(np.float32)               # (Nc,)
        Nc = cell_faces.shape[0]

        mx = int(cell_faces.max())
        if mx >= Nf:
            raise RuntimeError(
                f"Face indexing mismatch: cell_faces_idx max={mx} but dataset Nf={Nf}.\n"
                "Check that cfd_xy_*.npz uses internal face indexing compatible with cell_maps_*.npz."
            )
        if cell_faces.shape[1] != args.num_faces_per_cell:
            raise ValueError(f"cell_faces_idx second dim must be {args.num_faces_per_cell}, got {cell_faces.shape}")

        print(f"[INFO] cont_eval=ON  cell_faces_idx.shape={cell_faces.shape}  Nc={Nc}")
    else:
        cell_faces = cell_sign = V = None
        Nc = 0
        print("[INFO] cont_eval=OFF")

    # Buffers
    Ypred_face = np.empty((len(t_list), Nf), dtype=np.float32)
    cont_loss = np.full((len(t_list),), np.nan, dtype=np.float32)

    with torch.no_grad():
        for ti, t in enumerate(t_list):
            Xt = X[t].astype(np.float32, copy=False)  # (Nf,3)

            # chunked face inference
            chunks = []
            for s in range(0, Nf, args.batch_faces):
                e = min(s + args.batch_faces, Nf)
                xb = torch.from_numpy(Xt[s:e]).to(device)  # (B,3)
                yb = model(xb).reshape(-1)                 # (B,)
                chunks.append(yb.detach().cpu().numpy())

            y_face = np.concatenate(chunks, axis=0)        # (Nf,)
            Ypred_face[ti, :] = y_face

            # per-timestep progress log (not too noisy)
            if (ti == 0) or ((ti + 1) % 10 == 0) or (ti == len(t_list) - 1):
                print(f"[INFO] inferred t={t} ({ti+1}/{len(t_list)})")

            # continuity diagnostic (scalar per timestep)
            if args.cont_eval:
                X6 = Xt[cell_faces, :]  # (Nc,6,3)
                Xc = X6.reshape(Nc, args.num_faces_per_cell * args.fields_per_face)  # (Nc,18)
                P6 = y_face[cell_faces]  # (Nc,6)

                Xc_t = torch.from_numpy(Xc).to(device)
                P6_t = torch.from_numpy(P6.astype(np.float32)).to(device)
                S6_t = torch.from_numpy(cell_sign).to(device)
                V_t  = torch.from_numpy(V).to(device)

                c = continuity_loss_cell6(
                    pEqnFlux_pred6=P6_t,
                    X_cell=Xc_t,
                    cell_face_sign=S6_t,
                    V=V_t,
                    fields_per_face=args.fields_per_face,
                    phi_in_face=args.phi_in_face,
                )
                cont_loss[ti] = float(c.detach().cpu().item())

    # ---- Metrics vs ground truth Y (aggregate) ----
    metrics = {}
    Ysel = None
    Pred = Ypred_face.astype(np.float32, copy=False)

    if args.y_key in z.files:
        Y = z[args.y_key]
        if Y.ndim == 3 and Y.shape[2] == 1:
            Y = Y[:, :, 0]  # (T,Nf)

        if Y.ndim == 2 and Y.shape[0] >= max(t_list) + 1 and Y.shape[1] == Nf:
            Ysel = Y[t_list].astype(np.float32, copy=False)  # (Tsel,Nf)
            err = Pred - Ysel

            mse = float(np.mean(err ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(err)))

            y_mean = float(np.mean(Ysel))
            ss_res = float(np.sum((Pred - Ysel) ** 2))
            ss_tot = float(np.sum((Ysel - y_mean) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

            y_std = float(np.std(Ysel))
            rel_rmse = float(rmse / y_std) if y_std > 0 else float("nan")

            metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "rel_rmse": rel_rmse}
            print(f"[INFO] metrics: MSE={mse:.6e} RMSE={rmse:.6e} MAE={mae:.6e} R2={r2:.6f} relRMSE={rel_rmse:.6e}")

            if args.debug_stats:
                print("[DBG] " + stats_line("Y (selected)", Ysel))
                print("[DBG] " + stats_line("Pred", Pred))
                print("[DBG] " + stats_line("Err", err))
        else:
            print(f"[WARN] Y exists but shape mismatch: Y.shape={getattr(Y,'shape',None)}; skip metrics.")
    else:
        print("[WARN] dataset has no Y; skip metrics.")

    # ---- Save NPZ ----
    face_out = Ypred_face.astype(np.float32) if args.save_float32 else Ypred_face.astype(np.float64)
    np.savez_compressed(
        args.out,
        t_index=np.asarray(t_list, dtype=np.int32),
        pEqnFlux_pred_face=face_out,     # (Tsel,Nf)
        continuity_loss=cont_loss,       # (Tsel,) NaN if cont_eval off
        **({f"metric_{k}": v for k, v in metrics.items()}),
    )
    print(f"[INFO] wrote {args.out}  pred.shape={face_out.shape}")

    # ---- Append CSV (optional) ----
    if args.metrics_csv:
        row = {
            "jit": os.path.basename(args.jit),
            "dataset": os.path.basename(args.dataset_npz),
            "timesteps": args.timesteps,
            "Tsel": len(t_list),
            "Nfaces": Nf,
        }
        row.update(metrics)
        if args.cont_eval:
            row["continuity_loss_mean"] = float(np.nanmean(cont_loss))

        write_header = (not os.path.exists(args.metrics_csv)) or (os.path.getsize(args.metrics_csv) == 0)
        with open(args.metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"[INFO] appended metrics to {args.metrics_csv}")


if __name__ == "__main__":
    main()


