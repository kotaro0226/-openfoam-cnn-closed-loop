import numpy as np
import os

from args import args 
from constants import FieldPredictionType
from dataloaders import load_data
from energy import EnergyMonitor
from helpers import check_and_create_dirs, get_calling_filename
from hypercubes import get_hypercube_extractor
from plotting import plot_samples, plot2d_contour
from subsampling import get_subsampler
from viz import save_vtu
from helpers import setup_rank_print
setup_rank_print()

def extract_yz_plane(X, timestep, feature_index, x_index, nx=128, ny=64, nz=128):
    """Extracts the y-z plane for a given x-index at a specific timestep and feature."""
    data_slice = X[timestep, :, feature_index]
    data_3d = data_slice.reshape(nx, ny, nz)
    return data_3d[x_index, :, :]


def subsample_data(X, Y, x, y, z, subsampler, args):
    num_timesteps = X.shape[0]
    print(f"\n\n\n\033[92m \U0001F680 Starting subsampling using {args.method} method\033[0m")
    print(f"num_timesteps: {num_timesteps}")
    num_samples_total = args.num_samples * args.num_hypercubes
    print(f"num_samples_total: {num_samples_total}")

    if args.method == "full":
        Xout = np.zeros((num_timesteps, X.shape[1], X.shape[2]))
    else:
        Xout = np.zeros((num_timesteps, num_samples_total, X.shape[2]))

    if args.field_prediction_type == FieldPredictionType.GLOBAL:
        Yout = np.zeros((num_timesteps, 1, Y.shape[2]))
    elif args.field_prediction_type == FieldPredictionType.LOCAL:
        if args.method == "full":
            raise Exception("For baseline full field input, prediction cannot be subsampled. Change `args.target`.")
        Yout = np.zeros((num_timesteps, num_samples_total, Y.shape[2]))
    elif args.field_prediction_type == FieldPredictionType.FULL:
        Yout = np.zeros((num_timesteps, Y.shape[1], Y.shape[2]))
    else:
        raise Exception("Enter a valid `args.target`.")

    subsampled_indices_list = []  # Store subsampled indices for later use

    for timestep in range(0, num_timesteps - args.window + 1, args.window):
        indices = subsampler.sample(num_samples_total, timestep)
        subsampled_indices_list.append(indices)
        print(f"timestep: {timestep}")

        #if args.plot and args.method != "full":
        #    plot_samples(indices, x, y, z, timestep, args)

        for sub_timestep in range(args.window):
            ts = timestep + sub_timestep
            if args.method == "full":
                subsampled_X = X[ts, :, :]
            else:
                subsampled_X = X[ts, indices]

            if args.field_prediction_type == FieldPredictionType.GLOBAL:
                subsampled_Y = Y[ts, :]
            elif args.field_prediction_type == FieldPredictionType.FULL:
                subsampled_Y = Y[ts, :, :]
            else:
                subsampled_Y = Y[ts, indices, :]
            
            Xout[ts, :] = subsampled_X
            Yout[ts, :] = subsampled_Y

            #if args.plot:
            #    if args.method == "full":
            #        yz_plane = extract_yz_plane(Xout, timestep, 3, 0, nx=args.nxsl, ny=args.nysl, nz=args.nzsl)
            #        plot2d_contour(yz_plane, y, z, ts)

    return Xout, Yout, np.array(subsampled_indices_list)


if __name__ == "__main__":
    """
    Output subsampled data for ML training.
    Output data shape:
        - Xout: [(T * num_cubes), [X,Y,Z]-or-NSAMPLES, C]
        - Yout: [(T * num_cubes), [X,Y,Z]-or-NSAMPLES-or-1, C]
    """
 
    # Ensure required directories exist
    check_and_create_dirs(args.output_dir)
    check_and_create_dirs(args.plot_dir)

    # Define hypercube extraction function
    extractor = get_hypercube_extractor(args.hypercubes, use_parallel=True)

    # Load the data
    print(f"\n\n\n\033[92m \U0001F680 Loading data / extracting hypercubes using {args.hypercubes} method\033[0m")
    X, Y, cv, x, y, z = load_data(args, extractor=extractor)
    num_timesteps = X.shape[0]
    print(f"X: {X.shape}; Y: {Y.shape}; cv: {cv.shape}; x: {x.shape}; y: {y.shape}; z: {z.shape}; num_timesteps: {num_timesteps}")

    # Check that sampling will work with current settings
    time_range = range(0, num_timesteps - args.window + 1, args.window)
    if len(time_range) == 0:
        raise ValueError("Error: The timestep loop will not execute because the computed range is empty. Check 'num_timesteps' and 'args.window'.")

    # Define subsample function based on method
    if args.method == "maxent":
        subsampler = get_subsampler(X, args, coords=(x, y, z), cv=cv)
    else:
        subsampler = get_subsampler(X, args)

    # Perform subsampling
    if os.path.exists("/sys/cray/pm_counters"):
        em = EnergyMonitor(get_calling_filename())
        em.start()

    Xout, Yout, indices_list = subsample_data(X, Y, x, y, z, subsampler, args)

    if os.path.exists("/sys/cray/pm_counters"):
        em.end()
        em.aggregate()

    print(f"Xout: {Xout.shape}; Yout: {Yout.shape}")

    # Save to VTK unstructured format
    if args.viz:
        save_vtu(X, Y, x, y, z, indices_list, args.output_dir, args.fileprefix)

    # Reshape Xout and Yout to 1D or 3D based on args.method and args.field_prediction_type
    num_timesteps *= args.num_hypercubes
    if args.method == "full":
        Xout = Xout.reshape(num_timesteps, len(x), len(y), len(z), Xout.shape[-1])
    else:
        Xout = Xout.reshape(num_timesteps, args.num_samples, Xout.shape[-1])
    
    if args.field_prediction_type == FieldPredictionType.FULL:
        Yout = Yout.reshape(num_timesteps, len(x), len(y), len(z), Yout.shape[-1])
    elif args.field_prediction_type == FieldPredictionType.LOCAL:
        Yout = Yout.reshape(num_timesteps, args.num_samples, Yout.shape[-1])

    print(f"After reshaping: Xout: {Xout.shape}; Yout: {Yout.shape}")



    # ============================================================
    # Save output
    # ============================================================
    outfilename = f"subsampled_{args.fileprefix}.npz"
    outfile = os.path.join(args.output_dir, outfilename)

    # NOTE:
    # At this point (after "After reshaping"), original SICKLE behavior is:
    #   Xout: (T_effective * num_hypercubes, num_samples, Cx)
    #   Yout: (T_effective * num_hypercubes, num_samples, Cy)
    #
    # To keep output format consistent across random/maxent, we also want:
    #   indices_list: list length = (T_effective * num_hypercubes)
    #                 each element length = num_samples
    #
    # However, in the current code, indices_list is stored per timestep:
    #   indices_list[t] length = (num_hypercubes * num_samples)
    #
    # So we "blockify" indices_list to match Xout/Yout first dimension.

    # -----------------------------
    # 1) Convert timestep-level indices_list -> block-level indices_list
    # -----------------------------
    T_effective = len(indices_list)
    if T_effective <= 0:
        raise RuntimeError("indices_list is empty; nothing to save.")

    # Xout already reshaped: (T_effective*H, Ns, C)
    if Xout.ndim != 3 or Yout.ndim != 3:
        raise RuntimeError(f"Unexpected Xout/Yout ndim: Xout.ndim={Xout.ndim}, Yout.ndim={Yout.ndim}")

    Ns = int(Xout.shape[1])  # == args.num_samples
    if Ns <= 0:
        raise RuntimeError(f"Invalid Ns inferred from Xout.shape[1]={Xout.shape[1]}")

    # infer H from first indices array length
    idx0 = np.asarray(indices_list[0], dtype=np.int64)
    if idx0.size % Ns != 0:
        raise RuntimeError(
            f"indices_list[0] length {idx0.size} is not divisible by Ns={Ns}. "
            "This means your indices packing is inconsistent with Xout reshaping."
        )
    H = int(idx0.size // Ns)

    # sanity against Xout leading dim
    expected_TH = T_effective * H
    if Xout.shape[0] != expected_TH or Yout.shape[0] != expected_TH:
        raise RuntimeError(
            f"Xout/Yout leading dim mismatch: "
            f"Xout.shape[0]={Xout.shape[0]}, Yout.shape[0]={Yout.shape[0]} "
            f"but expected T_effective*H={expected_TH} (T_effective={T_effective}, H={H})."
        )

    # blockify indices_list so it aligns with Xout[block_id]
    # indices_list_blocked length = T_effective*H, each (Ns,)
    indices_list_blocked = []
    for t, idx in enumerate(indices_list):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size != H * Ns:
            raise RuntimeError(
                f"indices_list[{t}] length mismatch: got {idx.size} but expected {H*Ns} (H={H}, Ns={Ns})."
            )
        idx2 = idx.reshape(H, Ns)  # assumes packing matches Xout reshape order
        for h in range(H):
            indices_list_blocked.append(idx2[h].copy())

    # store as object array (original SICKLE style)
    indices_obj = np.array(indices_list_blocked, dtype=object)

    # -----------------------------
    # 2) Save NPZ (keep original keys)
    # -----------------------------
    save_kwargs = dict(
        X=Xout,
        Y=Yout,
        x=x, y=y, z=z,
        indices_list=indices_obj,  # <-- now block-level, aligned with Xout/Yout
    )

    # -----------------------------
    # 3) CELL dtype: add extra keys WITHOUT changing base format
    # -----------------------------
    if getattr(args, "dtype", None) == "of-cell-npz":
        if not hasattr(args, "cell_npz") or not args.cell_npz:
            raise ValueError("dtype='of-cell-npz' requires --cell_npz pointing to cell_maps_*.npz")

        cm = np.load(args.cell_npz, allow_pickle=True)

        if "V" not in cm.files or "cell_face_sign" not in cm.files:
            raise KeyError(f"{args.cell_npz} must include 'V' and 'cell_face_sign'")

        V_all = cm["V"].astype(np.float32)               # (Ncells_total,)
        sign_all = cm["cell_face_sign"].astype(np.int8)  # (Ncells_total, 6)

        # Build V/S aligned with blockified indices_list
        V_blk = np.empty((expected_TH, Ns), dtype=np.float32)
        S_blk = np.empty((expected_TH, Ns, 6), dtype=np.int8)

        for b in range(expected_TH):
            cell_ids = np.asarray(indices_list_blocked[b], dtype=np.int64)  # (Ns,)
            V_blk[b, :] = V_all[cell_ids]
            S_blk[b, :, :] = sign_all[cell_ids, :]

        save_kwargs["V_sub_list"] = V_blk
        save_kwargs["cell_face_sign_sub_list"] = S_blk

        # optional metadata (do not break original loaders)
        save_kwargs["num_hypercubes"] = np.int32(H)
        save_kwargs["num_samples"] = np.int32(Ns)
        save_kwargs["T_effective"] = np.int32(T_effective)
        if "dims_cells" in cm.files:
            save_kwargs["dims_cells"] = cm["dims_cells"]

        print(f"[INFO] CELL: X={Xout.shape} Y={Yout.shape} indices_list(blocked)={len(indices_list_blocked)} "
              f"V_sub_list={V_blk.shape} cell_face_sign_sub_list={S_blk.shape} "
              f"(T_effective={T_effective}, H={H}, Ns={Ns})")

    np.savez(outfile, **save_kwargs)
    print(f"Subsampled data saved to {outfile}")




