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


    # Save output
    outfilename = f"subsampled_{args.fileprefix}.npz"
    outfile = os.path.join(args.output_dir, outfilename)

    # indices_list は object で保存（可変長対応）
    indices_obj = np.array(indices_list, dtype=object)

    # -----------------------------
    # CELL dtype: X is (T, Ns_cell, 18) and Y is (T, Ns_cell, 6)
    # indices_list are sampled CELL ids.
    # We DO NOT need face_ids_flat nor face_ids_list.
    # We only need cell volume and face signs to compute continuity loss.
    # -----------------------------
    if getattr(args, "dtype", None) == "of-cell-npz":
        if not hasattr(args, "cell_npz") or not args.cell_npz:
            raise ValueError("dtype='of-cell-npz' requires --cell_npz pointing to cell_maps_*.npz")

        cm = np.load(args.cell_npz, allow_pickle=True)

        # required in cell_maps
        if "V" not in cm.files or "cell_face_sign" not in cm.files:
            raise KeyError(f"{args.cell_npz} must include 'V' and 'cell_face_sign'")

        V_all = cm["V"].astype(np.float32)               # (Ncells_total,)
        sign_all = cm["cell_face_sign"].astype(np.int8)  # (Ncells_total, 6)

        V_sub_list = []
        cell_face_sign_sub_list = []

        for idx in indices_list:
            idx = np.asarray(idx, dtype=np.int64)  # sampled CELL ids for this block
            V_sub_list.append(V_all[idx])
            cell_face_sign_sub_list.append(sign_all[idx, :])

        save_kwargs = dict(
            X=Xout, Y=Yout, x=x, y=y, z=z,
            indices_list=indices_obj,
            V_sub_list=np.array(V_sub_list, dtype=object),
            cell_face_sign_sub_list=np.array(cell_face_sign_sub_list, dtype=object),
        )

        # optional metadata
        if "dims_cells" in cm.files:
            save_kwargs["dims_cells"] = cm["dims_cells"]

        np.savez(outfile, **save_kwargs)
        print(f"Subsampled data saved to {outfile}")

    # -----------------------------
    # FACE dtype (or others): keep the old behavior.
    # indices_list are sampled FACE ids.
    # If you want face_ids_list, you must supply args.face_ids_flat.
    # -----------------------------
    else:
        face_ids_list = None
        if hasattr(args, "face_ids_flat") and args.face_ids_flat is not None:
            flat = np.asarray(args.face_ids_flat)
            face_ids_list = np.array(
                [flat[np.asarray(idx, dtype=np.int64)] for idx in indices_list],
                dtype=object
            )

        save_kwargs = dict(
            X=Xout, Y=Yout, x=x, y=y, z=z,
            indices_list=indices_obj,
            face_ids_list=face_ids_list
        )

        np.savez(outfile, **save_kwargs)
        print(f"Subsampled data saved to {outfile}")





