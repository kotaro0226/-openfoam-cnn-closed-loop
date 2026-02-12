import numpy as np
import pandas as pd
import importlib
from helpers import broadcast_large_array
from constants import FieldPredictionType

# -------------------------------------------------------------------
# Parent Class
# -------------------------------------------------------------------

class DataLoader:
    def __init__(self, path, dims=2, verbose=False):
        self.path = path
        self.dims = dims
        self.verbose = verbose

    def to_csv(self, Y, X, time, columns):
        """Output CSV file named by timestamp, e.g. 1000.csv"""
        df = pd.DataFrame(np.concatenate((Y, X), axis=1), columns=columns)
        df.to_csv(str(time) + '.csv', index=False)

# -------------------------------------------------------------------
# Sequential Loader Factory Function
# -------------------------------------------------------------------

def load_data(args, **kwargs):
    """
    Factory function to create and use a dataloader instance based on args.dtype.
    
    Expects that the module in the dataloaders package is named exactly as args.dtype
    (for example, if args.dtype is "sst-binary", then the module is dataloaders/sst-binary.py)
    and that each module exposes its main dataloader class as "DataLoader".
    
    After instantiation, load_xyz() and load_multiple_timesteps() are called and their
    outputs are returned.
    
    Returns:
         X, Y, cv, x, y, z
    """
    module = importlib.import_module("dataloaders." + args.dtype)
    DataLoaderClass = module.DataLoader

    dl = DataLoaderClass(args, **kwargs)
    x, y, z = dl.load_xyz()
    X, Y, cv = dl.load_multiple_timesteps(
        args.write_interval, args.num_timesteps, target=args.target, cv=args.cluster_var
    )
    return X, Y, cv, x, y, z

# -------------------------------------------------------------------
# Parallel Loader Factory Function (with subsampling)
# -------------------------------------------------------------------

def parallel_load_data(args, subsample=True):
    """
    Parallel version of load_data.
    
    On the root process (rank 0), the data are loaded by calling load_data().
    Then, all data arrays are broadcast (in chunks) to all MPI processes.
    
    Optionally, the function performs subsampling on the time dimension:
      - It divides timesteps among processes.
      - Each process applies a subsample function (selected based on args.method).
      - Results are gathered back at the root and saved.
    
    Returns:
         X, Y, cv, x, y, z, subsample_results (the latter is only valid on rank 0)
    """
    from mpi4py import MPI
    import os
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # On the root process, ensure output directories (if needed) and load data
    if rank == 0:
        # (Assuming you have a helper function to check/create directories.)
        from helpers import check_and_create_dirs
        check_and_create_dirs(args.output_dir)
        check_and_create_dirs(args.plot_dir)
        X, Y, cv, x, y, z = load_data(args)
        fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}"
    else:
        X = Y = cv = x = y = z = None
        fileprefix = None

    # Broadcast the large arrays to all processes
    X  = broadcast_large_array(X,  comm, root=0)
    Y  = broadcast_large_array(Y,  comm, root=0)
    cv = broadcast_large_array(cv, comm, root=0)
    x  = broadcast_large_array(x,  comm, root=0)
    y  = broadcast_large_array(y,  comm, root=0)
    z  = broadcast_large_array(z,  comm, root=0)

    comm.Barrier()  # synchronize all processes

    if not subsample:
        return X, Y, cv, x, y, z, None

    # Divide timesteps among processes for subsampling
    local_timesteps = np.array_split(range(X.shape[0]), size)[rank]

    def get_subsample_fn():
        if args.method == "maxent":
            from algorithms import create_maxent_subsampler
            return create_maxent_subsampler(cv, args)
        elif args.method == "random":
            from algorithms import subsample_random
            return subsample_random
        elif args.method == "uips":
            # Example: define a placeholder subsample function for uips.
            def subsample_fn(X, n, t):
                # (Replace with actual uips subsampling implementation.)
                return np.arange(X[t].shape[0])
            return subsample_fn
        elif args.method == "full":
            return lambda X, n, t: np.arange(X.shape[1])
        else:
            raise ValueError(f"Unsupported sampling method: {args.method}")

    subsample_fn = get_subsample_fn()
    local_results = []
    for timestep in local_timesteps:
        indices = subsample_fn(X, args.num_samples, timestep)
        local_results.append((timestep, indices))

    # Gather subsampling results on the root process
    all_results = comm.gather(local_results, root=0)
    if rank == 0:
        # Flatten results
        all_results = [item for sublist in all_results for item in sublist]
        if args.method == "full":
            fileprefix = f"nxsl{args.nx}-nysl{args.ny}-nzsl{args.nz}-ns{args.num_samples}-window{args.window}"
        else:
            fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}-ns{args.num_samples}-window{args.window}"
        outfilename = f"subsampled_{fileprefix}.npz"
        outfile = os.path.join(args.output_dir, outfilename)
        np.savez(outfile, results=np.array(all_results, dtype=object))
        print(f"Results saved to {outfile}")
    else:
        all_results = None

    return X, Y, cv, x, y, z, all_results


def create_sequences(X, Y, args):
    """
    Create time sequences of a given window size (W) from the input arrays X and Y with specified overlap.

    X: [T * num_cubes, [X,Y,Z]-or-NSAMPLES, C] -> [B, W, [X,Y,Z]-or-NSAMPLES, C]
    Y: [T * num_cubes, [X,Y,Z]-or-NSAMPLES-or-1, C] -> [B, W, [X,Y,Z]-or-NSAMPLES-or-1, C]
    W: window size
    B: number of sequences (or total batch)
    """

    # Get sequence parameters from args
    overlap = args.overlap
    window_size = args.window
    field_prediction_type = args.field_prediction_type

    # Get dimensions of X & Y
    if args.method == "full":
        nt_nhyp, nx, ny, nz, nvars_X = X.shape
    else:
        nt_nhyp, nsamples, nvars_X = X.shape

    try:
        nt = nt_nhyp // args.num_hypercubes
    except ZeroDivisionError:
        raise ValueError("Invalid number of hypercubes; must be non-zero.")

    if field_prediction_type == FieldPredictionType.FULL:
        nx, ny, nz, nvars_Y = Y.shape[1:]
    else:
        nvars_Y = Y.shape[-1]

    # Determine sequence info
    stride = window_size - overlap
    print(f"stride: {stride}")
    assert stride > 0, f"window_size ({window_size}) must be > overlap ({overlap})"

    num_seqs_per_cube = (nt - window_size) // stride + 1
    print(f"num_seqs_per_cube {num_seqs_per_cube}; nt {nt}; window_size {window_size}")
    num_sequences = num_seqs_per_cube * args.num_hypercubes

    # Initialize sequence arrays
    if args.method == "full":
        X_sequences = np.zeros((num_sequences, window_size, nx, ny, nz, nvars_X))
    else:
        X_sequences = np.zeros((num_sequences, window_size, nsamples, nvars_X))

    if field_prediction_type == FieldPredictionType.GLOBAL:  # global quantity prediction
        Y_sequences = np.zeros((num_seqs_per_cube, window_size, 1, nvars_Y))
    elif field_prediction_type == FieldPredictionType.LOCAL:  # local field prediction
        if args.method == "full":
            raise Exception("For baseline full field input, prediction cannot be subsampled. Change `args.target`.")
        Y_sequences = np.zeros((num_sequences, window_size, nsamples, nvars_Y))
    elif field_prediction_type == FieldPredictionType.FULL:  # full field prediction
        Y_sequences = np.zeros((num_sequences, window_size, nx, ny, nz, nvars_Y))
    else:
        raise Exception("Enter a valid `args.target`.")

    # Get and store sequence data
    for j in range(args.num_hypercubes):
        for i in range(num_seqs_per_cube):
            start_index = (j * num_seqs_per_cube) + (i * stride)
            X_sequences[i] = X[start_index:start_index + window_size].reshape(X_sequences.shape[1:])
            Y_sequences[i] = Y[start_index:start_index + window_size].reshape(Y_sequences.shape[1:])

    return X_sequences, Y_sequences
