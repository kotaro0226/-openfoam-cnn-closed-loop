import os
import numpy as np
import glob
import time

from dataloaders import DataLoader
from helpers import check_data

# Import your hypercube handler if hypercube extraction is enabled.
# For example, if you implemented it in hypercubes/hypercube_manager.py:
try:
    from hypercubes.hypercube_manager import HypercubeHandler
except ImportError:
    HypercubeHandler = None


class GESTSDataLoader(DataLoader):

    def __init__(self, args, extractor=None):
        super().__init__(args.path, dims=args.dims)
        self.args = args
        self.path = args.path
        self.grid_size = (args.nx, args.ny, args.nz)
        self.subcube_size = (args.nxsl, args.nysl, args.nzsl)
        self.subcube_origin = (0, 0, 0)  # Default subcube origin
        self.varmap = dict(enstrophy='enst', velocity='uvw', pressure='p', dissipation='diss')
        self.verbose = getattr(args, 'verbose', False)
        nskips = (self.args.nxskip, self.args.nyskip, self.args.nzskip)

        # Use the extractor function if provided.
        if extractor:
            self.hypercube_handler = HypercubeHandler(
                method=args.hypercubes,
                dims_full=self.grid_size,
                dims_sl=self.subcube_size,
                nbytes=args.nbytes,
                num_hypercubes=args.num_hypercubes,
                num_clusters=self.args.num_clusters,
                nskips=nskips,
                use_parallel=True
            )
        else:
            self.hypercube_handler = None

    def _get_filenames(self, variable, verbose=True):
        """Retrieve sorted filenames for a given variable."""
        var_prefix = self.varmap.get(variable, variable)
        var_path = os.path.join(self.path, variable)
        file_pattern = os.path.join(var_path, f'cube_{var_prefix}.*')
        files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('.')[-1]))
        if self.verbose:
            print(f"Found {len(files)} files for variable: {variable}")
        return files

    def _extract_times(self, file_names):
        """Extract available timesteps from file names."""
        return sorted(set(int(f.split('.')[-1]) for f in file_names))

    def _read_binary_cube(self, filename, has_vector=False):
        """Read a binary cube file using memory mapping for large files and extract a subcube."""
        dtype = np.float32  # Assuming 32-bit floating point precision
        shape = (3, *self.grid_size) if has_vector else self.grid_size  # Use full grid size

        if self.verbose:
            print(f"Memory-mapping file: {filename} with expected shape: {shape}")

        data = np.memmap(filename, dtype=dtype, mode='r', shape=shape, order='F')
        if self.verbose:
            print(f"Loaded file {filename}")

        # Extract subcube region
        x0, y0, z0 = self.subcube_origin
        x1, y1, z1 = x0 + self.subcube_size[0], y0 + self.subcube_size[1], z0 + self.subcube_size[2]
        if has_vector:
            subcube = data[:, x0:x1, y0:y1, z0:z1]
        else:
            subcube = data[x0:x1, y0:y1, z0:z1]

        if self.verbose:
            print(f"Extracted sub-region shape: {subcube.shape}")

        # For vector data, reshape so that channels become features.
        return subcube.reshape(3, -1).T if has_vector else subcube.reshape(-1)

    # --- New helper methods for hypercube extraction ---
    def _get_hypercube_IDs(self, variable, ts):
        """
        Compute hypercube IDs for a given variable at timestep ts.
        We use one variable’s file as the basis.
        """
        var_prefix = self.varmap.get(variable, variable)
        file_path = os.path.join(self.path, variable, f"cube_{var_prefix}.{ts}")
        # We pass a list containing this file path to the hypercube handler.
        return self.hypercube_handler.extract_ids([file_path])

    def _load_and_process_hypercubes(self, variable, ts, hypercubeIDs):
        var_prefix = self.varmap.get(variable, variable)
        # Convert ts to int so that a float like 0.0 becomes 0
        file_path = os.path.join(self.path, variable, f"cube_{var_prefix}.{int(ts)}")
        
        # Determine channels if needed and run check_data, etc.
        channels = 3 if variable == 'velocity' else 1
        check_data(file_path, self.args.nx, self.args.ny, self.args.nz, self.args.nbytes, channels=channels)
        
        return self.hypercube_handler.load_hypercubes(file_path, hypercubeIDs, has_vector=channels==3)

    def load_xyz(self):
        """Generate 1D x, y, z coordinate arrays instead of full 3D grids."""
        if self.verbose:
            print("Generating coordinate arrays...")
        x = np.linspace(0, 1, self.subcube_size[0])
        y = np.linspace(0, 1, self.subcube_size[1])
        z = np.linspace(0, 1, self.subcube_size[2])
        self.num_pts = np.prod(self.subcube_size)
        if self.verbose:
            print(f"Coordinate arrays generated successfully. num_pts: {self.num_pts}")
        return x, y, z

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv, file_filter='*_*'):
        """Load multiple variables from different timesteps, extracting only the subcube region.
           If hypercube extraction is enabled, use that instead of reading the full subcube.
        """
        x_labels = self.args.input_vars
        y_labels = self.args.output_vars
        cv_labels = self.args.cluster_var

        # Extract timesteps from available files (assuming all variables share timesteps)
        file_names = self._get_filenames(x_labels[0])
        t_labels = self._extract_times(file_names)
        print('Available timesteps (t_labels):', t_labels)

        if self.args.timesteps:
            desired_timesteps = sorted(self.args.timesteps)
            # Only keep timesteps that are both desired and present in t_labels.
            t_labels = [int(ts) for ts in desired_timesteps if ts in t_labels]
            print('Filtered timesteps to load:', t_labels)

        num_timesteps = len(t_labels)

        # When extracting hypercubes, each loaded file returns an array of shape:
        num_points = self.args.num_hypercubes * self.num_pts  # number of hypercubes per file
        # Adjust the channel dimensions as needed (here we assume extra space for velocity channels)
        X = np.zeros((num_timesteps, num_points, len(x_labels) + 2))
        Y = np.zeros((num_timesteps, num_points, len(y_labels)))
        cv_arr = np.zeros((num_timesteps, num_points, len(cv_labels)))

        for j, ts in enumerate(t_labels[:num_timesteps]):
            if self.hypercube_handler is not None:
                # --- Use hypercube extraction ---
                # Compute hypercube IDs using one of the variables (for example, the first cv variable)
                hypercubeIDs = self._get_hypercube_IDs(cv_labels[0], ts)
                print(f"timestep {ts} hypercubeIDs {hypercubeIDs}")
                for i, var in enumerate(cv_labels):
                    cv_arr[j, :, i] = self._load_and_process_hypercubes(var, ts, hypercubeIDs)

                dest_col = 0
                for var in x_labels:
                    subcube = self._load_and_process_hypercubes(var, ts, hypercubeIDs)
                    if var == 'velocity':
                        # If velocity has multiple channels, spread them across consecutive channels.
                        #subcube = subcube.reshape(-1, 3)
                        X[j, :, dest_col:dest_col+3] = subcube
                        dest_col += 3
                    else:
                        X[j, :, dest_col] = subcube
                        dest_col += 1

                for i, var in enumerate(y_labels):
                    Y[j, :, i] = self._load_and_process_hypercubes(var, ts, hypercubeIDs)

            else:
                # --- Fallback to full subcube extraction ---
                for i, var in enumerate(cv_labels):
                    var_prefix = self.varmap.get(var, var)
                    cv_arr[j, :, i] = self._read_binary_cube(os.path.join(self.path, var, f"cube_{var_prefix}.{ts}"),
                                                              has_vector=False)
                for var in x_labels:
                    var_prefix = self.varmap.get(var, var)
                    subcube = self._read_binary_cube(os.path.join(self.path, var, f"cube_{var_prefix}.{ts}"),
                                                     has_vector=(var == 'velocity'))

                    dest_col = 0
                    if var == 'velocity':
                        X[j, :, dest_col:dest_col+3] = subcube
                        dest_col += 3
                    else:
                        X[j, :, dest_col] = subcube
                        dest_col += 1

                for i, var in enumerate(y_labels):
                    var_prefix = self.varmap.get(var, var)
                    Y[j, :, i] = self._read_binary_cube(os.path.join(self.path, var, f"cube_{var_prefix}.{ts}"),
                                                       has_vector=False)

        return X, Y, cv_arr


# For backward compatibility.
DataLoader = GESTSDataLoader

if __name__ == "__main__":
    # Example usage for testing:
    dataset_path = "/path/to/phy_cube_data/2048"
    # Make sure your args object includes the necessary fields, for example:
    class DummyArgs:
        def __init__(self):
            self.path = dataset_path
            self.dims = 3
            self.nx = 128
            self.ny = 128
            self.nz = 128
            self.nxsl = 32
            self.nysl = 32
            self.nzsl = 32
            self.input_vars = ['velocity', 'enstrophy']
            self.output_vars = ['pressure']
            self.cluster_var = ['dissipation']
            self.hypercubes = True
            self.nbytes = 4
            self.num_hypercubes = 16
            self.subsample_method = 'random'
            self.verbose = True

    args = DummyArgs()
    loader = GESTSDataLoader(args)
    x, y, z = loader.load_xyz()
    print(f"Loaded coordinate arrays: x({len(x)}), y({len(y)}), z({len(z)})")

    X, Y, cv_arr = loader.load_multiple_timesteps(write_interval=1, num_timesteps=8, target=None, cv=None)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, cv_arr shape: {cv_arr.shape}")
