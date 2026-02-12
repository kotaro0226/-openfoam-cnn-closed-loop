import os
import numpy as np
import glob

from dataloaders import DataLoader
from helpers import check_data

# Import your hypercube handler if hypercube extraction is enabled.
# For example, if you implemented it in hypercubes/hypercube_manager.py:
try:
    from hypercubes.hypercube_manager import HypercubeHandler
except ImportError:
    HypercubeHandler = None

num_cubes = 8

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

        # Save copies of the variable lists so they don’t get modified elsewhere
        self.input_vars = list(args.input_vars)
        self.output_vars = list(args.output_vars)
        self.cluster_var = list(args.cluster_var)

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

    def _get_filenames(self, variable, cubeid=0, verbose=True):
        var_prefix = self.varmap.get(variable, variable)
        # For the 8192 dataset, use the cubeid folder
        var_path = os.path.join(self.path, str(cubeid))
        file_pattern = os.path.join(var_path, f'cube_{var_prefix}.*')
        print("reading", file_pattern)
        files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('.')[-1]))
        if self.verbose:
            print(f"Found {len(files)} files for variable: {variable} in folder: {var_path}")
        return files

    def _extract_times(self, file_names):
        """Extract available timesteps from file names."""
        return sorted(set(int(f.split('.')[-1]) for f in file_names))

    def _read_binary_cube(self, filename, cubeid=0, has_vector=False):
        """
        Read a binary cube file with support for multiple cubes within a single file.
        
        Parameters:
          filename (str): Path to the binary cube file.
          cubeid (int): Cube identifier from 0 to 7 (for the 8192 case). For 2048 datasets, default to 0.
          has_vector (bool): True if data is a vector (e.g., velocity) with 3 components.
        
        Returns:
          subcube: Flattened array of the extracted subcube.
        """
        dtype = np.float32
        # Determine base grid dimensions.
        base_shape = self.grid_size  # e.g., (8192, 8192, 8192) for the entire dataset.

        # For 8192³, assume 8 cubes along X.
        #num_cubes_in_file = 8 if base_shape[0] > self.subcube_size[0] else 1
        cube_x_size = base_shape[0] // num_cubes
        cube_shape = (cube_x_size, base_shape[1], base_shape[2])
        #print(f"For cubeid {cubeid}: grid_size={self.grid_size}, cube_x_size={cube_x_size}")

        itemsize = np.dtype(dtype).itemsize
        bytes_per_cube = np.prod(cube_shape) * itemsize
        if has_vector:
            bytes_per_cube *= 3  # For U-V-W components stored sequentially.
            
        cube_offset = cubeid * bytes_per_cube

        if self.verbose:
            print(f"Reading file {filename} at cubeid {cubeid} with offset {cube_offset}")
            print(f"Expected cube shape: {cube_shape}, vector: {has_vector}")

        # Set up shape for memmap.
        if has_vector:
            # Data is stored as (3, X, Z, Y)
            memmap_shape = (3, *cube_shape)
        else:
            # Data is stored as (X, Z, Y)
            memmap_shape = cube_shape

        data = np.memmap(filename, dtype=dtype, mode='r', offset=cube_offset, shape=memmap_shape, order='F')

        if has_vector:
            # For velocity, data is stored as (3, X, Z, Y).
            # Convert the memmapped data to a writable array.
            data = np.array(data, copy=True)
            for comp in range(3):
                data[comp, :, :, :] = np.transpose(data[comp, :, :, :], (0, 2, 1))
        else:
            data = np.transpose(data, (0, 2, 1))  # Now (X, Y, Z)
                
        # Compute the subcube origin based on fileid and cubeid.
        # For example, call a helper function to convert these indices to an origin.
        x0, y0, z0 = self.compute_subcube_origin(filename, cubeid)
        x1 = x0 + self.subcube_size[0]
        y1 = y0 + self.subcube_size[1]
        z1 = z0 + self.subcube_size[2]
        #print(f"Computed x0={x0}, x1={x1}")

        if self.verbose:
            print(f"Extracting subcube indices: x:{x0}-{x1}, y:{y0}-{y1}, z:{z0}-{z1}")

        # Extract subcube region.
        if has_vector:
            subcube = data[:, x0:x1, y0:y1, z0:z1]
            # Reshape so that each row is a feature vector (3 channels).
            subcube = subcube.reshape(3, -1).T
        else:
            subcube = data[x0:x1, y0:y1, z0:z1]
            subcube = subcube.reshape(-1)

        if self.verbose:
            print(f"Final subcube shape: {subcube.shape}")
        
        return subcube

    def _get_hypercube_IDs(self, variable, ts):
        """
        Compute hypercube IDs for a given variable and a simulated timestep (cubeid).
        For the 8192 dataset, use the cubeid folder (e.g., "0", "1", etc.) since the files
        are stored in that directory.
        """
        var_prefix = self.varmap.get(variable, variable)
        file_path = os.path.join(self.path, str(ts), f"cube_{var_prefix}.{ts}")

        if self.verbose:
            print(f"_get_hypercube_IDs: Using file path: {file_path}")
        # Pass a list containing the file path to the hypercube handler.
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

        print("num timesteps", num_timesteps)
        print("x_labels:", self.input_vars)
        print("y_labels:", self.output_vars)
        print("cv_labels:", self.cluster_var)

        num_points = self.num_pts * num_cubes
        X = np.zeros((num_timesteps, num_points, len(self.args.input_vars) + 2))
        Y = np.zeros((num_timesteps, num_points, len(self.args.output_vars)))
        cv_arr = np.zeros((num_timesteps, num_points, len(self.args.cluster_var)))

        for cubeid in range(num_timesteps):
            print(f'\ncubeid: {cubeid}')
            X[cubeid], Y[cubeid], cv_arr[cubeid] = self.load_snapshot(cubeid=cubeid)
            print("X.shape:", X.shape, "Y.shape:", Y.shape, "cv_arr.shape:", cv_arr.shape)

        return X, Y, cv_arr

    def load_snapshot(self, cubeid=0):
        """
        Load a static snapshot for the configured input, output, and cluster variables.
        
        This function is intended to replace the timestep-based loader for datasets
        that represent a single static snapshot (like the 8192³ dataset).
        
        Parameters:
          target: Unused placeholder parameter (or could be used to select a specific target variable).
          cubeid (int): The cube identifier within the file.
          
        Returns:
          X: Data for input variables, shape (num_points, n_input_vars)
          Y: Data for output variables, shape (num_points, n_output_vars)
          cv: Data for cluster variable(s), shape (num_points, n_cv_vars)
        """
        # List of input, output, and cluster variable names from your args.
        # Make a copy b/c the variables were getting changed unexpectedly
        x_labels = [str(x) for x in self.input_vars]
        y_labels = [str(x) for x in self.output_vars]
        cv_labels = [str(x) for x in self.cluster_var]

        # Here we assume that each variable's file is found by _get_filenames, and we pick the first file.
        X_list = []
        for var in x_labels:
            files = self._get_filenames(var, cubeid=cubeid)
            if not files:
                raise ValueError("No files found for variable: " + var)
            var_data_list = []
            for f in files:
                print(f)
                # Determine if this variable represents vector data (velocity) or scalars
                if var == 'velocity':
                    data = self._read_binary_cube(f, cubeid=cubeid, has_vector=True)
                else:
                    data = self._read_binary_cube(f, cubeid=cubeid, has_vector=False)
                var_data_list.append(data)
            # Concatenate the data from all files along the sample axis.
            aggregated = np.concatenate(var_data_list, axis=0)
            X_list.append(aggregated)
        # Combine the variables as columns
        X = np.column_stack(X_list)

        Y_list = []
        for var in y_labels:
            files = self._get_filenames(var, cubeid=cubeid)
            if not files:
                raise ValueError("No files found for variable: " + var)
            var_data_list = []
            for f in files:
                print(f)
                data = self._read_binary_cube(f, cubeid=cubeid, has_vector=False)
                var_data_list.append(data)
            aggregated = np.concatenate(var_data_list, axis=0)
            Y_list.append(aggregated)
        Y = np.column_stack(Y_list)

        cv_list = []
        for var in cv_labels:
            files = self._get_filenames(var, cubeid=cubeid)
            if not files:
                raise ValueError("No files found for variable: " + var)
            var_data_list = []
            for f in files:
                print(f)
                data = self._read_binary_cube(f, cubeid=cubeid, has_vector=False)
                var_data_list.append(data)
            aggregated = np.concatenate(var_data_list, axis=0)
            cv_list.append(aggregated)
        cv = np.column_stack(cv_list)

        return X, Y, cv

    def compute_subcube_origin(self, filename, cubeid):
        """
        Compute the starting indices (x0, y0, z0) of the subcube to extract.
        
        If the file is already located in a cube folder (e.g. "/.../8192/1/"),
        then we assume the data is local to that cube and set x0 to 0.
        Otherwise, we compute x0 = cubeid * N_cube.
        """
        # Extract the fileid from the filename.
        base = os.path.basename(filename)  # e.g., "cube_enst.40"
        try:
            fileid = int(base.split('.')[-1])
        except Exception as e:
            raise ValueError("Failed to extract fileid from filename: " + filename) from e

        N_cube = self.grid_size[0] // num_cubes  # e.g., 2048//8 = 256
        # Compute Y and Z block indices using fileid.
        j = (fileid // num_cubes) % num_cubes
        k = fileid % num_cubes

        # Check if the file is in a directory that is just a cube ID.
        parent = os.path.basename(os.path.dirname(filename))
        if parent.isdigit():
            # The file is already in a cube folder. Do not add an extra X offset.
            x0 = 0
        else:
            # Otherwise, shift in X by the cubeid.
            x0 = cubeid * N_cube

        y0 = j * N_cube
        z0 = k * N_cube

        # Optionally add any base offset.
        if hasattr(self, 'subcube_origin'):
            base_origin = self.subcube_origin
            x0 += base_origin[0]
            y0 += base_origin[1]
            z0 += base_origin[2]

        return int(x0), int(y0), int(z0)



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
