import os
import glob
import numpy as np
import re
from dataloaders import DataLoader
from hypercubes.hypercube_manager import HypercubeHandler
from helpers import get_1Dgrid, check_data
from helpers import setup_rank_print
setup_rank_print()

class DataLoaderSSTBinary(DataLoader):

    def __init__(self, args, extractor=None):
        super().__init__(args.path, args.dims, args.verbose)
        self.args = args
        self.path = args.path
        if extractor:
            # Instead of assigning the extractor function directly,
            # instantiate a HypercubeHandler.
            dims_full = (self.args.nx, self.args.ny, self.args.nz)
            dims_sl   = (self.args.nxsl, self.args.nysl, self.args.nzsl)
            nskips = (self.args.nxskip, self.args.nyskip, self.args.nzskip)
            self.hypercube_handler = HypercubeHandler(
                self.args.hypercubes, dims_full, dims_sl,
                self.args.nbytes, self.args.num_hypercubes, 
                num_clusters=self.args.num_clusters, nskips=nskips, use_parallel=True  # or other selector_kwargs as needed
            )

    def _extract_times(self, file_names):
        pattern = r'_([0-9]+\.[0-9]+)$'
        time_pattern = re.compile(pattern)
        unique_times = set()
        for name in file_names:
            match = time_pattern.search(name)
            if match:
                unique_times.add(float(match.group(1)))
        return sorted(unique_times)

    def load_xyz(self):
        x = get_1Dgrid(self.args.Lh, self.args.nx-2, self.args.nxoffset, self.args.nxsl, self.args.nxskip)
        if self.args.gravity == 'y':
            y = get_1Dgrid(self.args.Lv, self.args.ny, self.args.nyoffset, self.args.nysl, self.args.nyskip)
            z = get_1Dgrid(self.args.Lh, self.args.nz, self.args.nzoffset, self.args.nzsl, self.args.nzskip)
        elif self.args.gravity == 'z':
            y = get_1Dgrid(self.args.Lh, self.args.ny, self.args.nyoffset, self.args.nysl, self.args.nyskip)
            z = get_1Dgrid(self.args.Lv, self.args.nz, self.args.nzoffset, self.args.nzsl, self.args.nzskip)
        else:
            raise Exception("Gravity should be defined")
        self.nx, self.ny, self.nz = x.shape[0], y.shape[0], z.shape[0]
        self.num_pts = self.nx * self.ny * self.nz
        print('num_pts:', self.num_pts)
        return x, y, z

    def _load_and_process_hypercubes(self, var, ts, file_path, hypercubeIDs):
        print(f'Loading file using hypercube IDs for {var} at timestep {ts:0.6f}')
        check_data(file_path, self.args.nx, self.args.ny, self.args.nz, self.args.nbytes)
        return self.hypercube_handler.load_hypercubes(file_path, hypercubeIDs)

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv, file_filter='*_*'):
        self.path = os.path.dirname(self.path)
        file_names = glob.glob(os.path.join(self.path, file_filter))
        file_names = [os.path.basename(f) for f in file_names]
        print('Files:', sorted(file_names))

        x_labels = self.args.input_vars  # e.g., ['u', 'v', 'w', 'r']
        y_labels = self.args.output_vars  # e.g., ['p']
        cv_labels = self.args.cluster_var   # e.g., ['pv']
        t_labels = self._extract_times(file_names)
        print('Available timesteps (t_labels):', t_labels)

        if self.args.timesteps:
            desired_timesteps = sorted(self.args.timesteps)
            tolerance = 1e-6
            filtered_t_labels = []
            for dt in desired_timesteps:
                matches = [ts for ts in t_labels if abs(ts - dt) < tolerance]
                if matches:
                    filtered_t_labels.append(matches[0])
                else:
                    print(f"Warning: Timestep {dt} not found in the data.")
            t_labels = filtered_t_labels
            print('Filtered timesteps to load:', t_labels)

        num_timesteps = len(t_labels)

        # When extracting hypercubes, each loaded file returns an array of shape:
        num_points = self.args.num_hypercubes * self.num_pts  # number of hypercubes per file
        X = np.zeros((num_timesteps, num_points, len(x_labels)))
        Y = np.zeros((num_timesteps, num_points, len(y_labels)))
        cv_arr = np.zeros((num_timesteps, num_points, len(cv_labels)))


        for j, ts in enumerate(t_labels):
            """
            TODO:
            This for loop is embarrassingly parallel.
            Same as subsampling for loop.
            """
            file_paths = [os.path.join(self.path, f'{v}_{ts:0.6f}') for v in cv_labels]
            hypercube_ids = self.hypercube_handler.extract_ids(file_paths)

            # Save hypercube_ids per timestep
            out_file = os.path.join(self.args.output_dir, f"hypercube_ids_{ts:0.6f}.npz")
            np.savez(out_file, hypercube_ids=hypercube_ids)
            print(f"Hypercube IDs for timestep {ts} saved to {out_file}")

            for i, var in enumerate(cv_labels):
                file_path = os.path.join(self.path, f'{var}_{ts:0.6f}')
                cv_arr[j, :, i] = self._load_and_process_hypercubes(var, ts, file_path, hypercube_ids)
            for i, var in enumerate(x_labels):
                file_path = os.path.join(self.path, f'{var}_{ts:0.6f}')
                X[j, :, i] = self._load_and_process_hypercubes(var, ts, file_path, hypercube_ids)
            for i, var in enumerate(y_labels):
                file_path = os.path.join(self.path, f'{var}_{ts:0.6f}')
                Y[j, :, i] = self._load_and_process_hypercubes(var, ts, file_path, hypercube_ids)

        return X, Y, cv_arr

DataLoader = DataLoaderSSTBinary

if __name__ == "__main__":
    # Simple test code for DataLoaderSSTBinary.
    class DummyArgs:
        def __init__(self):
            self.path = "path/to/sst/binary/data"
            self.dims = 3
            self.Lh = 10
            self.Lv = 5
            self.nx = 100
            self.ny = 50
            self.nz = 20
            self.nxoffset = 0
            self.nyoffset = 0
            self.nzoffset = 0
            self.nxsl = 1
            self.nysl = 1
            self.nzsl = 1
            self.nxskip = 1
            self.nyskip = 1
            self.nzskip = 1
            self.nbytes = 4
            self.input_vars = ['u', 'v', 'w', 'r']
            self.output_vars = ['p']
            self.cluster_var = ['pv']
            self.timesteps = None
            # Flag to enable hypercube extraction and its parameters:
            self.subsample_hypercube = True
            self.nxsl = 32
            self.nysl = 32
            self.nzsl = 32
            self.subsample_method = 'random'
            self.num_hypercubes = 16  # for example

    args = DummyArgs()
    dl = DataLoaderSSTBinary(args)
    # The following call would load data and, if enabled, extract hypercubes.
    X, Y, cv_arr = dl.load_multiple_timesteps(write_interval=1, num_timesteps=1, target=None, cv=None)
