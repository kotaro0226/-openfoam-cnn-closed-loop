import os
import re
import glob
import numpy as np
from dataloaders import DataLoader

class DataLoaderNPZ(DataLoader):

    #def __init__(self, args, extractor=None, verbose=False, prefix='cylinder_t', zwidth=4, dims=2):
    def __init__(self, args, **kwargs):
        super().__init__(path, dims, verbose)
    
    def _extract_times(self, file_names):
        time_pattern = re.compile(r'_t(\d+\.\d+)\.npz')
        unique_times = set()
        for name in file_names:
            match = time_pattern.search(name)
            if match:
                unique_times.add(float(match.group(1)))
        return sorted(unique_times)

    def load_xyz(self):
        data = np.load(self.path, allow_pickle=True)
        x, y, z = data['x'], data['y'], data['z']
        self.nx, self.ny, self.nz = x.shape[0], y.shape[0], z.shape[0]
        self.nskip = data['args'].item().nxskip
        self.num_pts = self.nx * self.ny * self.nz
        print('num_pts:', self.num_pts)
        return x, y, z
    
    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv, file_filter='*t*', label='PVsample'):
        self.path = os.path.dirname(self.path)
        file_names = glob.glob(os.path.join(self.path, file_filter))
        file_names = [os.path.basename(f) for f in file_names] 
        print('files:', sorted(file_names))
        x_labels = ['u', 'v', 'w', 'r']
        t_labels = self._extract_times(file_names)
        print('t_labels:', t_labels)
        
        num_timesteps = len(t_labels)
        num_pts = self.num_pts
        Y = np.zeros((num_timesteps, num_pts))
        X = np.zeros((num_timesteps, num_pts, len(x_labels)))
        cv_arr = np.zeros((num_timesteps, num_pts))

        # Read neural net inputs / observations
        for i, var in enumerate(x_labels):
            for j, ts in enumerate(t_labels):
                file_path = os.path.join(self.path, f'{var}_{label}-nx{self.nx}ny{self.ny}nz{self.nz}_nskip{self.nskip}_t{ts:0.6f}.npz')
                data = np.load(file_path)
                box = data["datacube"]
                X[j, :, i] = box.reshape(-1)

        # Read neural net outputs / targets
        y_labels = ['p']  # pressure
        for j, ts in enumerate(t_labels):
            file_path = os.path.join(self.path, f'{y_labels[0]}_{label}-nx{self.nx}ny{self.ny}nz{self.nz}_nskip{self.nskip}_t{ts:0.6f}.npz')
            data = np.load(file_path)
            box = data["datacube"]
            Y[j, :] = box.reshape(-1)

        # Read cluster variable for MaxEnt analysis
        cv_labels = ['pv']  # potential vorticity
        for j, ts in enumerate(t_labels):
            file_path = os.path.join(self.path, f'{cv_labels[0]}_{label}-nx{self.nx}ny{self.ny}nz{self.nz}_nskip{self.nskip}_t{ts:0.6f}.npz')
            data = np.load(file_path)
            box = data["datacube"]
            cv_arr[j, :] = box.reshape(-1)

        return X, Y, cv_arr

DataLoader = DataLoaderNPZ

if __name__ == "__main__":
    # Simple test code for DataLoaderNPZ.
    test_path = "path/to/npz_file.npz"
    dl = DataLoaderNPZ(test_path, verbose=True)
    try:
        x, y, z = dl.load_xyz()
        print("XYZ shapes:", x.shape, y.shape, z.shape)
        X, Y, cv = dl.load_multiple_timesteps(write_interval=100, num_timesteps=5, target='p', cv=['pv'])
        print("Data shapes:", X.shape, Y.shape, cv.shape)
    except Exception as e:
        print("Error during NPZ DataLoader test:", e)
