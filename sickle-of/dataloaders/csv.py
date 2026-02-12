import os
import numpy as np
import pandas as pd
from dataloaders import DataLoader

class DataLoaderCSV(DataLoader):
    
    #def __init__(self, args, extractor=None, verbose=False, prefix='cylinder_t', zwidth=4, dims=2):
    def __init__(self, args, **kwargs):
        super().__init__(args.path, args.dims, args.verbose)
        self.prefix = prefix
        self.zwidth = zwidth
    
    def load_xyz(self):
        dfpath = os.path.join(self.path, f'{self.prefix}{str(1).zfill(self.zwidth)}.csv')
        data = pd.read_csv(dfpath)
        x = data["X"].to_numpy()
        y = data["Y"].to_numpy()
        self.num_points = len(x)
        return x, y

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv):
        num_pts = self.num_points
        x_labels = ["dudx", "dudy", "dvdx", "dvdy", "vortZ"]
        Y = np.zeros((num_timesteps, num_pts))
        X = np.zeros((num_timesteps, num_pts, len(x_labels)))
        cv_arr = np.zeros((num_timesteps, num_pts))
        
        for i, ts in enumerate(range(write_interval, write_interval*num_timesteps+1, write_interval)):
            dfpath = os.path.join(self.path, f'cylinder_t{str(i+1).zfill(self.zwidth)}.csv')
            data = pd.read_csv(dfpath)
            tke_val = abs(data["TKE"].to_numpy())
            tke_0 = np.where(tke_val <= 1.0e-9)[0]
            tke_val[tke_0] = 1.0e-8
            Y[i, :] = np.log(tke_val)
            X[i, :] = data[x_labels].to_numpy()
            cv_arr[i, :] = data["vortZ"].to_numpy()

        return X, Y, cv_arr

DataLoader = DataLoaderCSV

if __name__ == "__main__":
    # Simple test code for DataLoaderCSV.
    test_path = "path/to/csv/files"
    dl = DataLoaderCSV(test_path, verbose=True)
    try:
        x, y = dl.load_xyz()
        print("Loaded XYZ:", len(x), len(y))
        X, Y, cv = dl.load_multiple_timesteps(write_interval=100, num_timesteps=5, target='p', cv=['vortZ'])
        print("Data shapes:", X.shape, Y.shape, cv.shape)
    except Exception as e:
        print("Error during CSV DataLoader test:", e)
