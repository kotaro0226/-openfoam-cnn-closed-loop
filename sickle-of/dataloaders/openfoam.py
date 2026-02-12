import numpy as np
from dataloaders import DataLoader
from constants import FieldPredictionType

# We do this so that users who are not using OpenFOAM need not install fluidfoam
try:
    from fluidfoam import readscalar, readvector, readforce
except ImportError:
    print("WARNING: fluidfoam not able to be loaded")

class DataLoaderOF(DataLoader):

    def __init__(self, args, **kwargs):
        super().__init__(args.path, args.dims, args.verbose)

    def load_forces(self, write_interval=100):
        forces = readforce(self.path, time_name='0', name='forces')
        # Drag force is composed of both a viscous and pressure components
        time = forces[:, 0]
        drag = forces[:, 1] + forces[:, 2]
        return time[::write_interval], drag[::write_interval]

    def load_xyz(self):
        x, y, z = readvector(self.path, '0', 'C.gz')
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        z = np.expand_dims(z, axis=1)
        return x, y, z

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv):
        p = readscalar(self.path, str(write_interval), 'p.gz')
        num_pts = p.shape[0]
        
        print(num_pts, num_timesteps)

        p = np.zeros((num_timesteps, num_pts))
        u = np.zeros((num_timesteps, num_pts))
        v = np.zeros((num_timesteps, num_pts))
        w = np.zeros((num_timesteps, num_pts))
        wz = np.zeros((num_timesteps, num_pts))

        for i, ts in enumerate(range(write_interval, write_interval*num_timesteps+1, write_interval)):
            print(i, ts)
            p[i, :] = readscalar(self.path, str(ts), 'p.gz')
            u[i, :], v[i, :], w[i, :] = readvector(self.path, str(ts), 'U.gz')
            _, _, wz[i, :] = readvector(self.path, str(ts), 'vorticity.gz')
    
        params = {'p': p, 'wz': wz, 'pwz': np.stack((p, wz), axis=1)}

        if target == 'drag':
            params['drag'] = self.load_forces()[1].reshape(-1, 1)

        if self.dims == 2:
            X = np.stack((u, v), axis=-1)
        elif self.dims == 3:
            X = np.stack((u, v, w), axis=-1)
        else:
            raise ValueError("dims must be either 2 or 3")
        Y = params[target]
        Y = np.expand_dims(Y, axis=-1)  # e.g., (100, 10000) => (100, 10000, 1)
        cv = params[cv[0]]

        return X, Y, cv

DataLoader = DataLoaderOF

if __name__ == "__main__":
    # Simple test code for DataLoaderOF.
    # Make sure to set test_path to a valid OpenFOAM case directory.
    test_path = "path/to/openfoam/case"
    dl = DataLoaderOF(test_path, dims=2, verbose=True)
    try:
        x, y, z = dl.load_xyz()
        print("XYZ shapes:", x.shape, y.shape, z.shape)
        X, Y, cv = dl.load_multiple_timesteps(write_interval=100, num_timesteps=5, target='p', cv=['wz'])
        print("Data shapes:", X.shape, Y.shape, cv.shape)
    except Exception as e:
        print("Error during OpenFOAM DataLoader test:", e)
