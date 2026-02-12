# dataloaders/of-face-npz.py
from __future__ import annotations
import numpy as np
from dataloaders import DataLoader as _Base

class DataLoaderOFFaceNPZ(_Base):
    def __init__(self, args, **kwargs):
        super().__init__(args.path, dims=args.dims, verbose=args.verbose)

        self.xy_npz   = getattr(args, "xy_npz", None)
        self.maps_npz = getattr(args, "maps_npz", None)
        self.direction = getattr(args, "direction", "x")
        self.cv_index  = int(getattr(args, "cv_index", 2))
        self.args = args

        if self.xy_npz is None or self.maps_npz is None:
            raise ValueError("dtype=of-face-npz requires --xy_npz and --maps_npz")

        if self.direction not in ("x","y","z"):
            raise ValueError("--direction must be x/y/z")

        m = np.load(self.maps_npz, allow_pickle=True)
        self._grid = self._internal_face_grid(m, self.direction)  # (nx,ny,nz)
        self.nxsl, self.nysl, self.nzsl = self._grid.shape
        self.num_pts = self.nxsl * self.nysl * self.nzsl

        # Keep args consistent for SICKLE downstream
        self.args.nxsl, self.args.nysl, self.args.nzsl = self.nxsl, self.nysl, self.nzsl
        self.args.nx, self.args.ny, self.args.nz = self.nxsl, self.nysl, self.nzsl

        # flatten index -> OpenFOAM internal face index (reversible)
        self.face_ids_flat = self._grid.reshape(-1).astype(np.int32, copy=False)
        self.args.face_ids_flat = self.face_ids_flat

    @staticmethod
    def _internal_face_grid(m, direction: str) -> np.ndarray:
        map_x = m["map_x"]
        map_y = m["map_y"]
        map_z = m["map_z"]

        if direction == "x":
            g = map_x[1:-1, :, :]
        elif direction == "y":
            g = map_y[:, 1:-1, :]
        else:  # "z"
            g = map_z[:, :, 1:-1]

        if g.min() < 0:
            raise RuntimeError("Found -1 in internal face grid; face map seems inconsistent.")
        return g.astype(np.int32, copy=False)

    def load_xyz(self):
        # index coordinates (sufficient for hypercube selection; later you can swap to physical coords)
        x = np.arange(self.nxsl, dtype=np.float32)
        y = np.arange(self.nysl, dtype=np.float32)
        z = np.arange(self.nzsl, dtype=np.float32)
        return x, y, z

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv, **kwargs):
        d = np.load(self.xy_npz, allow_pickle=True)
        Xall = d["X"]  # (T, Ninternal, Fx)
        Yall = d["Y"]  # (T, Ninternal, Fy)

        X = Xall[:, self.face_ids_flat, :]
        Y = Yall[:, self.face_ids_flat, :]

        cv_arr = X[:, :, self.cv_index]  # (T, Npoints)
        return X, Y, cv_arr

# ---- IMPORTANT: SICKLE expects module.DataLoader to exist ----
DataLoader = DataLoaderOFFaceNPZ




