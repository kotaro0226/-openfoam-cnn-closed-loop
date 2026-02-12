# sickle-main/dataloaders/of-cell-npz.py
import numpy as np
from dataloaders import DataLoader

class DataLoaderOFCellNPZ(DataLoader):
    """
    Build cell-wise samples from face-wise OpenFOAM NPZ.

    Inputs:
      - args.xy_npz   : cfd_xy_*.npz with X(T,Nf,Fx), Y(T,Nf,Fy)
      - args.cell_npz : cell_maps_*.npz with cell_faces_idx(Nc,6), cell_face_sign(Nc,6), V(Nc)
    Output (to SICKLE):
      - x,y,z : structured cell coordinate axes (or dummy)
      - X     : (T, Nc, 6*Fx)   concatenated per-cell 6-face features
      - Y     : (T, Nc, 6*Fy)   concatenated per-cell 6-face targets
      - cv    : (T, Nc)         cluster variable for MaxEnt (optional)
    """
    def __init__(self, args, **kwargs):
        super().__init__(args.path, dims=getattr(args, "dims", 3), verbose=getattr(args, "verbose", False))
        self.args = args
        self.xy_npz = args.xy_npz
        self.cell_npz = args.cell_npz  # <-- new arg
        self.cv_index = getattr(args, "cv_index", 0)

    def load_xyz(self):
        cm = np.load(self.cell_npz, allow_pickle=True)
        dims = tuple(int(x) for x in cm["dims_cells"].tolist())  # (Nx,Ny,Nz)
        Nx, Ny, Nz = dims
        # "cell axis" coordinates (index axes). For MaxEnt you may later use real coordinates.
        x = np.arange(Nx, dtype=np.float32)
        y = np.arange(Ny, dtype=np.float32)
        z = np.arange(Nz, dtype=np.float32)
        self.nx, self.ny, self.nz = Nx, Ny, Nz
        self.num_pts = Nx * Ny * Nz
        return x, y, z

    def load_multiple_timesteps(self, write_interval, num_timesteps, target=None, cv=None, **kwargs):
        d = np.load(self.xy_npz, allow_pickle=True)
        Xf = d["X"]  # (T,Nf,Fx)
        Yf = d["Y"]  # (T,Nf,Fy)

        cm = np.load(self.cell_npz, allow_pickle=True)
        cell_faces = cm["cell_faces_idx"].astype(np.int64)  # (Nc,6)
        cell_sign  = cm["cell_face_sign"].astype(np.int8)   # (Nc,6)  (we return it via extra keys later if needed)
        V          = cm["V"].astype(np.float32)             # (Nc,)

        T, Nf, Fx = Xf.shape
        Fy = Yf.shape[-1]
        Nc = cell_faces.shape[0]

        # Gather: (T,Nc,6,Fx) then flatten -> (T,Nc,6*Fx)
        X6 = Xf[:, cell_faces, :]          # fancy indexing
        Y6 = Yf[:, cell_faces, :]          # (T,Nc,6,Fy)
        Xc = X6.reshape(T, Nc, 6*Fx)
        Yc = Y6.reshape(T, Nc, 6*Fy)

        # Cluster variable for MaxEnt (optional): choose one channel from Xf and average over 6 faces
        # cv_index refers to the face-wise feature index in Xf: e.g., 2 for divPhiHbyA
        cv_face = X6[..., self.cv_index]   # (T,Nc,6)
        cv_cell = cv_face.mean(axis=2)     # (T,Nc)

        # Return cv as (T,Nc) but SICKLE expects (T,Nc) or (T,Nc,1) depending on implementation
        return Xc.astype(np.float32), Yc.astype(np.float32), cv_cell.astype(np.float32)

DataLoader = DataLoaderOFCellNPZ




