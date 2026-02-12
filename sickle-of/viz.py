import os
import numpy as np


def save_vtu(Xout, Yout, x, y, z, indices, output_dir, fileprefix):
    """
    Saves all timesteps in VTU (Unstructured Grid) format for time-series visualization in ParaView.

    Xout: (num_timesteps, num_samples, 4)  # Features: u, v, w, rho
    Yout: (num_timesteps, 32768, 1)  # Full field prediction, 1 target feature
    x, y, z: (32,)  # Grid coordinates (1D arrays)
    indices: (num_timesteps, num_samples)  # Indices of subsampled points in a flattened (32,32,32) grid
    """
    import pyvista as pv

    num_timesteps = Xout.shape[0]  # Number of time steps
    num_samples = Xout.shape[1]  # Number of subsampled points per timestep

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert x, y, z into a full (32, 32, 32) 3D grid
    X_full, Y_full, Z_full = np.meshgrid(x, y, z, indexing="ij")  # Shape: (32, 32, 32)

    # Step 2: Flatten to (32768, 3) so that we can index it correctly
    grid_points = np.column_stack([X_full.ravel(), Y_full.ravel(), Z_full.ravel()])  # Shape: (32768, 3)

    # Store filenames for the PVD time series file
    vtu_filenames = []

    for timestep in range(num_timesteps - 1):
        # Step 3: Extract subsampled spatial coordinates using indices
        subsampled_points = grid_points[indices[timestep]]  # Shape: (num_samples, 3)

        # Define connectivity: each point is a separate vertex cell
        cells = np.hstack([np.ones((num_samples, 1), dtype=int), np.arange(num_samples).reshape(-1, 1)]).flatten()

        # Create UnstructuredGrid for subsampled data
        subsampled_grid = pv.UnstructuredGrid(cells, np.full(num_samples, pv.CellType.VERTEX), subsampled_points)

        # Store velocity (u, v, w) and density (rho) as point data for subsampled points
        feature_names = ["u", "v", "w", "rho"]
        for i, name in enumerate(feature_names):
            subsampled_grid.point_data[name] = Xout[timestep, indices[timestep], i]

        # Add Yout (subsampled target) to the same subsampled grid
        yout_subsampled = Yout[timestep, indices[timestep], 0]  # Shape: (num_samples,)
        subsampled_grid.point_data["pv"] = yout_subsampled

        # Save each timestep as a VTU file
        vtu_filename = os.path.join(output_dir, f"subsampled_{fileprefix}_t{timestep}.vtu")
        subsampled_grid.save(vtu_filename)
        vtu_filenames.append((timestep, vtu_filename))

    # Generate a PVD file for time animation
    pvd_filename = os.path.join(output_dir, f"subsampled_{fileprefix}.pvd")
    with open(pvd_filename, "w") as pvd_file:
        pvd_file.write('<VTKFile type="Collection" version="1.0">\n')
        pvd_file.write('  <Collection>\n')
        for timestep, vtu_file in vtu_filenames:
            pvd_file.write(f'    <DataSet timestep="{timestep}" file="{os.path.basename(vtu_file)}"/>\n')
        pvd_file.write('  </Collection>\n')
        pvd_file.write('</VTKFile>\n')

    print(f'Saved all timesteps in {pvd_filename}')
