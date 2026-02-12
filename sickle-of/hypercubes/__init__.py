import numpy as np
import os


def get_hypercube_extractor(method, **selector_kwargs):
    """
    Returns an extractor function that partitions a 3D array into non-overlapping hypercubes
    and selects a subset based on the specified method.
    
    The returned extractor has the signature:
      extractor(loadpaths, nbytes, file_shape, cube_shape, num_cubes)
    
    Supported methods:
      - 'uniform': Selects the first num_cubes cubes.
      - 'random': Randomly selects num_cubes cubes.
      - 'maxent': Uses maxent subsampling. If the keyword "use_parallel" is True,
                  the parallel implementation is used; otherwise, the sequential one.
    
    Example usage:
    
        extractor = get_hypercube_extractor('maxent', use_parallel=True)
        hypercubeIDs = extractor(loadpaths, nbytes, (nx, ny, nz), (hx, hy, hz), num_cubes)
    
    Returns:
      extractor : function
          A closure that returns a list of selected hypercube IDs.
    """
    import numpy as np

    # --- Determine the low-level selector function based on the method ---
    if method == 'uniform':
        # Simply take the first num_cubes indices.
        def selector_func(indices, num_cubes, **kwargs):
            return np.arange(num_cubes)
    elif method == 'random':
        # Randomly select num_cubes indices without replacement.
        def selector_func(indices, num_cubes, **kwargs):
            return np.random.choice(len(indices), num_cubes, replace=False)
    elif method == 'maxent':
        # Choose the maxent implementation based on the use_parallel flag.
        use_parallel = selector_kwargs.get("use_parallel", False)
        if use_parallel:
            from .maxent_parallel import maxent_hypercubes as maxent_func
        else:
            from .maxent_sequential import maxent_hypercubes as maxent_func

        def selector_func(indices, num_cubes, nx, ny, nz, hx, hy, hz, **kwargs):
            sampled_subcubes = maxent_func(
                kwargs["loadpaths"],
                nx, ny, nz,
                hx, hy, hz,
                nskips=kwargs["nskips"],
                n_clusters=kwargs["num_clusters"],
                n_cubes=num_cubes,
                batch_size=1024,
                n_init=20,
                max_iter=10,
                n_iters=10
            )
            # Compute grid dimensions for converting 3D subcube coordinates to a 1D index.
            num_x = (nx - 2) // hx
            num_y = ny // hy
            return np.array([
                ix + num_x * (iy + num_y * iz)
                for (ix, iy, iz) in sampled_subcubes
            ])
    else:
        raise ValueError(f"Unsupported hypercube selection method: {method}")

    # --- The extractor closure ---
    def extractor(loadpaths, nbytes, file_shape, cube_shape, num_cubes, **kwargs):
        nx, ny, nz = file_shape
        hx, hy, hz = cube_shape

        # Compute how many cubes fit in each dimension.
        num_x = (nx - 2) // hx
        num_y = ny // hy
        num_z = nz // hz

        # Build a list of all hypercube indices. 
        # Need to flip order of x and z to be consistent with row-major order of data (z is slowest).
        indices = [(ix, iy, iz)
                   for iz in range(num_z)
                   for iy in range(num_y)
                   for ix in range(num_x)]

        # Determine selected indices.
        if num_cubes is not None and num_cubes <= len(indices):
            # Call the pre-determined selector_func.
            # For 'maxent', additional parameters are passed.
            sel_indices = selector_func(
                indices, num_cubes,
                nx=nx, ny=ny, nz=nz,
                hx=hx, hy=hy, hz=hz,
                nskips=kwargs.get("nskips"),
                loadpaths=loadpaths,
                num_clusters=kwargs.get("num_clusters")
            )
        else:
            sel_indices = np.arange(len(indices))

        hypercubeIDs = [indices[i] for i in sel_indices]
        print("selected hypercubes:", hypercubeIDs)
        return hypercubeIDs

    return extractor
