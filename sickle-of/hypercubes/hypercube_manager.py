import os
import numpy as np
from helpers import check_data
from . import get_hypercube_extractor

class HypercubeHandler:
    def __init__(self, method, dims_full, dims_sl, nbytes, num_hypercubes, 
                 **selector_kwargs):
        """
        Initializes the hypercube handler.
        
        Parameters:
          method (str): The hypercube selection method ('uniform', 'random', 'maxent').
          dims_full (tuple): The full dimensions of the data (nx, ny, nz).
          dims_sl (tuple): The subcube dimensions (nxsl, nysl, nzsl).
          nbytes (int): Number of bytes per data point.
          num_hypercubes (int): Number of hypercubes to extract.
          selector_kwargs: Additional keyword arguments for the extractor.
        """
        self.dims_full = dims_full
        self.dims_sl = dims_sl
        self.selector_kwargs = selector_kwargs
        self.nbytes = nbytes
        self.num_hypercubes = num_hypercubes
        
        # Create an extractor function using the provided method.
        self.extractor = get_hypercube_extractor(method, **self.selector_kwargs)
    
    def extract_ids(self, loadpaths):
        """
        Extracts the hypercube IDs given a list of file paths.
        
        Parameters:
          loadpaths (list): List of file paths for the variables at a given timestep.
          
        Returns:
          List of selected hypercube indices.
        """
        return self.extractor(loadpaths, self.nbytes, self.dims_full, self.dims_sl, 
                              self.num_hypercubes, **self.selector_kwargs)
    
    def load_hypercubes(self, file_path, hypercube_ids, has_vector=False):
        nx, ny, nz = self.dims_full
        hx, hy, hz = self.dims_sl
        
        # If the variable is vector, read with an extra channel dimension.
        if has_vector:
            # Expecting data stored as [channel, z, y, x]
            data_memmap = np.memmap(file_path, dtype=np.float32, mode='r',
                                    shape=(3, nz, ny, nx), order='F')
        else:
            data_memmap = np.memmap(file_path, dtype=np.float32, mode='r',
                                    shape=(nz, ny, nx), order='F')
        
        cubes = []
        for ix, iy, iz in hypercube_ids:
            x0 = ix * hx
            y0 = iy * hy
            z0 = iz * hz
            if has_vector:
                # Extract subcube for each channel.
                cube = data_memmap[:, z0:z0+hz, y0:y0+hy, x0:x0+hx]
                # Rearrange from shape (3, hz, hy, hx) to (hx, hy, hz, 3)
                cube = cube.copy().transpose(3, 2, 1, 0)
                # Flatten spatial dimensions so that each hypercube is (num_points, 3)
                cube = cube.reshape(-1, 3)
            else:
                cube = data_memmap[z0:z0+hz, y0:y0+hy, x0:x0+hx]
                cube = cube.copy().transpose(2, 1, 0).reshape(-1)
            cubes.append(cube)
        
        data_memmap._mmap.close()
        # Concatenate cubes along the first dimension.
        return np.concatenate(cubes, axis=0)
