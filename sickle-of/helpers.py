import importlib
import inspect
import numpy as np
import os
import time
from constants import *

from matplotlib import pyplot as plt

import builtins
from mpi4py import MPI

def setup_rank_print():
    """
    Override the built-in print function to only print from rank 0.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    original_print = builtins.print

    def rank_print(*args, force=False, **kwargs):
        if rank == 0 or force:
            original_print(*args, **kwargs)

    builtins.print = rank_print


def compute_euclidean_distance(x, y):
    return np.sqrt(x**2 + y**2)


def scale(func, x):
    """convert data to 2D scale and reshape back to 3D"""
    return func(x.reshape(-1, x.shape[-1])).reshape(x.shape)


def scale_probabilities(probs, a=0.01, b=0.99):
    """
    Scale a list of probabilities linearly from range [a, b].
    
    Args:
    - probs: List of probabilities
    - a, b: Range for scaling (default is [0.01, 0.99])

    Returns:
    - Scaled list of probabilities
    """
    A, B = min(probs), max(probs)
    scaled_probs = [(x - A) * (b - a) / (B - A) + a for x in probs]
    return np.array(scaled_probs)


def print_stats(label, X, Y):

    stats = lambda x : f"min: {np.amin(x):.04f}, mean: {np.mean(x):.04f}, max: {np.amax(x):.04f}"

    print(label)
    print(X.shape)
    print('X[0]:', stats(X[:, 0]))
    print('X[1]:', stats(X[:, 1]))
    print('Y:', stats(Y[:]))


def verbose_io(func):
    def wrapper(*args, **kwargs):
        print(f"{func.__name__} {args[0]}")
        return func(*args, **kwargs)
    return wrapper


@verbose_io
def load(*args, **kwargs):
    return np.load(*args, **kwargs)


@verbose_io
def savez(*args, **kwargs):
    np.save(*args, **kwargs)


# Function to compute grid coordinates for subdomain/box
def get_1Dgrid(Lh, nx, nxoffset, nxsl, nxskip):
    '''
      Lh: Length of grid dimension
      nx: # of points in original grid
      nxoffset: corner of the original gridfrom which the subdomain grid to be created
      nxsl: # of points of subdomain
      nxskip: # points to be skipped from original domain to create subdomain - subsampling
    '''
    dx = Lh/nx
    xin = 0 + (dx*nxoffset)
    xfi = xin + dx*nxsl*nxskip
    x = np.linspace(xin, xfi, nxsl)
    return x


def get_data_memmap(loadpath, nx, ny, nz, nxsl, nysl, nzsl, nxoffset, nyoffset, nzoffset, nxskip, nyskip, nzskip, nbytes):
    # Check data
    check_data(loadpath, nx, ny, nz, nbytes)
    # Memory-map the binary file
    t = time.time()
    data_memmap = np.memmap(loadpath, dtype=np.float32, mode='r', shape=(nz, ny, nx)) # NOTE: data is stored [z, y, x]
    elpsdt = time.time() - t
    # print(f'Time elapsed for memmap: {int(elpsdt/60)} min {elpsdt%60:.4f} sec')
    # Extract the sub-cube
    t = time.time()
    sub_cube = data_memmap[ nzoffset:nzoffset+(nzsl*nzskip):nzskip, # start from `nzoffset` location and get `nzsl` points, but skip every `nzskip` point
                          nyoffset:nyoffset+(nysl*nyskip):nyskip, 
                          nxoffset:nxoffset+(nxsl*nxskip):nxskip] 
    elpsdt = time.time() - t
    # print(f'Time elapsed for slice: {int(elpsdt/60)} min {elpsdt%60:.4f} sec')
    # Copy the sub-cube to a new array to avoid memory-mapping issues when processing
    t = time.time()
    datacube = sub_cube.copy().transpose(2, 1, 0) # transposing data to be [x, y, z]
    elpsdt = time.time() - t
    # print(f'Time elapsed for copying data: {int(elpsdt/60)} min {elpsdt%60:.4f} sec')
    data_memmap._mmap.close()
    del data_memmap, sub_cube
    # Print the shape of the sub-cube
    # print(f'Shape of the sub-cube: {datacube.shape}')
    return datacube


#def check_data(loadpath, nx, ny, nz, nbyte):
#  # print('Checking data file...')
#  # read in test binary and check number of samples
#  binary = open(loadpath, 'rb')
#  binary.seek(0,2) ## seeks to the end of the file (needed for getting number of bytes)
#  num_bytes = binary.tell() ## how many bytes are in this file is stored as num_bytes
#  
#  if int(num_bytes/nbyte)==nx*ny*nz:
#      num_samp = nx*ny*nz
#      # print(f'Number of samples counted == actual. Check complete.')
#  else:
#      print(f'Number of bytes in file =\t{num_bytes:,}')
#      print(f'Number of counted samples =\t{int(num_bytes/nbyte):,}')
#      print(f'Number of actual samples =\t{nx*ny*nz:,}')
#      raise Exception(f'Number of samples counted != actual')
#  binary.close()

def check_data(loadpath, nx, ny, nz, nbyte, channels=1):
    with open(loadpath, 'rb') as binary:
        binary.seek(0, 2)  # Seek to the end of the file.
        num_bytes = binary.tell()
    
    expected_samples = nx * ny * nz * channels
    counted_samples = int(num_bytes / nbyte)
    
    if counted_samples == expected_samples:
        # File size is as expected.
        return
    else:
        print(f'Number of bytes in file =\t{num_bytes:,}')
        print(f'Number of counted samples =\t{counted_samples:,}')
        print(f'Number of actual samples =\t{expected_samples:,}')
        raise Exception('Number of samples counted != actual')

def check_and_create_dirs(directory):
    """ Checks if a directory exists, and creates it if it doesn't.  """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_calling_filename():
    current_module = os.path.basename(__file__)
    for frame in inspect.stack()[1:]:
        caller_filename = os.path.basename(frame.filename)
        if caller_filename != current_module:
            # Return the base name without the extension
            return os.path.splitext(caller_filename)[0]
    # Fallback if all frames come from the current module
    return os.path.splitext(current_module)[0]


def estimate_memory(shape, dtype):
    """
    Estimate the memory required for a NumPy array.

    Parameters:
    - shape (tuple): The shape of the array.
    - dtype (numpy.dtype or str): The data type of the array.

    Returns:
    - memory_bytes (int): The estimated memory usage in bytes.
    """
    dtype = np.dtype(dtype)
    num_elements = np.prod(shape)  # Total number of elements
    bytes_per_element = dtype.itemsize  # Bytes per element
    return num_elements * bytes_per_element  # Total memory in bytes


def compute_memory(data):
    """
    Compute the total memory required for all arrays in a loaded NPZ file.

    Parameters:
    - data (np.lib.npyio.NpzFile): The loaded NPZ file from np.load(npz_file).

    Returns:
    - total_memory (dict): Total memory in bytes, MB, and GB.
    """
    total_memory_bytes = 0

    for key in data.files:
        array = data[key]
        memory_bytes = estimate_memory(array.shape, array.dtype)
        print(f"Array '{key}': {memory_bytes / (1024**2):.2f} MB")
        total_memory_bytes += memory_bytes

    return {
        "bytes": total_memory_bytes,
        "MB": total_memory_bytes / (1024**2),
        "GB": total_memory_bytes / (1024**3),
    }


def broadcast_large_array(data, comm, root=0):
    """
    Broadcast large arrays in chunks.

    This function first broadcasts the metadata (shape and dtype), then splits the array into
    chunks that are safely below the INT_MAX limit and broadcasts each chunk.
    """
    rank = comm.Get_rank()
    if rank == root:
        shape = data.shape
        dtype = data.dtype
    else:
        shape = None
        dtype = None
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    if rank != root:
        data = np.empty(shape, dtype=dtype)

    MAX_ELEMENTS = 2**30  # safely below INT_MAX
    total_elements = np.prod(shape)

    # Flatten array for chunking
    flat_data = data.ravel()

    for i in range(0, total_elements, MAX_ELEMENTS):
        end = min(i + MAX_ELEMENTS, total_elements)
        comm.Bcast([flat_data[i:end], np.dtype(dtype).char], root=root)

    return data

def print_sparsity(data, tolerance=1e-6):
    """
    Prints the sparsity of the data as the percentage of elements that are zero
    (or within a specified tolerance of zero).

    Parameters:
        data (np.ndarray): The input array.
        tolerance (float): Values with absolute value below this threshold
                           are considered zero. Default is 1e-6.
    """
    total_elements = data.size
    non_zero_elements = np.count_nonzero(np.abs(data) > tolerance)
    zero_elements = total_elements - non_zero_elements
    sparsity = zero_elements / total_elements
    print(f"Sparsity: {sparsity:.2%}")


if __name__ == "__main__":
    # Example usage:
    shape = (128, 16, 128)
    dtype = np.float64
    memory_estimate = estimate_memory(shape, dtype)
    print(f"Estimated memory usage: {memory_estimate['bytes']} bytes "
          f"({memory_estimate['MB']:.2f} MB, {memory_estimate['GB']:.4f} GB)")
