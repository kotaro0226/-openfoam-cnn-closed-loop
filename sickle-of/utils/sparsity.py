import numpy as np
import sys

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

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename.npz>")
        sys.exit(1)
    filename = sys.argv[1]
    
    try:
        npz_data = np.load(filename)
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        sys.exit(1)
    
    # Assume the NPZ file contains at least one array; we use the first array.
    keys = list(npz_data.keys())
    print(keys)
    if not keys:
        print("No arrays found in the file.")
        sys.exit(1)
    
    print(npz_data['X'].shape)
    data = npz_data[keys[0]]
    print(data)
    print_sparsity(data)
