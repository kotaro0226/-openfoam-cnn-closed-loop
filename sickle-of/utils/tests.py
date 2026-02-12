import numpy as np
import os
from args import args

def explore_npz(file_path):
    """
    Read an NPZ file and display information about its contents.
    
    Parameters:
    file_path (str): Path to the NPZ file
    
    Returns:
    dict: Dictionary containing the arrays from the NPZ file
    """
    # Load the NPZ file
    data = np.load(file_path)
    
    # Get list of all arrays in the file
    print("Arrays in the NPZ file:")
    print("-" * 50)
    
    # Create a dictionary to store the arrays
    arrays_dict = {}
    
    # Iterate through each array in the NPZ file
    for name in data.files:
        array = data[name]
        arrays_dict[name] = array
        
        # Display information about each array
        print(f"\nArray name: {name}")
        print(f"Shape: {array.shape}")
        print(f"Data type: {array.dtype}")
        print(f"Sample of data: {array.flatten()[:3]}...")  # Show first 3 elements
    
    data.close()  # Close the NPZ file
    return arrays_dict


if __name__ == "__main__":
    explore_npz(os.path.join(args.output_dir, "subsampled_random.npz"))
    explore_npz(os.path.join(args.output_dir, "subsampled_maxent.npz"))

