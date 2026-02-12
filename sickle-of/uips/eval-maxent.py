import numpy as np
import matplotlib.pyplot as plt
from algorithms import create_maxent_subsampler, subsample_random
from sklearn.preprocessing import MinMaxScaler
from args import args


def subsample_data(X, subsample_fn, num_samples):
    indices = subsample_fn(X, num_samples, 0)
    return X[indices,:]


class CombustionDataLoader():
    def __init__(self, path, verbose=False):
        self.path = path
        self.verbose = verbose

    def load_2D_combustion(self):
        """
        Load the 2D combustion dataset stored as a NumPy array.
        Assumes the dataset has shape (N, 2), where N is the number of samples.
        """
        # Load dataset
        data = np.load(self.path)
        if self.verbose:
            print(f"Loaded data shape: {data.shape}")

        # Extract features
        eC = data[:, 0]  # Progress variable
        gC002 = data[:, 1]  # Subfilter variance

        # Prepare outputs for MaxEnt framework
        X = np.stack((eC, gC002), axis=-1)  # Features (can include both variables)
        Y = eC  # Target (progress variable)
        cv = eC  # Cluster variable (same as target if limited data)

        if self.verbose:
            print(f"X shape: {X.shape}, Y shape: {Y.shape}, cv shape: {cv.shape}")

        return X, Y, cv

    def preprocess_data(self, data):
        """
        Normalize the data using Min-Max scaling.
        """
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        if self.verbose:
            print("Data successfully normalized.")
        
        return scaled_data

    def downsample_data(self, data, num_samples):
        """
        Downsample the data using random sampling or a specific entropy-driven method.
        """
        # Random sampling (default)
        indices = np.random.choice(data.shape[0], num_samples, replace=False)
        downsampled = data[indices]

        if self.verbose:
            print(f"Downsampled data shape: {downsampled.shape}")

        return downsampled

    def save_downsampled_data(self, data, prefix, num_samples):
        """
        Save downsampled data to NPZ format.
        """
        filename = f"{prefix}_{num_samples}.npz"
        np.savez(filename, data=data)

        if self.verbose:
            print(f"Saved downsampled data to {filename}")

# Scatter Plotting for Comparison
def scatter_plot(data, downsampled_data, label, filename):
    print(f"Generating scatter plot: {label}")
    plt.figure(figsize=(6, 6))
    plt.plot(data[:, 0], data[:, 1], "o", color="k", markersize=1, label="full DS")
    plt.plot(
        downsampled_data[:, 0],
        downsampled_data[:, 1],
        "o",
        color="r",
        markersize=2,
        label="downsampled",
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved scatter plot to {filename}")


if __name__ == "__main__":
    # Initialize loader
    loader = CombustionDataLoader('data/combustion2DToDownsampleSmall.npy', verbose=True)

    # Load data
    X, Y, cv = loader.load_2D_combustion()

    # Preprocess
    normalized_X = loader.preprocess_data(X)

    # Define the subsampling function for maximum entropy
    cv = np.expand_dims(X[:,0], axis=0) # test1
    #cv = np.expand_dims(X[:,1], axis=0) # test2
    #cv = np.expand_dims(X, axis=0) # test3 - 2D test see README.md for change needed in algorithms.py
    subsample_fn = create_maxent_subsampler(cv, args)

    # Perform subsampling
    maxent_downsampled = subsample_data(normalized_X, subsample_fn, num_samples=1000)

    # Save output
    #outfile = os.path.join(args.output_dir, 'subsampled_maxent.npz')
    #np.savez(outfile, X=Xout, Y=Yout, x=x, y=y, z=z)

    # Random sampling
    random_downsampled = loader.downsample_data(normalized_X, num_samples=1000)

    # Load Phase-Space downsampled data and normalize it
    phase_space_downsampled = np.load('downSampledData_1000_it1.npz')['data']
    phase_space_downsampled = loader.preprocess_data(phase_space_downsampled)

    # Generate scatter plots
    scatter_plot(normalized_X, maxent_downsampled, 'MaxEnt Downsampled Data', 'scatter_maxent_downsampled.png')
    scatter_plot(normalized_X, random_downsampled, 'Randomly Downsampled Data', 'scatter_random_downsampled.png')
    scatter_plot(normalized_X, phase_space_downsampled, 'Phase-Space Downsampled Data', 'scatter_phasespace_downsampled.png')
