import numpy as np
import matplotlib.pyplot as plt
import os

from args import args
from subsampling import get_subsampler
from dataloaders import load_data
from scipy.stats import gaussian_kde
from hypercubes import get_hypercube_extractor


if __name__ == "__main__":

    # Define hypercube extraction function
    extractor = get_hypercube_extractor(args.hypercubes, use_parallel=True)

    # Load the data
    print("loading data")
    X, Y, cv, x, y, z = load_data(args, extractor=extractor)
    print("done loading")
    num_timesteps = X.shape[0] // args.window * args.window + 1
    print(f"X: {X.shape}; Y: {Y.shape}; cv: {cv.shape}; x: {x.shape}; y: {y.shape}; z: {z.shape}; num_timesteps: {num_timesteps}")

    # OF2D
    # python compare_methods.py contrib/configs/OF/default.yaml 
    ts = 97
    ymax = 1000

    # SST-P1 - make sure cluster_var set to just 'pv'
    # python compare_methods.py contrib/configs/SST/P1/Hrandom-Xmaxent-1.yaml --timesteps 17.2
    #ts = 0
    #ymax = 10

    # GESTS-2048
    # python compare_methods.py contrib/configs/GESTS/2048/Hmaxent-Xmaxent.yaml --timesteps 1
    #ts = 0
    #ymax = 100

    histograms = {}  # Store histogram data for plotting
    for method in ["full", "random", "uips", "maxent"]:

        # Define subsample function based on method
        if args.method == "maxent":
            subsampler = get_subsampler(X, args, method=method, coords=(x, y, z), cv=cv)
        else:
            subsampler = get_subsampler(X, args, method=method)

        # Apply the subsampling function to get indices
        indices = subsampler.sample(args.num_samples, ts)

        # Extract the subsampled cluster variable
        subsampled_cv = cv[ts, indices]
        histograms[method] = subsampled_cv

    print("Data Lengths:")
    for method, subsampled_cv in histograms.items():
        print(f"{method}: {len(subsampled_cv)}")

    # Use a predefined colormap
    colormap = plt.cm.tab10  # You can use 'tab10', 'viridis', 'plasma', etc.
    colors = [colormap(i) for i in range(len(histograms))]
    print(colors)

    # Plot the histograms
    plt.figure(figsize=(6, 4))
    for i, (method, subsampled_cv) in enumerate(histograms.items()):
        color = colors[i]

        xmin, xmax = np.min(subsampled_cv), np.max(subsampled_cv)
        print(f"xmin: {xmin}, xmax: {xmax}")
        bins = np.linspace(xmin, xmax, args.bins)

        plt.hist(subsampled_cv, bins=bins, alpha=0.6, label=method, density=True, color=color)

        kde = gaussian_kde(subsampled_cv.T)
        kde_values = kde(bins)
        #kde_values_normalized = kde_values / np.max(kde_values)
        #plt.plot(bins, kde_values, label=method, color=color)
        plt.plot(bins, kde_values, color=color)

        values, bin_edges = np.histogram(subsampled_cv, bins=bins, density=True)
        area = np.sum(values * np.diff(bin_edges))
        print(f"Total Area: {area}")

    fs = 10

    xlabel = {'p': 'Pressure', 'pv': 'Potential Vorticity', 'enstrophy': 'Enstrophy', 'wz': 'Vorticy ($\omega_z$)'}

    # Set plot title and labels
    #plt.title(args.dtype, fontsize=fs)
    plt.xlabel(xlabel[args.cluster_var[0]], fontsize=fs)
    plt.ylabel("Probability Density", fontsize=fs)
    plt.yscale('log')

    # Set tick label size
    plt.tick_params(axis='both', labelsize=fs)

    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)

    # Add legend
    plt.legend(loc='upper right', ncol=2, handlelength=1.0, fontsize=fs)

    # Set x-axis limits
    plt.xlim(xmin, xmax)
    plt.ylim(top=ymax)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, f"subsampling_methods_histograms_t{ts}_ns{args.num_samples}.png"))
    plt.close()
