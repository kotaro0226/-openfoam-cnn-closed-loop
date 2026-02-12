import matplotlib.pyplot as plt
import numpy as np
import os
from args import args

elev = 20
azim = -45
show_ticks = False
figsize = (9, 6)

def plot_samples(indices, labels, x, y, z, ts):
    """Plot subsampled data based on input dimensions and settings."""
    plt.clf()
    plt.rcParams.update({'font.size': 10})

    if args.dims == 3:
        ax = plt.subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)

        if args.dtype in ['npz', 'sst-binary', 'gests']:
            x_indices, y_indices, z_indices = np.unravel_index(
                indices, (x.shape[0], y.shape[0], z.shape[0])
            )
            ax.scatter(
                x[x_indices], z[z_indices], y[y_indices], c=labels, s=5, alpha=0.5
            )
        else:
            ax.scatter(x[indices], y[indices], z[indices], c=labels, s=5, alpha=0.5)
    else:
        plt.figure(figsize=(9, 2))
        plt.scatter(x[indices], y[indices], c=labels, s=5, alpha=0.5)
        plt.xlim([-25, 65])
        plt.ylim([-10, 10])
        plt.axis('equal')

    # Hide ticks and background grid if show_ticks is False
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        #ax.grid(False)
        #ax._axis3don = False

    fn = f'subsample_plot_t{ts:04d}.png'
    print(f'Creating {fn}')
    plt.savefig(os.path.join(args.plot_dir, fn), dpi=100, bbox_inches='tight')
    plt.close()


def plot2d_contour(data, y, z, timestep):
    """Plots a 2D plane from 1D data, y, and z coordinates."""
    data_2d = data.reshape(len(y), len(z))
    plt.figure(figsize=(8, 6))
    plt.contourf(z, y, data_2d, cmap="viridis")
    plt.colorbar(label="Value")
    plt.xlabel("z")
    plt.ylabel("y")
    plt.title("yz plane")
    plt.savefig(os.path.join(args.plot_dir, f'plot2d_{timestep:04d}.png'), dpi=100)
    plt.close()


def plot_kmeans_2d(x, y, labels, timestep):
    plt.figure(figsize=(9, 2))
    plt.scatter(x, y, c=labels, marker='.', cmap='tab10')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'KMeans clustering of {args.cluster_var}')
    plt.colorbar()
    plt.savefig(os.path.join(args.plot_dir, f'kmeans_{timestep:04d}.png'), dpi=100)
    plt.close()


def plot_kmeans_3d(x, y, z, labels, timestep, plot_dir, cluster_var,
                   show_colorbar=False,
                   colorbar_range=(0, 20),
                   show_title=False):
    """
    Plots the 3D k-means clustering result, with options to hide ticks,
    show/hide colorbar, and show/hide title.

    Parameters:
      x, y, z        : Coordinate arrays (flattened or meshed).
      labels         : Cluster labels from k-means.
      timestep       : The current timestep (used in the filename).
      plot_dir       : Directory where the plot will be saved.
      cluster_var    : Name of the cluster variable (for plot title).
      show_ticks     : Whether to display axis tick labels and the background grid.
      show_colorbar  : Whether to display the colorbar.
      colorbar_range : (min, max) range of integer ticks on the colorbar.
      show_title     : Whether to display the plot title.
    """
    #plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    
    # Check if the provided x, y, z are 1D and of length != x.shape[0]
    if x.ndim == 1 and x.size != x.shape[0]:
        new_x = np.linspace(np.min(x), np.max(x), x.shape[0])
        new_y = np.linspace(np.min(y), np.max(y), y.shape[0])
        new_z = np.linspace(np.min(z), np.max(z), z.shape[0])
        Xgrid, Ygrid, Zgrid = np.meshgrid(new_x, new_y, new_z, indexing='ij')
        x_points = Xgrid.flatten()
        y_points = Ygrid.flatten()
        z_points = Zgrid.flatten()
    else:
        # If the axes are already the desired grid or multi-dimensional,
        # assume they are correctly defined; if they're 1D, build the grid.
        if x.ndim == 1:
            Xgrid, Ygrid, Zgrid = np.meshgrid(x, y, z, indexing='ij')
            x_points = Xgrid.flatten()
            y_points = Ygrid.flatten()
            z_points = Zgrid.flatten()
        else:
            x_points = x
            y_points = y
            z_points = z

    # Create the scatter plot
    sc = ax.scatter(x_points, y_points, z_points, c=labels, marker='.', cmap='tab10')

    # Optionally set integer ticks on the colorbar
    if show_colorbar:
        cbar = plt.colorbar(sc, ax=ax)
        # Make integer ticks from colorbar_range[0] to colorbar_range[1]
        ticks = np.arange(colorbar_range[0], colorbar_range[1] + 1, 1)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
    else:
        # If you want to hide the colorbar, just don't create one.
        pass

    # Axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Toggle the title
    if show_title:
        ax.set_title(f"KMeans clustering of {cluster_var}")

    # Hide ticks and background grid if show_ticks is False
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax._axis3don = False

    fn = f'kmeans_{timestep:04d}.png'
    plt.savefig(os.path.join(plot_dir, fn), dpi=100)
    plt.close()


def plot_adjacency_matrix(adj_matrix, n_dists, timestep):
    plt.clf()
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(12, 10), facecolor='1')
    ticks = np.arange(n_dists)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlabel('Cluster number')
    plt.ylabel('Cluster number')
    plt.imshow(adj_matrix, cmap='inferno')
    cbar = plt.colorbar()
    cbar.set_label(r'relative entropy, $D$')
    plt.axis('equal')
    fn = f'adj_matrix_{timestep:04d}.png'
    print(f'Creating {fn}', flush=True)
    plt.savefig(os.path.join(args.plot_dir, fn), dpi=100)
    plt.close()


def plot_prob_dists(bin_edges, global_prob_dist, random_prob_dist, maxent_prob_dist, timestep, cluster_var):
    """
    Plots probability distributions for the full dataset, random sampling, and MaxEnt sampling.
    
    Parameters:
      bin_edges          : Bin edges used for the histograms.
      global_prob_dist   : Probability distribution for the full dataset.
      random_prob_dist   : Probability distribution from random sampling.
      maxent_prob_dist   : Probability distribution from MaxEnt sampling.
      timestep           : The current timestep (used in the filename).
      plot_dir           : Directory where the plot will be saved.
      cluster_var        : Name of the cluster variable (used in the xlabel).
    """
    plt.figure(figsize=(6, 4))
    width = np.diff(bin_edges)
    plt.bar(bin_edges[:-1], global_prob_dist, width=width,
            color='black', align='edge', alpha=0.2, label='Full dataset',
            edgecolor='black', linewidth=2)
    plt.bar(bin_edges[:-1], random_prob_dist, width=width,
            color='blue', align='edge', alpha=0.2, label='Sampled via Random',
            edgecolor='blue', linewidth=2)
    plt.bar(bin_edges[:-1], maxent_prob_dist, width=width,
            color='green', align='edge', alpha=0.2, label='Sampled via MaxEnt',
            edgecolor='green', linewidth=2)
    plt.xlabel(f'Cluster variable ({cluster_var})')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.legend()
    fn = f'prob_dists_{timestep:04d}.png'
    print(f'Creating {fn}', flush=True)
    plt.savefig(os.path.join(args.plot_dir, fn), dpi=100)
    plt.close()


def plot_cluster_histogram(cluster_labels, num_clusters, timestep, plot_dir):
    """
    Plots a histogram of the cluster labels.
    
    Parameters:
      cluster_labels : Array of cluster labels.
      num_clusters   : Total number of clusters.
      timestep       : The current timestep (used in the filename).
      plot_dir       : Directory where the plot will be saved.
    """
    plt.figure()
    plt.hist(cluster_labels, bins=num_clusters, edgecolor='k')
    plt.xlabel('Cluster')
    plt.ylabel('Frequency')
    plt.title('Cluster Histogram')
    fn = f'histogram_{timestep:04d}.png'
    print(f'Creating {fn}', flush=True)
    plt.savefig(os.path.join(args.plot_dir, fn), dpi=100)
    plt.close()


def plot_histograms(X_train, X_test, Y_train, Y_test):
    bins = 50
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(X_train[:,0], bins=bins, alpha=0.8, edgecolor='lightblue', color='lightblue')
    ax[0].hist(X_test[:,0], bins=bins, alpha=0.5, edgecolor='red', color='red')
    ax[0].set_title('Histogram of X[0]')
    ax[1].hist(X_train[:,1], bins=bins, alpha=0.8, edgecolor='lightblue', color='lightblue')
    ax[1].hist(X_test[:,1], bins=bins, alpha=0.5, edgecolor='red', color='red')
    ax[1].set_title('Histogram of X[1]')
    ax[2].hist(Y_train, bins=bins, alpha=0.8, edgecolor='lightblue', color='lightblue')
    ax[2].hist(Y_test, bins=bins, alpha=0.5, edgecolor='red', color='red')
    ax[2].set_title('Histogram of Y')
    plt.tight_layout()
    fn = 'histogram_train_test.png'
    print(f'Creating {fn}', flush=True)
    plt.savefig(os.path.join(args.plot_dir, fn), dpi=100)
    plt.close()


def plot_learning_curve(train_loss_history, val_loss_history, title=None):
    plt.figure(figsize=(10, 5.5))
    plt.rcParams.update({'font.size': 18})
    if title:
        plt.title(title)
    plt.plot(train_loss_history, label='training')
    plt.plot(val_loss_history, label='validation', alpha=0.5)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel(r'Loss ($mse$)')
    plt.legend(frameon=False)
    plt.grid(True)
    fn = 'ML_loss-curves.png'
    print(f'Creating {fn}', flush=True)
    plt.savefig(os.path.join(args.plot_dir, fn), dpi=100, bbox_inches='tight')
    plt.close()


def plot_ML_outputs(Y_test_ML, Y_test):
    print(f"Plotting data: {Y_test_ML.shape}, {Y_test.shape}")
    nvar = Y_test.shape[1]
    plt.clf()
    plt.rcParams.update({'font.size': 15})
    ncols = 2  
    nrows = int(np.ceil(nvar / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 6 * nrows / 2), sharex=True, facecolor="1")
    axs = axs.ravel()
    for i in range(nvar):
        axs[i].scatter(Y_test[:, i], Y_test_ML[:, i], s=20)
        min_val = min(Y_test[:, i].min(), Y_test_ML[:, i].min())
        max_val = max(Y_test[:, i].max(), Y_test_ML[:, i].max())
        axs[i].plot([min_val, max_val], [min_val, max_val], '--', color='k')
    for i in range(nrows):
        axs[i * ncols].set_ylabel('Predicted')
    for i in range(min(ncols, nvar)):
        axs[-ncols + i].set_xlabel('True')
    for i in range(nvar, nrows * ncols):
        axs[i].set_visible(False)
    plt.tight_layout()
    fn = 'ML_loss-curves.png'
    print(f'Creating {fn}', flush=True)
    plt.savefig(os.path.join(args.plot_dir, fn), dpi=100)
    plt.close()

def plot_contour_box_3d(x, y, z, data, timestep):
    # Create a new figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    datacube = data.reshape(xx.shape)
    clevels = np.linspace(datacube.min(), datacube.max(), 101)
    kw = {
        'vmin': clevels.min(),
        'vmax': clevels.max(),
        'levels': clevels,
        'cmap': 'RdBu',
        'extend': 'both',
        'alpha': 0.5
    }
    A = ax.contourf(xx[:, -1, :], zz[:, -1, :], datacube[:, -1, :],
                     zdir='z', offset=yy.max(), **kw)
    B = ax.contourf(xx[:, :, 0], datacube[:, :, 0], yy[:, :, 0],
                     zdir='y', offset=0, **kw)
    C = ax.contourf(datacube[-1, :, :], zz[-1, :, :], yy[-1, :, :],
                     zdir='x', offset=xx.max(), **kw)
    xmin, xmax = xx.min(), xx.max()
    ymin, ymax = yy.min(), yy.max()
    zmin, zmax = zz.min(), zz.max()
    ax.set(xlim=[xmin, xmax], zlim=[ymin, ymax], ylim=[zmin, zmax])
    edges_kw = dict(color='0.5', linewidth=0.5, zorder=1e3)
    ax.plot([xmax, xmax], [zmin, zmax], ymin, **edges_kw)
    ax.plot([xmax, xmax], [zmin, zmax], ymax, **edges_kw)
    ax.plot([xmin, xmax], [zmin, zmin], ymin, **edges_kw)
    ax.plot([xmin, xmax], [zmin, zmin], ymax, **edges_kw)
    ax.plot([xmax, xmax], [zmin, zmin], [ymin, ymax], **edges_kw)
    ax.set(xlabel='X', ylabel='Z', zlabel='Y')
    ax.view_init(elev, azim)
    aspectratio = int(len(x) / len(y))
    ax.set_box_aspect([aspectratio, aspectratio, 1], zoom=1)
    ax.grid(False)

    # Hide ticks and background grid if show_ticks is False
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax._axis3don = False

    # Save plot
    fn = f'contour_{timestep:04d}.png'
    print(f'Creating {fn}', flush=True)
    plt.savefig(os.path.join(args.plot_dir, fn), dpi=100)
    plt.close(fig)


#def plot_corner(X):
#    import corner
#    figure = corner.corner(
#        X,
#        bins=30,
#        quantiles=[0.16, 0.5, 0.84],
#        show_titles=True,
#        title_fmt=".2f",
#        labels=[f"var_{i}" for i in range(X.shape[1])]
#    )
#    plt.savefig(os.path.join(args.plot_dir, f'uips_pdf.png'), dpi=100)
#    plt.close()
