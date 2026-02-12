"""
Parallel maxEnt-based hypercube selection for SST data.
"""
from mpi4py import MPI
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.stats import entropy
from collections import defaultdict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    seed_value = np.random.randint(0, 1000000)
else:
    seed_value = None

# Broadcast the seed to all ranks so they use the same random state
seed_value = comm.bcast(seed_value, root=0)
np.random.seed(seed_value)  # Set the seed for NumPy


def load_data_mpi(comm, loadpaths, nx, ny, nz, rank, size, nskips=(4,4,4)):
    """
    Load a full 4D dataset (stacked along a new axis) in parallel via MPI.
    The dataset is loaded on rank 0 from a list of loadpaths, then split along the specified axis.
    
    Parameters:
        comm: MPI communicator.
        loadpaths: list of str
            List of file paths.
        nx, ny, nz: int
            Dimensions of the full data cube (before transpose).
        rank: int
            Rank of the current MPI process.
        size: int
            Total number of MPI processes.
        split_axis: int, optional
            Axis along which to split the data (default 0).
        nskip: int, optional
            Stride when reading the memmap (default 1).

    Returns:
        local_data: np.ndarray
            Flattened local data with shape (n_voxels, n_features) where n_features equals the number
            of load paths.
        local_coords: np.ndarray
            Flattened coordinate array with shape (n_voxels, 3).
    """
    nxskip, nyskip, nzskip = nskips
    n_features = len(loadpaths)
    chunk_rank = (nz + size - 1) // size
    z_start = rank * chunk_rank
    z_end = min((rank + 1) * chunk_rank, nz)
    
    # Downsampled sizes for coordinate generation
    z_indices = np.arange(z_start, z_end, nzskip)
    y_indices = np.arange(0, ny, nyskip)
    x_indices = np.arange(0, nx - 2, nxskip)

    if z_start >= nz or len(z_indices) == 0: # if rank is more than slice
        return np.empty((0, n_features)), np.empty((0, 3))
    
    # Allocate array for local data
    total_samples = len(z_indices) * len(y_indices) * len(x_indices)
    local_data = np.empty((total_samples, n_features), dtype=np.float32)
    
    # Loop through each feature file
    for i, path in enumerate(loadpaths):
        data_memmap = np.memmap(path, dtype=np.float32, mode='r', shape=(nz, ny, nx))
        data_chunk = data_memmap[z_start:z_end:nzskip, ::nyskip, :-2:nxskip]  # adjust for SST padding
        local_data[:, i] = data_chunk.reshape(-1)

    # Generate local coordinates for histogram
    zz, yy, xx = np.meshgrid(z_indices, y_indices, x_indices, indexing='ij')
    local_coords = np.stack((zz, yy, xx), axis=-1)  # shape (nz_chunk, ny_chunk, nx_chunk, 3)
    local_coords = local_coords.reshape(-1, 3)

    return local_data, local_coords


def load_data_mpi2(comm, loadpath, nx, ny, nz, rank, size, split_axis=0):
    """
    Load a portion of the dataset for each MPI rank using memmap.
    Distribute data across ranks using MPI Scatter.
    Parameters:
    - loadpath: str - path to the binary file
    - nx, ny, nz: int - full data cube dimensions
    - rank: int - MPI rank of the process
    - size: int - total number of MPI processes

    Returns:
    - subcube: array pointer, shape (n_points, n_features=1) - portion of the dataset assigned to the process
    """

    if rank == 0:
        n_features = 1
        nskip = 1
        # Load full dataset on rank 0
        data_memmap = np.memmap(loadpath, dtype=np.float32, mode='r', shape=(nz, ny, nx))[::nskip,::nskip,:-2:nskip]
        # data_memmap = data_memmap.reshape(-1, n_features)  # Flatten for k-means

        # Create a coordinate grid that matches the shape of data_memmap.
        # Using np.indices yields an array of shape (3, nz_out, ny_out, nx_out).
        # Transpose to get shape (nz_out, ny_out, nx_out, 3) so that splitting along axis 0 (z-axis) works directly.
        coords = np.indices(data_memmap.shape).transpose(1, 2, 3, 0)
        
        # Split data along the Z-axis
        split_data = np.array_split(data_memmap, size, axis=split_axis)
        split_coords = np.array_split(coords, size, axis=split_axis)
    else:
        split_data = None  # Other ranks have no data yet
        split_coords = None
        n_features = None

    n_features = comm.bcast(n_features, root=0)

    # Scatter chunks to all ranks
    local_data = comm.scatter(split_data, root=0).reshape(-1, n_features)  # Flatten for k-means

    # Scatter the coordinate chunks; each process reshapes its local coordinates to (-1, 3).
    local_coords = comm.scatter(split_coords, root=0).reshape(-1, 3)

    return local_data, local_coords

def mpi_kmeans(local_data, n_clusters, batch_size=10000, n_init=10, max_iter=100, n_iters=10):
    """
    Parallel k-means clustering using MPI and iterative global synchronization.

    Parameters:
        load_data: array, shape (n_pints, n_features) - local data for each MPI rank
        n_clusters: int -  number of clusters
        batch_size: int - mini-batch size for k-means
        n_init: int - number of initialization runs
        max_iter: int - maximum iterations for k-means
        n_iters: int - # of syncs between ranks to converge on the centers

    Returns:
    - Cluster centers from root process.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start timing for MPI K-Means
    mpi_start_time = MPI.Wtime()

    # Step 1: Initialize cluster centers on rank 0
    if rank == 0:
        kmeans_init = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init=n_init, max_iter=max_iter, init="k-means++", random_state=seed_value)
        kmeans_init.fit(local_data)  # Initial training to get centers
        cluster_centers = kmeans_init.cluster_centers_
    else:
        cluster_centers = np.empty((n_clusters, local_data.shape[1]), dtype=np.float32)

    comm.Bcast(cluster_centers, root=0)  # Broadcast initial centers to all ranks

    # Step 2: Iteratively update centers with global synchronization
    for i in range(n_iters):
        # Use the precomputed centers to initialize MiniBatchKMeans properly
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init=1, max_iter=max_iter, init=cluster_centers, random_state=seed_value)

        # Each rank updates centers using its local data
        kmeans.partial_fit(local_data)

        # Collect updated centers across all ranks
        local_centers = kmeans.cluster_centers_

        # Use Allreduce to sum the cluster centers across all ranks
        global_centers = np.zeros_like(local_centers)
        comm.Allreduce(local_centers, global_centers, op=MPI.SUM)

        # Normalize by number of processes
        cluster_centers = global_centers / size

        if rank == 0: 
            print(f"Iteration {i+1}/{n_iters} - Synchronizing cluster centers...")

    comm.Barrier()  # Ensure all ranks finish

    # Stop timing
    mpi_end_time = MPI.Wtime()
    mpi_time = mpi_end_time - mpi_start_time

    # if rank == 0:
    #     print(f"Parallel (MPI) K-Means Time: {mpi_time:.4f} seconds")

    return cluster_centers

def compute_local_histograms(local_data, local_labels, n_clusters, num_bins, bin_range):
    local_histograms = []
    for k in range(n_clusters):
        # Select all feature values for cluster k.
        cluster_data = local_data[local_labels == k].flatten()
        if cluster_data.size > 0:
            counts, _ = np.histogram(cluster_data, bins=num_bins, range=bin_range, density=False)
        else:
            counts = np.zeros(num_bins, dtype=np.int64)
        local_histograms.append(counts)
    return local_histograms

def reduce_histograms(comm, local_histograms, n_clusters):
    """
    Reduce local histograms from all processes into global histograms.
    
    Parameters:
       comm: MPI communicator.
       local_histograms: list of np.ndarray for each cluster.
       n_clusters: int, number of clusters.
       
    Returns:
       global_histograms: list of np.ndarray for each cluster (on the root process).
    """
    global_histograms = []
    for k in range(n_clusters):
        global_hist = np.zeros_like(local_histograms[k])
        comm.Reduce(local_histograms[k], global_hist, op=MPI.SUM, root=0)
        global_histograms.append(global_hist)
    return global_histograms

def compute_cluster_probability_distributions(global_histograms):
    """
    Normalize each histogram to obtain a probability distribution.
    
    Parameters:
       global_histograms: list of np.ndarray - Global histogram counts for each cluster.
       
    Returns:
       cluster_distributions: list of np.ndarray - Normalized distributions (each sums to 1).
    """
    cluster_distributions = []
    for hist in global_histograms:
        total = hist.sum()
        if total > 0:
            distribution = hist / total
        else:
            distribution = hist  # remains zeros if no counts
        cluster_distributions.append(distribution + 1e-10) # to avoid 0 probability
    return cluster_distributions

def compute_node_strengths(cluster_distributions):
    """
    Compute node strengths for each cluster using KL divergence.
    
    For cluster i, the node strength is computed as the sum over j != i of:
         KL(P_i || P_j)
    where P_i is the probability distribution of cluster i.
    
    Parameters:
       cluster_distributions: list of np.ndarray - The probability distributions for each cluster.
           
    Returns:
       node_strengths: np.ndarray of shape (K,)
    """
    K = len(cluster_distributions)
    node_strengths = np.zeros(K)
    for i in range(K):
        for j in range(K):
            if i != j:
                # scipy.stats.entropy computes KL(P || Q)
                kl_div = entropy(cluster_distributions[i], cluster_distributions[j])
                node_strengths[i] += kl_div
    return node_strengths

def extract_local_subcube_totals(coords, data_node_strength, subcube_size):
    """
    Partition the 3D domain into subcubes and sum node strengths in each subcube.
    Add node strength of each local data to corresponding subcube ID.
    
    Parameters:
        coords: np.ndarray, shape (N,3) - 3D spatial coordinates for each data point.
        data_node_strength: np.ndarray, shape (N,) - The node strength for each data point.
        subcube_size: tuple (nxsl, nysl, nzsl) - The dimensions of each subcube.
    
    Returns:
        subcube_totals: dict - Mapping from subcube ID (a tuple of indices) to the summed node strength.
    """
    nxsl, nysl, nzsl = subcube_size
    subcube_totals = defaultdict(float)
    for (z, y, x), strength in zip(coords, data_node_strength):
        cube_id = (int(x // nxsl), int(y // nysl), int(z // nzsl))
        subcube_totals[cube_id] += strength
    return subcube_totals

def maxent_hypercubes(loadpath, nx, ny, nz, nxsl, nysl, nzsl, nskips, n_clusters, n_cubes, batch_size=10000, n_init=10, max_iter=100, n_iters=10):
    """
    Parallel maxEnt hypercube selection using k-means clustering using MPI4PY + MiniBatchKMeans.

    Parameters:
        loadpath: str - path to the binary file
        nx, ny, nz: int - full data cube dimensions
        n_clusters: int - number of clusters
        batch_size: int - mini-batch size for k-means
        n_init: int - number of initialization runs
        max_iter: int - maximum iterations for k-means
        n_iters: int - # of syncs between ranks to converge on the centers

    Returns:
        sampled_subcubes: list (n_cubes, 3) - Selected hypercube IDs
    """

    # Initialize MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Process rank
    size = comm.Get_size()  # Total processes
    if rank == 0:
        print(f"Total processors: {size}")

    # Load dataset slice for each MPI rank
    if rank == 0:
        print(f"Distributing data for parallel k-means...")
    local_data, local_coords = load_data_mpi(comm, loadpath, nx, ny, nz, rank, size, nskips)

    # Start timing for MPI K-Means
    if rank == 0:
        print(f"Parallel k-means...")
    mpi_start_time = MPI.Wtime()

    # 0. Parallel k-means using MiniBatchKMeans
    cluster_centers = mpi_kmeans(local_data, n_clusters=n_clusters, batch_size=int(4096/size), n_init=20, max_iter=10, n_iters=10)
    # print("cluster_centers:", cluster_centers)
    # print("type(cluster_centers):", type(cluster_centers))
    cluster_centers = cluster_centers[np.lexsort(tuple(cluster_centers[:, i] for i in range(cluster_centers.shape[1]-1, -1, -1)))]

    # Stop timing
    mpi_end_time = MPI.Wtime()
    mpi_time = mpi_end_time - mpi_start_time

    if rank == 0:
        print(f"Parallel (MPI) K-Means Time: {mpi_time:.4f} seconds", flush=True)
        print(f"Centers: {cluster_centers}", flush=True)

    # 1. Compute local data labels
    local_labels = pairwise_distances_argmin(local_data, cluster_centers)

    # 2. Compute local histograms for each label
    local_min = np.min(local_data)
    local_max = np.max(local_data)
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    bin_range = (global_min, global_max)
    num_bins = 10
    local_histograms = compute_local_histograms(local_data, local_labels, n_clusters, num_bins, bin_range)
    
    # 3. Reduce local histograms to global histogram for all lables
    global_histograms = reduce_histograms(comm, local_histograms, n_clusters)

    # 4. On the root process, compute the probability distributions and node strength of each cluster.
    if rank == 0:
        # Compute probability from counts
        cluster_distributions = compute_cluster_probability_distributions(global_histograms)
        # Compute node strengths (using pairwise KL divergence)
        node_strengths = compute_node_strengths(cluster_distributions)
    else:
        cluster_distributions = None
        node_strengths = None
    
    # 5. Assign node strength to data points (based on cluster_label) 
    # Broadcast the computed node strengths to all processes.
    node_strengths = comm.bcast(node_strengths, root=0)
    if rank == 0:
        print(f"node_strengths: {node_strengths}")
    # For each data point, assign its cluster’s node strength.
    local_data_node_strength = np.array([node_strengths[label] for label in local_labels])

    # 6. Partition the 3D domain into subcubes and aggregate node strengths.
    subcube_size = (nxsl, nysl, nzsl)
    local_subcube_totals = extract_local_subcube_totals(local_coords, local_data_node_strength, subcube_size)
    
    # 7. Combine local subcube node strengths to global
    # Gather the local subcube totals to the root
    all_local_subcubes = comm.gather(local_subcube_totals, root=0)
    if rank == 0:
        # Merge dictionaries to obtain global subcube totals.
        global_subcube_totals = defaultdict(float)
        for local_dict in all_local_subcubes:
            for cube_id, total in local_dict.items():
                global_subcube_totals[cube_id] += total
       
        # 8. Rank subcubes based on total node strength and sample.
        subcube_ids = list(global_subcube_totals.keys())
        strengths = np.array([global_subcube_totals[cube_id] for cube_id in subcube_ids])
        
        # If all subcubes have zero strength, sample uniformly.
        if strengths.sum() > 0:
            probabilities = strengths / strengths.sum()
        else:
            probabilities = None

        sampled_indices = np.random.choice(len(subcube_ids), size=n_cubes, replace=False, p=probabilities)
        sampled_subcubes = [subcube_ids[i] for i in sampled_indices]
        # print("Sampled subcube IDs:", sampled_subcubes)
    else:
        sampled_subcubes = None

    sampled_subcubes = comm.bcast(sampled_subcubes, root=0)
    return sampled_subcubes
