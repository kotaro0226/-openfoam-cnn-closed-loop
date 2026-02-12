import os
import numpy as np
from sklearn.cluster import KMeans
import scipy.stats
import pandas as pd
from .base import Subsampler
from plotting import plot_adjacency_matrix, plot_kmeans_3d, plot_prob_dists, \
                     plot_cluster_histogram, plot_contour_box_3d, plot_samples

class MaxentSubsampler(Subsampler):
    def __init__(self, data, args, **kwargs):
        super().__init__(data, args)

        # Extract coords from kwargs
        coords = kwargs.get('coords')
        if coords is None:
            raise ValueError("MaxentSubsampler requires coords")
        self.coords = coords

        # Extract cv from kwargs
        cv = kwargs.get('cv')
        if cv is None:
            raise ValueError("MaxentSubsampler requires a cv array")
        self.cv = cv

    def sample(self, num_samples, timestep):
        # Use cv for clustering instead of self.data.
        num_cvs = len(self.args.cluster_var)
        data = self.cv[timestep, :].reshape(-1, num_cvs)
        num_clusters = self.args.num_clusters
        
        # Perform k-means clustering.
        labels, clusters = self.perform_kmeans(data, num_clusters)
        
        # Compute in-strength values (entropy) from the clusters.
        in_strengths = self.compute_entropy(clusters, timestep=timestep)
        
        # Create probabilities based on the in-strength values.
        probs = np.zeros(data.shape[0])
        for i in range(num_clusters):
            probs[labels == i] = in_strengths[i]
        
        # Normalize probabilities.
        probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
        probs /= np.sum(probs)
        
        # Randomly sample indices based on the computed probabilities.
        indices = np.random.choice(data.shape[0], num_samples, replace=False, p=probs)

        # Generate additional plots if enabled.
        if self.args.plot:
            self.generate_plots(data, labels, clusters, indices, timestep, num_samples)

        return np.array(indices)

    def perform_kmeans(self, data, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(data)
        labels = kmeans.labels_
        clusters = [data[labels == i].flatten() for i in range(num_clusters)]
        return labels, clusters

    def compute_entropy(self, clusters, timestep=0, num_bins=50):
        prob_dists = []
        bin_edges_list = []
        bin_range = (min([np.min(cluster) for cluster in clusters]),
                     max([np.max(cluster) for cluster in clusters]))
        for cluster in clusters:
            counts, bin_edges = np.histogram(cluster, bins=num_bins, range=bin_range, density=False)
            prob_dist = counts / np.sum(counts)
            prob_dists.append(prob_dist)
            bin_edges_list.append(bin_edges)
        n_clusters = len(clusters)
        adj_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                p = prob_dists[i] + 1e-10  # avoid division by zero
                q = prob_dists[j] + 1e-10
                adj_matrix[i, j] = scipy.stats.entropy(p, q)
        in_strengths = np.sum(adj_matrix, axis=0)
        df = pd.DataFrame(adj_matrix)
        print(df)

        if self.args.plot:
            plot_adjacency_matrix(adj_matrix, n_clusters, timestep)
        return in_strengths

    def generate_plots(self, data, labels, clusters, maxent_indices, timestep, num_samples):
        """
        Generate plots for KMeans clustering, cluster histogram, and
        probability distributions comparing the full dataset, random sampling,
        and MaxEnt sampling.

        Parameters:
          data           : The cv data at the given timestep.
          labels         : KMeans cluster labels.
          clusters       : List of cluster arrays.
          maxent_indices : Indices chosen via MaxEnt sampling.
          timestep       : Current timestep (for filenames).
          num_samples    : Number of samples requested.
        """
        x, y, z = self.coords

        # Plot the KMeans 3D scatter
        plot_kmeans_3d(x, y, z, labels, timestep, self.args.plot_dir, self.args.cluster_var)

        # Plot the 3D contour box if you have data to show.
        # We assume 'self.cv[timestep, :]' is shape (len(x)*len(y)*len(z),).
        contour_data = self.cv[timestep, :]
        plot_contour_box_3d(x, y, z, contour_data, timestep)

        # Plot samples
        plot_samples(maxent_indices, labels[maxent_indices], x, y, z, timestep)

        # Plot the cluster histogram.
        plot_cluster_histogram(labels, self.args.num_clusters, timestep, self.args.plot_dir)

        # Compute probability distributions for comparison:
        num_bins = 50
        bin_range = (min([np.min(cluster) for cluster in clusters]),
                     max([np.max(cluster) for cluster in clusters]))
        # Global probability distribution from the entire cv data at this timestep.
        global_counts, bin_edges = np.histogram(data, bins=num_bins, range=bin_range, density=False)
        global_prob_dist = global_counts / np.sum(global_counts)

        # Random sampling: select random indices and compute histogram.
        rand_indices = np.random.choice(data.shape[0], num_samples, replace=False)
        rand_counts, _ = np.histogram(data[rand_indices], bins=num_bins, range=bin_range, density=False)
        random_prob_dist = rand_counts / np.sum(rand_counts)

        # MaxEnt sampling probability distribution from the selected indices.
        maxent_counts, _ = np.histogram(data[maxent_indices], bins=num_bins, range=bin_range, density=False)
        maxent_prob_dist = maxent_counts / np.sum(maxent_counts)

        # Plot the probability distributions.
        plot_prob_dists(bin_edges, global_prob_dist, random_prob_dist, maxent_prob_dist,
                        timestep, self.args.cluster_var)
