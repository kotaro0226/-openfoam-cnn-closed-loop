import numpy as np
from .base import Subsampler

class PhaseSpaceSubsampler(Subsampler):
    def sample(self, num_samples, timestep):
        # For phase-space sampling, first extract data for the given timestep.
        X_local = self.data[timestep]
        hist, bin_edges = self.build_pdf(X_local)
        return self.subsample_uips(X_local, num_samples, hist, bin_edges)

    def build_pdf(self, X, nbins=10):
        # Flatten data over spatial dimensions if needed.
        X_flat = X.reshape(-1, X.shape[-1])
        hist, bin_edges = np.histogramdd(X_flat, bins=nbins)
        return hist, bin_edges

    def subsample_uips(self, X, n, hist, bin_edges):
        X_flat = X.reshape(-1, X.shape[-1])
        # Determine bin indices for each point.
        bin_indices_per_dim = []
        for dim in range(X_flat.shape[1]):
            bi = np.digitize(X_flat[:, dim], bin_edges[dim]) - 1
            bi = np.clip(bi, 0, len(bin_edges[dim]) - 2)
            bin_indices_per_dim.append(bi)
        bin_indices_per_dim = np.array(bin_indices_per_dim).T
        
        # Combine multi-dimensional indices into a single bin ID.
        bin_id = np.ravel_multi_index(tuple(bin_indices_per_dim.T), dims=hist.shape)
        nonzero_bin_ids = np.where(hist.ravel() > 0)[0]
        
        # Choose n bins uniformly from non-zero bins.
        chosen_bins = np.random.choice(nonzero_bin_ids, size=n, replace=True)
        chosen_indices = []
        for cb in chosen_bins:
            candidate_points = np.where(bin_id == cb)[0]
            if candidate_points.size:
                chosen_indices.append(np.random.choice(candidate_points))
        return np.array(chosen_indices, dtype=int)
