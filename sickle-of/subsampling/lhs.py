import numpy as np
from .base import Subsampler

class LatinHypercubeSubsampler(Subsampler):
    def sample(self, num_samples, timestep):
        """
        Generates a Latin Hypercube design for the given timestep's data and
        selects samples whose features are closest to the design points.
        """
        X_local = self.data[timestep]  # shape: (N, d)
        N, d = X_local.shape

        # Compute per-dimension ranges.
        mins = np.min(X_local, axis=0)
        maxs = np.max(X_local, axis=0)
        
        # Generate an LHS design matrix of shape (num_samples, d).
        design = np.zeros((num_samples, d))
        for j in range(d):
            # Random permutation of n intervals.
            perm = np.random.permutation(num_samples)
            # For each design point, pick a random number in the corresponding interval.
            design[:, j] = (perm + np.random.rand(num_samples)) / num_samples
        
        # Scale the design to the range of each feature.
        design = mins + design * (maxs - mins)
        
        # For each design point, find the closest available sample.
        chosen_indices = []
        available = np.ones(N, dtype=bool)
        for i in range(num_samples):
            design_point = design[i, :]
            # Compute Euclidean distances from the design point to all samples.
            distances = np.linalg.norm(X_local - design_point, axis=1)
            # Ignore samples already chosen.
            distances[~available] = np.inf
            idx = np.argmin(distances)
            chosen_indices.append(idx)
            available[idx] = False  # Mark this sample as selected.
        
        return np.array(chosen_indices)
