import numpy as np
from .base import Subsampler

class StratifiedSubsampler(Subsampler):
    def sample(self, num_samples, timestep):
        """
        Stratified sampling by sorting data on the first feature.
        Divides the sorted indices into num_samples strata and selects one index from each.
        """
        # Extract the samples at the given timestep.
        X_local = self.data[timestep]  # shape: (N, features)
        N = X_local.shape[0]

        # Sort the indices based on the first feature.
        sorted_indices = np.argsort(X_local[:, 0])
        
        # Divide the sorted indices into num_samples groups (strata)
        strata = np.array_split(sorted_indices, num_samples)
        chosen = []
        for group in strata:
            if group.size > 0:
                chosen.append(np.random.choice(group))
        
        # If there are fewer groups than num_samples (edge case), fill in extra indices randomly.
        if len(chosen) < num_samples:
            missing = num_samples - len(chosen)
            remaining = list(set(range(N)) - set(chosen))
            extra = np.random.choice(remaining, missing, replace=False)
            chosen.extend(extra)
        
        return np.array(chosen)
