import numpy as np
from .base import Subsampler

class RandomSubsampler(Subsampler):
    def sample(self, num_samples, timestep):
        # Optionally set seed if required by your args.
        if not self.args.noseed:
            np.random.seed()  # Or use a more sophisticated seeding strategy.
        return np.random.choice(self.data.shape[1], num_samples, replace=False)
