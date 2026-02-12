import numpy as np
from .random import RandomSubsampler
from .maxent import MaxentSubsampler
from .phase_space import PhaseSpaceSubsampler
from .stratified import StratifiedSubsampler
from .lhs import LatinHypercubeSubsampler


class FullSubsampler:
    def __init__(self, data, args, **kwargs):
        # Create an array of all indices; assume data is of shape (features, ...)
        self.indices = np.arange(data.shape[1])
    def sample(self, num_samples, timestep):
        # For the "full" method, simply return all indices
        return self.indices


def get_subsampler(data, args, method=None, **kwargs):
    """Factory function to select subsampler based on args.method"""
    method = method or args.method
    if method == "maxent":
        return MaxentSubsampler(data, args, **kwargs)
    elif method == "random":
        return RandomSubsampler(data, args)
    elif method == "uips":
        return PhaseSpaceSubsampler(data, args)
    elif method == "stratified":
        return StratifiedSubsampler(data, args)
    elif method == "lhs":
        return LatinHypercubeSubsampler(data, args)
    elif method == "full":
        return FullSubsampler(data, args)
    else:
        raise ValueError(f"Unsupported sampling method: {args.method}")
