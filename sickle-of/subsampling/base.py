class Subsampler:
    def __init__(self, data, args):
        """
        Initialize with the full dataset and arguments.
        """
        self.data = data
        self.args = args

    def sample(self, num_samples, timestep):
        """
        Return subsampled indices for the given timestep.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement the sample method.")
