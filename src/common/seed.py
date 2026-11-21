"""One seed for everything (project default: 42).

Where this file fits: every training script calls set_seed() first, so that random choices
(weight start values, data shuffling, dropout) are the same on each run and results can be repeated.
Note: this does not force fully deterministic GPU kernels, so tiny differences can still happen.
"""
# random = Python's own generator, numpy = numpy's generator.
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, numpy and (if installed) PyTorch on CPU and all GPUs. Returns nothing."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # imported here so this file also works when torch is not installed

        torch.manual_seed(seed)  # CPU generator (also used for weights and shuffling)
        torch.cuda.manual_seed_all(seed)  # every GPU generator; does nothing without a GPU
    except ImportError:  # allow the shared code to be used without torch installed
        pass
