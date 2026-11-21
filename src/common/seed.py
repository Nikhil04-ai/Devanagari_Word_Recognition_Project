"""One seed for everything (project default: 42).

Where this file fits: every training script calls set_seed() first, so that random choices
(weight start values, data shuffling, dropout) are the same on each run and results can be repeated.
Note: this does not force fully deterministic GPU kernels, so tiny differences can still happen.
"""
# random = Python's own generator, numpy = numpy's generator.
import random

import numpy as np

# ===== CONCEPTS USED IN THIS FILE =====
# Seed: the starting number of a random number generator. The same seed gives the same "random" numbers.
# Reproducibility: getting the same results again on a second run. A fixed seed is the first step.
# Random generators: Python's random, numpy's random and PyTorch's (CPU and each GPU) are separate, so each is seeded.
# What is random in training: initial weights, shuffling of data, dropout masks.
# Optional import: torch is imported inside the function, so this shared file also works without torch.
# Deterministic kernels: some GPU operations still vary slightly; seeding alone does not remove that.

# ===== VIVA Q&A FOR THIS FILE =====
# Q: What part of the project is this file?
# A: A small shared helper that makes random choices repeatable across all models.
# Q: How is it run?
# A: Not directly. Training scripts call set_seed(cfg["seed"]) at the start (default 42, see train.py).
# Q: What does it read and write?
# A: It reads only the seed number. It writes nothing; it only sets the internal state of the generators.


def set_seed(seed: int = 42) -> None:
    """Seed Python, numpy and (if installed) PyTorch on CPU and all GPUs. Returns nothing."""
    # Q: What does it do and why does it exist?
    # A: It seeds Python, numpy and PyTorch so weights, shuffling and dropout repeat on each run. Then a
    #    result can be repeated and different experiments can be compared fairly.
    # Q: Why is torch imported inside a try block?
    # A: The shared code can then be used (for example in tests) when torch is not installed. On ImportError
    #    the function just skips the torch seeding.
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # imported here so this file also works when torch is not installed

        torch.manual_seed(seed)  # CPU generator (also used for weights and shuffling)
        torch.cuda.manual_seed_all(seed)  # every GPU generator; does nothing without a GPU
    except ImportError:  # allow the shared code to be used without torch installed
        pass
