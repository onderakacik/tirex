import numpy as np
import torch
import random
import os

# Global seed value used for worker initialization
_BASE_SEED = 0


def set_base_seed(seed):
    """Set the base seed for worker initialization"""
    global _BASE_SEED
    _BASE_SEED = seed


# Define a worker initialization function to set seeds for data loading workers
def worker_init_fn(worker_id):
    """
    Initialize the worker with a deterministic seed derived from base_seed and worker_id.
    Each worker gets a unique but deterministic seed: base_seed + worker_id
    """
    # Use the global base seed plus worker_id as the seed for this worker
    global _BASE_SEED
    seed = _BASE_SEED + worker_id

    # Set Python hash seed for this process
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set all relevant random states with this seed
    random.seed(seed)  # Set Python's random seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU RNG seed
    torch.cuda.manual_seed(seed)  # Set CUDA RNG seed for current device
    torch.cuda.manual_seed_all(seed)  # Set CUDA RNG seed for all devices
