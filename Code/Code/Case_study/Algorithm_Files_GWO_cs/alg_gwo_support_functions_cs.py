"""
This file contains the support funtions for the algorithm.
"""

import numpy as np


def get_generators(seed):
    """
    Create reproducible random number generators for the simulation.
    Returns a dictionary of RNGs.
    """
    master_rng = np.random.default_rng(seed)
    # Generate enough seeds
    rng_names = [
        "initial_selection_rng", "position_rng"
    ]
    master_seeds = master_rng.integers(0, 1_000_000, len(rng_names))
    
    # Create RNGs
    rngs = {name: np.random.default_rng(seed=s) for name, s in zip(rng_names, master_seeds)}
    return rngs

def get_generators_continuous(seed):
    """
    Create reproducible random number generators for the simulation.
    Returns a dictionary of RNGs.
    """
    master_rng = np.random.default_rng(seed)
    # Generate enough seeds
    rng_names = [
        "initial_selection_rng", "position_rng", "update_rng"
    ]
    master_seeds = master_rng.integers(0, 1_000_000, len(rng_names))
    
    # Create RNGs
    rngs = {name: np.random.default_rng(seed=s) for name, s in zip(rng_names, master_seeds)}
    return rngs

def hamming_distance(vector_1, vector_2):
    """Calculate the Hamming distance between two lists."""
    return sum(x != y for x, y in zip(vector_1, vector_2))