from collections import defaultdict
import numpy as np

def create_individuals(size_population, rng, districts, num_servers):
    """
    Create initial population of wolves (each as a NumPy array).
    """
    # vectorized approach (much faster than Python loop)
    base_sequence = np.repeat(np.array(districts), num_servers)
    population = np.empty((size_population, len(base_sequence)), dtype=int)

    for i in range(size_population):
        seq = base_sequence.copy()
        rng.shuffle(seq)
        population[i] = seq

    return population

def create_population_with_initial_wolf(size_population, rng, districts, num_servers, mclp_wolf):
    base_sequence = np.repeat(districts, num_servers)
    random_population = np.array([rng.permutation(base_sequence) for _ in range(size_population)], dtype=int)
    mclp_wolf_full = adjust_initial_wolf(mclp_wolf, districts, num_servers)
    population = np.vstack([random_population, mclp_wolf_full])
    return population

def adjust_initial_wolf(initial_wolf, districts, num_servers):
    """
    Adjusts an initial wolf (partial assignment) to match the full-length required configuration.
    Ensures each district appears num_servers times total.
    """
    initial_wolf = np.array(initial_wolf, dtype=int)
    districts = np.array(districts, dtype=int)
    
    # Count how many times each district already appears
    unique, counts = np.unique(initial_wolf, return_counts=True)
    used_counts = dict(zip(unique, counts))

    # Compute how many times each district is still available
    remaining_counts = {d: num_servers - used_counts.get(d, 0) for d in districts}
    remaining_sequence = np.repeat(
        [d for d, c in remaining_counts.items() for _ in range(max(0, c))],
        1
    )

    # Combine initial and remaining
    completed_wolf = np.concatenate([initial_wolf, remaining_sequence])

    expected_length = len(districts) * num_servers
    if len(completed_wolf) != expected_length:
        raise ValueError(
            f"Completed wolf length mismatch: {len(completed_wolf)} vs {expected_length}. "
            f"Counts per district: {remaining_counts}"
        )

    return completed_wolf


