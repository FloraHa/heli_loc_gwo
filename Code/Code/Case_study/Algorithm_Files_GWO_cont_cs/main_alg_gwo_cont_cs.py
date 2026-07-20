import numpy as np
import time

from Code.Case_study.Algorithm_Files_General_cs.alg_create_individuals_cs import create_individuals, adjust_initial_wolf
from Code.Case_study.Algorithm_Files_GWO_cs.alg_gwo_support_functions_cs import get_generators_continuous
from Code.Case_study.Algorithm_Files_General_cs.alg_eval_fitness_cs import evaluate_fitness_continuous, evaluate_single_wolf, evaluate_wolves

def extract_wolf_info(top_wolves, idx, fitness_cache):
    pos = np.array(top_wolves[idx][0], dtype=int)
    score = top_wolves[idx][1]
    key = pos.tobytes()
    imp = fitness_cache[key]["dispatch_importance"]
    return pos, score, key, imp

def extract_top(top_wolves, fitness_cache):
    # Extract alpha, beta, delta positions and importance vectors
    positions, scores, imps = [], [], []
    for wolf, score in top_wolves[:3]:
        arr = np.array(wolf, dtype=int)
        positions.append(arr)
        scores.append(score)
        imps.append(fitness_cache[arr.tobytes()]["dispatch_importance"])
    return positions, scores, imps

def get_position_from_importance_vector(importance_vector, params_case_study, params_gwo, rng=None):
    """
    Get the helicopter positioning from the importance vector.
    Randomizes tie-breaking for equal importances.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_locs = len(params_case_study.pot_locations)

    # Add a tiny random noise to break ties (without changing order meaningfully)
    jitter = rng.uniform(0, 1e-6, size=n_locs)
    noisy_importance = importance_vector + jitter

    # Sort descending with randomized tie-breaking
    sorted_indices = np.argsort(-noisy_importance)

    positioning = []
    for idx in sorted_indices:
        if len(positioning) >= params_case_study.num_helicopters:
            break

        loc = params_case_study.pot_locations[idx]
        imp = importance_vector[idx]

        # Add once normally
        positioning.append(loc)

        # If highly important and capacity left, add a second helicopter
        if imp >= params_gwo.threshold and len(positioning) < params_case_study.num_helicopters:
            positioning.append(loc)
            
    if len(positioning) < params_case_study.num_helicopters:
        # Loop again through the most important locations
        for idx in sorted_indices:
            if len(positioning) >= params_case_study.num_helicopters:
                break
            positioning.append(params_case_study.pot_locations[idx])

    # Ensure exact length
    positioning = positioning[:params_case_study.num_helicopters]

    return np.array(positioning, dtype=int)

def grey_wolf_optimizer_cont(params_case_study, params_fire, params_attack, params_sim_gwo, params_gwo,
    random_seed=1, path_extension=""):

    start_time_all = time.time()

    # Initialize RNGs
    rngs = get_generators_continuous(random_seed)

    # Initialize fitness cache
    fitness_cache = {}
    best_scores = []

    # Adjust initial population size if initial wolf is provided
    num_initial_wolves = (1 if params_gwo.initial_wolf_mclp is not None else 0) + (1 if params_gwo.initial_wolf_hist_heur is not None else 0)
    size_initial_population_adj = params_gwo.population_size - num_initial_wolves

    # Create base population as 2D NumPy array
    population = np.array(create_individuals(
        size_initial_population_adj, rngs['initial_selection_rng'], params_case_study.pot_locations, params_case_study.max_helicopter_per_district
    ), dtype=int)

    population = population[:, :params_case_study.num_helicopters]

    # Append initial wolf (also truncated) if provided
    if params_gwo.initial_wolf_mclp is not None:
        initial_wolf_array = np.array(
            adjust_initial_wolf(params_gwo.initial_wolf_mclp,
                                params_case_study.pot_locations,
                                params_case_study.max_helicopter_per_district),
            dtype=int
        )

        initial_wolf_array = initial_wolf_array[:params_case_study.num_helicopters]  # truncate
        # ensure shapes align for vstack
        population = np.vstack([population, initial_wolf_array])

    if params_gwo.initial_wolf_hist_heur is not None:
        initial_wolf_array = np.array(
            adjust_initial_wolf(params_gwo.initial_wolf_hist_heur,
                                params_case_study.pot_locations,
                                params_case_study.max_helicopter_per_district),
            dtype=int
        )
        initial_wolf_array = initial_wolf_array[:params_case_study.num_helicopters]  # truncate
        # ensure shapes align for vstack
        population = np.vstack([population, initial_wolf_array])
        
    print(f"Time for population generation: {time.time() - start_time_all:.2f} s")

    # --- Initial evaluation ---
    start_time_eval = time.time()
    top_wolves, fitness_cache, _ = evaluate_wolves(
        population, fitness_cache, params_case_study, params_fire, params_attack,
        params_sim_gwo, size_elite=3, path_extension=path_extension
    )
    elapsed_eval = time.time() - start_time_eval
    print(f"Initial evaluation time: {elapsed_eval:.2f} s")

    # Extract top 3 wolves
    [alpha_pos, beta_pos, delta_pos], [alpha_score, beta_score, delta_score], [alpha_imp, beta_imp, delta_imp] = extract_top(top_wolves, fitness_cache)
    alpha_pos_all_best = alpha_pos
    alpha_score_all_best = alpha_score

    # Store alpha score
    best_scores.append(alpha_score)

    print(f"Alpha position: {alpha_pos}, Alpha importance: {alpha_imp}")
        
    #print(f"Alpha score: {alpha_score}, Beta score: {beta_score}, Delta score: {delta_score}")

    # --- Main loop ---
    total_time_update = 0
    total_time_eval = 0
    for iteration in range(params_gwo.max_iter):
        start_time_update = time.time()
        a = 2 * (1 - (iteration / params_gwo.max_iter)**2)  # linearly decreasing

        for imp in [alpha_imp, beta_imp, delta_imp]:
            imp += rngs['update_rng'].normal(0, 0.2, size=imp.shape)
            imp[:] = np.clip(imp, 0, 1)

        for i, wolf in enumerate(population):

            # --- Retrieve current wolf info ---
            wolf_key = wolf.tobytes()
            wolf_data = fitness_cache[wolf_key]
            wolf_imp = wolf_data["dispatch_importance"]
            old_fitness = wolf_data["fitness"]

            A1, C1 = 2*a*rngs['update_rng'].random() - a, 2*rngs['update_rng'].random()
            A2, C2 = 2*a*rngs['update_rng'].random() - a, 2*rngs['update_rng'].random()
            A3, C3 = 2*a*rngs['update_rng'].random() - a, 2*rngs['update_rng'].random()

            D_alpha = np.abs(C1 * alpha_imp - wolf_imp)
            D_beta  = np.abs(C2 * beta_imp - wolf_imp)
            D_delta = np.abs(C3 * delta_imp - wolf_imp)

            X1 = alpha_imp - A1 * D_alpha
            X2 = beta_imp  - A2 * D_beta
            X3 = delta_imp - A3 * D_delta

            new_importance = (X1 + X2 + X3) / 3.0
            mutation_strength = 0.05 * a  # scales down as iterations go on
            new_importance += rngs['update_rng'].normal(0, mutation_strength, size=new_importance.shape)

            # --- Normalize and clip to [0,1] ---
            new_importance = np.clip(new_importance, 0, 1)
            new_importance /= new_importance.sum() + 1e-9


            # --- Map back to a new discrete positioning ---
            new_positions = get_position_from_importance_vector(new_importance, params_case_study, params_gwo, rngs['position_rng'])
            
            # --- Check the performance of the new position
            top_dummy, fitness_cache, (wolf_arr, new_fitness, new_importance_stored) = evaluate_wolves(
                new_positions, fitness_cache, params_case_study, params_fire, params_attack,
                params_sim_gwo, path_extension=path_extension
            )

            # --- If the new position is better, add it to the population
            population[i] = new_positions
            
        # --- End of inner loop: identify new guiding wolves ---
        top_wolves, fitness_cache, _ = evaluate_wolves(
            population, fitness_cache, params_case_study, params_fire, params_attack,
            params_sim_gwo, size_elite=3, path_extension=path_extension
        )
        [alpha_pos, beta_pos, delta_pos], [alpha_score, beta_score, delta_score], [alpha_imp, beta_imp, delta_imp] = extract_top(top_wolves, fitness_cache)
        best_scores.append(alpha_score)

        if alpha_score < alpha_score_all_best:
            alpha_pos_all_best = alpha_pos
            alpha_score_all_best = alpha_score

        print(f"Alpha position: {alpha_pos}, Alpha importance: {alpha_imp}")

        # Optional: print progress
        print(f"Iter {iteration+1}/{params_gwo.max_iter} | "
              f"Alpha: {alpha_score:.4f}, Beta: {beta_score:.4f}, Delta: {delta_score:.4f}")
              
        print(f"Time for update: {time.time() - start_time_update:.2f} s")
    
    elapsed_total = time.time() - start_time_all

    print(best_scores)
    
    return alpha_pos_all_best, alpha_score_all_best, best_scores, elapsed_total, fitness_cache



