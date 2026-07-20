import numpy as np
import pandas as pd
import heapq
from collections import Counter

from Code.Case_study.Evaluation_Files_GWO_cs.eval_des import simulate

def evaluate_fitness(population, size_elite, fitness_cache, params_case_study,
                     params_fire, params_attack, params_sim_alg, path_extension):

    sim_rng = np.random.default_rng(seed=params_sim_alg.random_seed)
    sim_seeds = sim_rng.integers(0, 1000, params_sim_alg.replications)

    top_candidates = []

    print_num_fires = True

    for indiv in population:
        indiv_arr = np.array(indiv, dtype=int)
        positioning = indiv_arr[:params_case_study.num_helicopters]
        pos_key = positioning.tobytes()

        if pos_key not in fitness_cache:
            area_burned_ls = np.zeros(params_sim_alg.replications)
            response_time_ls = np.zeros(params_sim_alg.replications)
            sol_times_ls = np.zeros(params_sim_alg.replications)

            for i, seed in enumerate(sim_seeds):
                res = simulate(
                    positioning=positioning, params_case_study=params_case_study, params_fire=params_fire, params_attack=params_attack, params_sim=params_sim_alg, random_seed=seed, bool_print=False, training=False,
                )
                area_burned_ls[i] = res["area_burned_rep"]
                response_time_ls[i] = res["response_times_all_fires"]
                sol_times_ls[i] = res["sol_time_rep"]
                if print_num_fires:
                    print(f"Number of fires: {res['num_fires_rep']}")
                    print_num_fires = False

            area_burned = area_burned_ls.mean()
            response_time = response_time_ls.mean()
            sol_time = sol_times_ls.mean()
            value = response_time if path_extension == "response_time" else area_burned
            fitness_cache[pos_key] = (value, sol_time)
        else:
            if path_extension == "response_time":
                response_time, sol_time = fitness_cache[pos_key]
            else:
                area_burned, sol_time = fitness_cache[pos_key]

        # Keep a min-heap of top x wolfes / candidates
        if path_extension == "response_time":
            heapq.heappush(top_candidates, (-response_time, tuple(indiv_arr)))  # Convert array to tuple
        else:
            heapq.heappush(top_candidates, (-area_burned, tuple(indiv_arr)))  # Convert array to tuple

        if len(top_candidates) > size_elite:
            heapq.heappop(top_candidates)

    # Extract top 3, sorted by fitness ascending
    top_wolves = sorted([ (list(indiv), -score) for score, indiv in top_candidates ], key=lambda x: x[1])
    return top_wolves, fitness_cache


def evaluate_fitness_continuous(population, size_elite, fitness_cache, params_case_study,
                                params_fire, params_attack, params_sim_alg, path_extension):
    """
    Continuous version of evaluate_fitness:
    - Stores fitness + importance in cache.
    - Keeps top 3 wolves directly.
    """

    sim_rng = np.random.default_rng(seed=params_sim_alg.random_seed)
    sim_seeds = sim_rng.integers(0, 1000, params_sim_alg.replications)
    pot_locations = params_case_study.pot_locations
    loc_to_idx = {loc: idx for idx, loc in enumerate(pot_locations)}
    num_locations = len(pot_locations)

    top_candidates = []  # will contain tuples (-fitness, wolf_array)
    print_num_fires = True

    for indiv in population:
        indiv_arr = np.array(indiv, dtype=int)
        positioning = indiv_arr[:params_case_study.num_helicopters]
        pos_key = positioning.tobytes()

        # --- Retrieve from cache if available ---
        if pos_key not in fitness_cache:
            area_burned_ls = np.zeros(params_sim_alg.replications)
            response_time_ls = np.zeros(params_sim_alg.replications)
            sol_times_ls = np.zeros(params_sim_alg.replications)
            dispatch_counter = Counter()

            for i, seed in enumerate(sim_seeds):
                res = simulate(
                    positioning=positioning,
                    params_case_study=params_case_study,
                    params_fire=params_fire,
                    params_attack=params_attack,
                    params_sim=params_sim_alg,
                    random_seed=seed,
                    bool_print=False,
                    training=False,
                )
                area_burned_ls[i] = res["area_burned_rep"]
                response_time_ls[i] = res["response_times_all_fires"]
                sol_times_ls[i] = res["sol_time_rep"]

                if "dispatch_dict" in res:
                    for (loc, _district), cnt in res["dispatch_dict"].items():
                        if loc != "Lost":
                            dispatch_counter[loc] += cnt

                if print_num_fires:
                    print(f"Number of fires: {res['num_fires_rep']}")
                    print_num_fires = False

            # --- Aggregate results ---
            area_burned = area_burned_ls.mean()
            response_time = response_time_ls.mean()
            sol_time = sol_times_ls.mean()

            # --- Compute importance vector ---
            loc_counts = np.zeros(num_locations)
            for loc, cnt in dispatch_counter.items():
                if loc in loc_to_idx:
                    loc_counts[loc_to_idx[loc]] += cnt
            total = loc_counts.sum()
            importance_vector = loc_counts / total if total > 0 else np.zeros(num_locations)


            if path_extension == "response_time":
                fitness_cache[pos_key] = {
                    "fitness": float(response_time),
                    "sol_time": float(sol_time),
                    "dispatch_importance": importance_vector
                }
            else:
                fitness_cache[pos_key] = {
                    "fitness": float(area_burned),
                    "sol_time": float(sol_time),
                    "dispatch_importance": importance_vector
                }

        else:
            if path_extension == "response_time":
                response_time = fitness_cache[pos_key]["fitness"]
            else:
                area_burned = fitness_cache[pos_key]["fitness"]

            sol_time = fitness_cache[pos_key]["sol_time"]
            importance_vector = fitness_cache[pos_key]["dispatch_importance"]

        # Maintain top candidates (minimize area_burned)
        if path_extension == "response_time":
            heapq.heappush(top_candidates, (-response_time, tuple(indiv_arr)))
        else:
            heapq.heappush(top_candidates, (-area_burned, tuple(indiv_arr)))
        
        if len(top_candidates) > size_elite:
            heapq.heappop(top_candidates)

    # Extract top 3 (sorted ascending by fitness)
    top_wolves = sorted(
        [(list(indiv), -score) for score, indiv in top_candidates],
        key=lambda x: x[1]
    )
    return top_wolves, fitness_cache

def evaluate_single_wolf(wolf, fitness_cache, params_case_study,
                         params_fire, params_attack, params_sim_alg, path_extension):
    """
    Evaluates a single wolf (positioning) if not in cache.
    Returns (area_burned, sol_time, importance_vector)
    """
    pos_key = wolf.tobytes()
    if pos_key not in fitness_cache:
        sim_rng = np.random.default_rng(seed=params_sim_alg.random_seed)
        sim_seeds = sim_rng.integers(0, 1000, params_sim_alg.replications)
        pot_locations = params_case_study.pot_locations
        loc_to_idx = {loc: idx for idx, loc in enumerate(pot_locations)}
        num_locations = len(pot_locations)

        area_burned_ls = np.zeros(params_sim_alg.replications)
        response_time_ls = np.zeros(params_sim_alg.replications)
        sol_times_ls = np.zeros(params_sim_alg.replications)
        dispatch_counter = Counter()

        for i, seed in enumerate(sim_seeds):
            res = simulate(
                positioning=wolf,
                params_case_study=params_case_study,
                params_fire=params_fire,
                params_attack=params_attack,
                params_sim=params_sim_alg,
                random_seed=seed,
                bool_print=False,
                training=False,
            )
            area_burned_ls[i] = res["area_burned_rep"]
            response_time_ls[i] = res["response_times_all_fires"]
            sol_times_ls[i] = res["sol_time_rep"]

            if "dispatch_dict" in res:
                for key, cnt in res["dispatch_dict"].items():
                    loc = key[0] if isinstance(key, tuple) else key
                    if loc != "Lost":
                        dispatch_counter[loc] += cnt

        area_burned = area_burned_ls.mean()
        response_time = response_time_ls.mean()
        sol_time = sol_times_ls.mean()

        loc_counts = np.zeros(num_locations)
        for loc, cnt in dispatch_counter.items():
            if loc in loc_to_idx:
                loc_counts[loc_to_idx[loc]] += cnt
        total = loc_counts.sum()
        importance_vector = loc_counts / total if total > 0 else np.zeros(num_locations)
        
        if path_extension == "response_time":
            fitness_cache[pos_key] = {
                "fitness": float(response_time),
                "sol_time": float(sol_time),
                "dispatch_importance": importance_vector
            }
        else:
            fitness_cache[pos_key] = {
                "fitness": float(area_burned),
                "sol_time": float(sol_time),
                "dispatch_importance": importance_vector
            }

    else:
        if path_extension == "response_time":
            response_time = fitness_cache[pos_key]["fitness"]
        else:
            area_burned = fitness_cache[pos_key]["fitness"]
        
        importance_vector = fitness_cache[pos_key]["dispatch_importance"]

    if path_extension == "response_time":
        return fitness_cache, float(response_time), importance_vector
    else:
        return fitness_cache, float(area_burned), importance_vector

def evaluate_wolves(wolves, fitness_cache, params_case_study, params_fire, params_attack, params_sim_alg, size_elite=None, path_extension=None):
    """
    """
    
    # Ensure wolves is 2D
    single_wolf = False
    if isinstance(wolves, np.ndarray) and wolves.ndim == 1:
        wolves = wolves.reshape(1, -1)
        single_wolf = True

    sim_rng = np.random.default_rng(seed=params_sim_alg.random_seed)
    sim_seeds = sim_rng.integers(0, 1000, params_sim_alg.replications)
    pot_locations = params_case_study.pot_locations
    loc_to_idx = {loc: idx for idx, loc in enumerate(pot_locations)}
    num_locations = len(pot_locations)

    results = []
    top_candidates = []

    for indiv in wolves:
        indiv_arr = np.array(indiv, dtype=int)
        pos_key = indiv_arr.tobytes()

        # --- Retrieve from cache or evaluate ---
        if pos_key not in fitness_cache:
            area_burned_ls = np.zeros(params_sim_alg.replications)
            response_time_ls = np.zeros(params_sim_alg.replications)
            sol_times_ls = np.zeros(params_sim_alg.replications)
            dispatch_counter = Counter()

            for rep_idx, seed in enumerate(sim_seeds):
                res = simulate(
                    positioning=indiv_arr,
                    params_case_study=params_case_study,
                    params_fire=params_fire,
                    params_attack=params_attack,
                    params_sim=params_sim_alg,
                    random_seed=seed,
                    bool_print=False,
                    training=False,
                )
                area_burned_ls[rep_idx] = res["area_burned_rep"]
                response_time_ls[rep_idx] = res["response_times_all_fires"]
                sol_times_ls[rep_idx]   = res["sol_time_rep"]

                if "dispatch_dict" in res:
                    for key, cnt in res["dispatch_dict"].items():
                        loc = key[0] if isinstance(key, tuple) else key
                        if loc != "Lost":
                            dispatch_counter[loc] += cnt

            area_burned = float(area_burned_ls.mean())
            response_time = float(response_time_ls.mean())

            loc_counts = np.zeros(num_locations)
            for loc, cnt in dispatch_counter.items():
                if loc in loc_to_idx:
                    loc_counts[loc_to_idx[loc]] += cnt
            total = loc_counts.sum()
            importance_vector = loc_counts / total if total > 0 else np.zeros(num_locations)

            if path_extension == "response_time":
                fitness_cache[pos_key] = {
                    "fitness": response_time,
                    "dispatch_importance": importance_vector
                }
            else:
                fitness_cache[pos_key] = {
                    "fitness": area_burned,
                    "dispatch_importance": importance_vector
                }
            
        else:
            if path_extension == "response_time":
                response_time = fitness_cache[pos_key]["fitness"]
            else:
                area_burned = fitness_cache[pos_key]["fitness"]
            
            importance_vector = fitness_cache[pos_key]["dispatch_importance"]

        if path_extension == "response_time":
            results.append((indiv_arr, response_time, importance_vector))
        else:
            results.append((indiv_arr, area_burned, importance_vector))

        if size_elite is not None:
            if path_extension == "response_time":
                heapq.heappush(top_candidates, (-response_time, tuple(indiv_arr)))
            else:
                heapq.heappush(top_candidates, (-area_burned, tuple(indiv_arr)))
                
            if len(top_candidates) > size_elite:
                heapq.heappop(top_candidates)

    top_wolves = None
    if size_elite is not None:
        top_wolves = sorted([(list(indiv), -score) for score, indiv in top_candidates], key=lambda x: x[1])

    return top_wolves, fitness_cache, results[0] if single_wolf else results

