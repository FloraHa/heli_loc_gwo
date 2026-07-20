"""
This file contains the support function for the validation using historic deployment.
"""

import pandas as pd
import datetime
import numpy as np
from collections import defaultdict
import scipy.stats as stats

from Code.Case_study.Evaluation_Files_GWO_cs.eval_des import simulate

def read_bases():
    """
    This function reads the bases.
    """

    data = pd.read_excel("Data/bases_sopfeu_2018-2022_VF_index.xlsx")

    dates = data.iloc[:, 0]  # Column A contains dates
    bases_data = data.iloc[:, 1:]  # The rest of the columns contain base indices

    # Convert the dates to a datetime format
    dates = pd.to_datetime(dates, errors='coerce')

    bases_df = pd.DataFrame({
        'Date': dates,
        'Base_Indices': bases_data.apply(lambda row: row.dropna().astype(int).tolist(), axis=1)
    })
    return bases_df

def get_bases_list(bases_df, date):
    """
    This function returns the list of the used bases for a certain day
    """
    selected_date = datetime.datetime(date[0], date[1], date[2])
    selected_row = bases_df[bases_df['Date'] == pd.to_datetime(selected_date)]
    used_bases_indices = selected_row.iloc[0]['Base_Indices']
    return used_bases_indices
    
def margin_error(array, confidence=0.95):
    m, s, n = np.mean(array), np.std(array, ddof=1), len(array)
    t = stats.t.ppf((1 + confidence) / 2, df= n - 1)
    e = t * (s / np.sqrt(n))
    return e


def simulate_validation(positioning, params_case_study, params_fire, params_attack, params_sim, historic_validation=False, historic_fires=None, path=""):

    # Create a rng for the simulation
    sim_rng = np.random.default_rng(seed=params_sim.random_seed)
    sim_seeds = sim_rng.integers(0, 1000, params_sim.replications)

    area_burned_ls = np.zeros(params_sim.replications)
    area_burned_attacked_ls = np.zeros(params_sim.replications)
    area_burned_escaped_ls = np.zeros(params_sim.replications)
    sol_times_ls = np.zeros(params_sim.replications)
    distance_fires_ls = np.zeros(params_sim.replications)
    num_fires_ls = np.zeros(params_sim.replications)
    num_fires_within_range_ls = np.zeros(params_sim.replications)
    response_times_ls = np.zeros(params_sim.replications)

    area_burned_ls_anti = np.zeros(params_sim.replications)
    sol_times_ls_anti = np.zeros(params_sim.replications)
    distance_fires_ls_anti = np.zeros(params_sim.replications)
    num_fires_ls_anti = np.zeros(params_sim.replications)
    num_fires_within_range_ls_anti = np.zeros(params_sim.replications)

    fire_counts = defaultdict(int)
    for rep in range(params_sim.replications):
        seed = sim_seeds[rep]
        results_sim = simulate(positioning=positioning, params_case_study=params_case_study,
                                            params_fire=params_fire, params_attack=params_attack, params_sim=params_sim, random_seed=seed,
                                            bool_print=False, historic_validation=historic_validation, historic_fires=historic_fires)

        area_burned_ls[rep] = results_sim["area_burned_rep"]
        area_burned_attacked_ls[rep] = results_sim["area_burned_attacked_rep"]
        area_burned_escaped_ls[rep] = results_sim["area_burned_escaped_rep"]
        sol_times_ls[rep] = results_sim["sol_time_rep"]
        distance_fires_ls[rep] = results_sim["distance_fires_rep"]
        num_fires_ls[rep] = results_sim["num_fires_rep"]
        num_fires_within_range_ls[rep] = results_sim["num_fires_within_range_rep"]
        response_times_ls[rep] = results_sim["response_times_all_fires"]

        # Combine fire counts
        for district in set(results_sim["fire_counts"]):
            fire_counts[district] += results_sim["fire_counts"].get(district, 0)
            
        burned_area_margin = margin_error(area_burned_ls)
        distance_margin = margin_error(distance_fires_ls)
        fires_within_range_margin = margin_error(num_fires_within_range_ls)
        
    results_validation = {
        "area_burned": area_burned_ls,
        "area_burned_attacked": area_burned_attacked_ls,
        "area_burned_escaped": area_burned_escaped_ls,
        "sol_time": sol_times_ls,
        "distance_fires": distance_fires_ls,
        "num_fires": num_fires_ls,
        "num_fires_within_range": num_fires_within_range_ls,
        "total_response_times": response_times_ls
    }
    
    results_validation = pd.DataFrame.from_dict(results_validation)
        
    results_validation.to_csv(f"Test_results_validation/{path}_validation.csv", index=False)

    results = {"area_burned": np.mean(area_burned_ls), "area_burned_margin": burned_area_margin, "area_burned_attacked": np.mean(area_burned_attacked_ls), "area_burned_escaped": area_burned_escaped_ls, "sol_time": np.mean(sol_times_ls),
               "distance_fires": np.mean(distance_fires_ls), "distance_fires_margin": distance_margin, "num_fires": np.mean(num_fires_ls),
               "num_fires_within_range": np.mean(num_fires_within_range_ls), "fires_within_range_margin": fires_within_range_margin,
               "fire_counts": fire_counts, "total_response_times": np.mean(response_times_ls)}

    return results