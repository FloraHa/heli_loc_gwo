"""
This file contains the main for the optimization for the case study.
"""

"""
Libraries
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import json
import os
import scipy.stats as stats
import csv
from dataclasses import replace


from Code.Case_study.Validation.support_optimization_val import check_and_optimize
from Code.Case_study.Algorithm_Files_GWO_cont_cs.main_alg_gwo_cont_cs import grey_wolf_optimizer_cont


def optimize_gwo(date, params_case_study, params_fire, params_attack, params_sim_gwo, params_gwo, replication_id, path_extension):

    if params_gwo.initialize_gwo_mclp:
        bases_list_mclp = check_and_optimize(params_case_study.num_helicopters, date, params_case_study.hfi_data_mclp, params_case_study.hfi_data_base, params_gwo.optimize_mclp)
    else:
        bases_list_mclp = None

    params_gwo = replace(params_gwo, initial_wolf_mclp=bases_list_mclp)


    """
    Optimize
    """
    best_instances_df = pd.DataFrame(columns=["Replication Number", "Best Chromosome", "Fitness Value"])

    for rep in range(params_gwo.replication_gwo):
        random_seed = rep
 
        alpha_pos, alpha_score, best_scores, elapsed_time, fitness_cache = grey_wolf_optimizer_cont(params_case_study=params_case_study, params_fire=params_fire, params_attack=params_attack,
                                                                                           params_sim_gwo=params_sim_gwo, params_gwo=params_gwo, random_seed=random_seed, path_extension=path_extension)

        # Create a DataFrame for the best instance
        best_instance_df = pd.DataFrame({
            "Replication Number": [rep],
            "Best Chromosome": [alpha_pos],
            "Fitness Value": [alpha_score]
        })

        # Append the best instance to the main DataFrame
        best_instances_df = pd.concat([best_instances_df, best_instance_df], ignore_index=True)
        print(f"Time required for GWO: {elapsed_time}")
        print(f"Replication {rep}")

    best_solution = best_instances_df.loc[best_instances_df['Fitness Value'].idxmin()]
    best_positioning = best_solution['Best Chromosome'][:params_case_study.num_helicopters]

    if date[2] < 10:
        path_dir = f"Case_study/Results/GWO/HFI_prob/{path_extension}/{date[0]}_0{date[1]}_0{date[2]}_population{params_gwo.population_size}_maxiter{params_gwo.max_iter}_repsim{params_sim_gwo.replications}_repid{replication_id}"
    else:
        path_dir = f"Case_study/Results/GWO/HFI_prob/{path_extension}/{date[0]}_0{date[1]}_{date[2]}_population{params_gwo.population_size}_maxiter{params_gwo.max_iter}_repsim{params_sim_gwo.replications}_repid{replication_id}"

    os.makedirs(path_dir, exist_ok=True)
    with open(f"{path_dir}/Result.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(best_positioning)
    
    with open(f"{path_dir}/Time.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([elapsed_time])
        
    with open(f"{path_dir}/Value.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([best_solution])
        

    fitness_only = {
        tuple(np.frombuffer(pos_key, dtype=np.int64)): entry["fitness"]
        for pos_key, entry in fitness_cache.items()
    }
        
    df = pd.DataFrame.from_dict(
        fitness_only,
        orient="index",
        columns=["fitness"]
    )
    
    df.to_csv(f"{path_dir}/fitness_values.csv")


    return best_positioning, elapsed_time