"""
This is the main to test the model on snellius
"""

"""
This file is the main to automatically validate the results of the optimization.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import argparse
from dataclasses import replace
import os

from Code.Case_study.Validation.automatic_gwo_support import validate_day_gwo
from Code.Case_study.Support_Files_cs.classes_parameters import ParamsGWO, ParamsFireBehavior, ParamsFireAttack, ParamsSimulation, ParamsCaseStudy

"""
Set the simulation parameters
"""

def main(experiment_id):
    experiment_id = int(experiment_id)
    
    date_id = experiment_id // (20*1)
    remainder = experiment_id % (20*1)
    param_id = remainder // 20
    replication_id = remainder % 20
    
    path_extension = "penalty_2_6"
    

    """
    Set the parameters for the fire behavior
    """

    params_fire = ParamsFireBehavior(initial_area_distribution={"Distribution": "Lognormal", "Parameters": {"s": 2.0157767088127527, "loc": 0.0, "scale": 0.9551506373508084}},     # This distribution is for the area in ha,
                           ROS_distribution={"Distribution": "Lognormal", "Parameters": {"s": 0.937232429661267, "loc": 0.0, "scale": 3.1602594461771907}},               # This distribution is for the spread in m/min,
                           HFI_factor_ROS={"Blue": 0.435899, "Green": 3.427957, "Yellow": 5.902193, "Red": 11.949297},
                           probabilities_heli_deploy_HFI={"Blue": 0.530943, "Green": 0.526435, "Yellow": 0.648265, "Red": 0.744479},
                           adjust_fact_lambda_HFI={'Blue': 0.697402104355971,
                                                   'Green': 1.4367020266265311,
                                                   'Orange': 1.258628182175241,
                                                   'Red': 2.123186403463939},
                           fop=True,
                           penalty=2*60,
                           method_update_ROS="hfi_index",   # Alternative: hfi_index, propagation_index
                           factor_arrival=1.0,
                           factor_my=1.0,
                           factor_ROS=1.0)           

    params_attack = ParamsFireAttack(travel_speed=226, # km/h
                                     base_speed_1=4, # m/min, 4
                                     base_speed_2=3, # m/min, 3
                                     setup_time=0.0, # min
                                     dispatch_range=113) # km
    
    params_sim_validation = ParamsSimulation(replications=100000, random_seed=1+replication_id, SIMTIME=(100*12*60), min_num_fires_sim=0)
    params_sim_validation = ParamsSimulation(replications=10, random_seed=1+replication_id, SIMTIME=(100*12*60), min_num_fires_sim=0)
    
    params_sim_gwo = ParamsSimulation(replications=100, random_seed=10000+replication_id, SIMTIME=(100*12*60), min_num_fires_sim=100)
    
    params_gwo = ParamsGWO(replication_gwo=1, population_size=100, size_elite=3, max_iter=10, initialize_gwo_hist_heur = True, initialize_gwo_mclp=True, optimize_mclp=False, initial_wolf_hist_heur=None, initial_wolf_mclp=None, threshold=0.8)
    
    params_case_study = ParamsCaseStudy(hfi_data_base="raw", 
                                        hfi_data_mclp={},
                                        max_helicopter_per_district=2,
                                        districts=[],
                                        lambda_j={},
                                        pot_locations=[],
                                        travel_distances={},
                                        travel_distances_across_zones={},
                                        regional_data=None,
                                        num_helicopters=0,
                                        zone_inc_only=True,
                                        threshold_other_heli=150.0,
                                        secondary_bases=False)

    

    """
    Testing parameters
    """

    hfi_date = [(2021, 6, 3), (2021, 6, 8), (2021, 6, 11), (2021, 6, 12), (2021, 6, 14), (2021, 6, 15), (2021, 6, 17),
            (2021, 7, 5), (2021, 7, 10), (2021, 7, 13), (2021, 7, 21), (2021, 7, 24), (2021, 7, 25),
            (2022, 6, 7), (2022, 6, 26), (2022, 6, 28), (2022, 7, 10), (2022, 7, 15), (2022, 7, 16), (2022, 7, 17)]
            
    population_sizes = [25]

    max_iters = [40]
    
    replications_gwo = [100]
                        
                        
    date = hfi_date[date_id]
    params_gwo = replace(params_gwo, population_size = population_sizes[param_id])
    params_gwo = replace(params_gwo, max_iter = max_iters[param_id])
    params_sim_gwo = replace(params_sim_gwo, replications = replications_gwo[param_id])
    
    
    # Adjust the parameters:
    if path_extension == "arrival_0.8":
        params_fire = replace(params_fire, factor_arrival=0.8)
    elif path_extension == "arrival_1.2":
        params_fire = replace(params_fire, factor_arrival=1.2)
    elif path_extension == "arrival_1.5":
        params_fire = replace(params_fire, factor_arrival=1.5)
    elif path_extension == "ROS_0.5":
        params_fire = replace(params_fire, factor_ROS=0.5)
    elif path_extension == "ROS_0.8":
        params_fire = replace(params_fire, factor_ROS=0.8)
    elif path_extension == "ROS_1.2":
        params_fire = replace(params_fire, factor_ROS=1.2)
    elif path_extension == "ROS_1.5":
        params_fire = replace(params_fire, factor_ROS=1.5)
    elif path_extension == "base_speed_p1":
        params_attack = replace(params_attack, base_speed_1=5)
        params_attack = replace(params_attack, base_speed_2=4)
    elif path_extension == "base_speed_p2":
        params_attack = replace(params_attack, base_speed_1=6)
        params_attack = replace(params_attack, base_speed_2=5)
    elif path_extension == "travel_263":
        params_attack = replace(params_attack, travel_speed=263) 
    elif path_extension == "travel_300":
        params_attack = replace(params_attack, travel_speed=300)
    elif path_extension == "secondary_bases_adapted":
        params_case_study = replace(params_case_study, secondary_bases=True)
    elif path_extension == "penalty_4":
        params_fire = replace(params_fire, penalty=4*60)
    elif path_extension == "penalty_6":
        params_fire = replace(params_fire, penalty=6*60)
    elif path_extension == "penalty_2_6":
        params_fire = replace(params_fire, penalty=[120, 360])
    


    """
    Set sundry parameters
    """
  
    validate_day_gwo(date=date, params_fire=params_fire, params_attack=params_attack,
                                params_sim_validation=params_sim_validation, params_sim_gwo=params_sim_gwo,
                                params_gwo=params_gwo, params_case_study=params_case_study,
                                plot=False, experiment_id=experiment_id, replication_id=replication_id, path_extension=path_extension)
  

if __name__ == "__main__":    
  parser = argparse.ArgumentParser()    
  parser.add_argument("--experiment_id", type=str, required=True, help="ID of the experiment to run")   
  args = parser.parse_args()    
  main(args.experiment_id)
