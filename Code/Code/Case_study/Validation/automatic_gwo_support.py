"""
This file stores the function for automatic validation
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import csv
from scipy.stats import lognorm
from dataclasses import replace

from Code.Case_study.Support_Files_cs.read_data_cs import read_params
from Code.Case_study.Plotting.plot_zone import preprocess_fire_data
from Code.Case_study.HFI_data.read_hfi_data import read_hfi_data_raw, convert_raw_data_mclp, read_file_hfi_preprocessed_data, convert_preprocessed_hfi_data, convert_preprocessed_hfi_data_mclp, determine_color, add_hfi
from Code.Case_study.Validation.support_historic_val import read_bases, get_bases_list, simulate_validation
from Code.Case_study.Algorithm_hist_deployment.get_historic_deployment_heuristic import get_positioning_historic, get_positioning_historic_secondary, positioning_to_list, remove_helicopters
from Code.Case_study.Validation.support_optimization_val import read_optimization_results, read_facility_list_MCLP_raw
from Code.Case_study.MCLP.mclp import optimize_MCLP
from Code.Case_study.Algorithm_Files_GWO_cs.gwo import optimize_gwo
from Code.Case_study.FOP_data.read_fop_data import read_fop_data
from Code.Case_study.Evaluation_Files_GWO_cs.eval_support import RegionCache

def validate_day_gwo(date, params_fire, params_attack, params_sim_validation, params_sim_gwo, params_gwo, params_case_study,
                 plot, experiment_id, replication_id, path_extension):
    
    """
    Inputs:

    """
    
    print(path_extension)

    """
    Preprocess the data
    """

    districts, lambda_j, pot_locations, travel_distances, travel_distances_across_zones, regional_data, fires = read_params("Case_study/Processed_data", zone_inc_only=params_case_study.zone_inc_only)
    zpi = gpd.read_file("Data/donnees_sopfeu_opti/donnees_sopfeu_opti/ZPI.shp", layer='ZPI').to_crs(4326)

    fire_data = pd.read_excel('Data/donnees_feu_frederic/bdobjop_2022INC_cropped.xlsx')
    gdf_fire_data = gpd.GeoDataFrame(
        fire_data, geometry=gpd.points_from_xy(fire_data.iLon, fire_data.iLat), crs="EPSG:4326"
    )

    gdf_fire_data = preprocess_fire_data(gdf_fire_data, date)
    if params_case_study.hfi_data_base == "preprocessed":
        hfi_csv = read_file_hfi_preprocessed_data(date)
        hfi_data = convert_preprocessed_hfi_data(hfi_csv)
    elif params_case_study.hfi_data_base == "raw":
        hfi_data = read_hfi_data_raw(date)

    hfi_data['color'] = hfi_data['Intensite'].apply(determine_color)
    regional_data = add_hfi(regional_data, hfi_data, params_fire.adjust_fact_lambda_HFI)

    # Read the HFI linked to distances for MCLP and the historic deployment heuristic
    if params_case_study.hfi_data_base == "preprocessed":
        data_mclp = convert_preprocessed_hfi_data_mclp(hfi_csv, params_attack.dispatch_range)
    elif params_case_study.hfi_data_base == "raw":
        distance_df = pd.read_csv(f"Case_study/Processed_data/HFI_distances/hfi_distance_{date[0]}_{date[1]}_{date[2]}.csv")
        data_mclp = convert_raw_data_mclp(distance_df, params_attack.dispatch_range)
        # print(data_mclp["primary_facility_ids"])

    if params_fire.fop:
        regional_data, lambda_j, fop_lightening_path = read_fop_data(date, regional_data, districts)
    else:
        for district_id in lambda_j.keys():
            dist_adjustment = regional_data.loc[district_id, "adjustment_factor"]
            lambda_j[district_id] *= dist_adjustment

    lambda_j = {k: v / (12*60) for k, v in lambda_j.items()}  # Conversion in minutes assuming a day from 8:00 to 20:00 
    lambda_j = {k: v * params_fire.factor_arrival for k, v in lambda_j.items()}  # Increase arrival rate

    regional_data_sim = regional_data[["region_UID", 'W_0', 'W_1', 'W_4', 'W_8', 'Forecast_cond_prob_heli_requirement', 'P_0', 'P_1', 'P_2', 'P_3', 'P_4', 'P_5']]
    regional_data_sim = RegionCache(regional_data_sim)

    """
    Read historic data
    """

    # Read the bases for the historic deployment

    print("Read in historic deployment")

    bases_df = read_bases()
    bases_list_hist = get_bases_list(bases_df, date)
    print(bases_list_hist)
    num_bases = len(bases_list_hist)

    # Update the params for the case study
    params_case_study = replace(params_case_study, districts=districts)
    params_case_study = replace(params_case_study, lambda_j=lambda_j)
    if path_extension == "secondary_bases_adapted":
        params_case_study = replace(params_case_study, pot_locations=districts)
    else:     
        params_case_study = replace(params_case_study, pot_locations=pot_locations)
    params_case_study = replace(params_case_study, travel_distances=travel_distances)
    params_case_study = replace(params_case_study, travel_distances_across_zones=travel_distances_across_zones)
    params_case_study = replace(params_case_study, regional_data=regional_data_sim)
    params_case_study = replace(params_case_study, num_helicopters=num_bases)
    params_case_study = replace(params_case_study, hfi_data_mclp=data_mclp)


    """
    Get historic deployment heuristic
    """
    
    if path_extension =="secondary_bases_adapted":
        bases_list_hist_heuristic = get_positioning_historic_secondary(distance_df, regional_data, fop_lightening_path, threshold_other_heli=params_case_study.threshold_other_heli, travel_distances=params_case_study.travel_distances, threshold_distance=params_attack.dispatch_range, thresholds=[2001, 4001], threshold_prob_lightening=0.5)
   
    else:
        bases_list_hist_heuristic_data = get_positioning_historic(distance_df, regional_data, fop_lightening_path, threshold_distance=params_attack.dispatch_range, thresholds=[2001, 4001])
        if path_extension.startswith("remove_helicopters"):
            x = int(path_extension.split("_")[-1])
            bases_list_hist_heuristic_data = remove_helicopters(
                bases_list_hist_heuristic_data, x
            )
    bases_list_hist_heuristic = positioning_to_list(bases_list_hist_heuristic_data)
    params_case_study = replace(params_case_study, num_helicopters=len(bases_list_hist_heuristic))
    print(bases_list_hist_heuristic)

    """
    Optimize and evaluate MCLP
    """

    print("Start MCLP optimization")
    optimize_MCLP(params_case_study.num_helicopters, data_mclp, date, path_extension)

    # Simulate with the optimized deployment

    if params_case_study.hfi_data_base == "preprocessed":
        bases_list_mclp = read_optimization_results(date, path_extension=path_extension)
    elif params_case_study.hfi_data_base == "raw":
        bases_list_mclp = read_facility_list_MCLP_raw(date, path_extension=path_extension)
        

    """
    Optimize and evaluate GWO
    """
    # Optimize the GWO with the historic number of bases
    print("Start optimization GWO")
    if params_gwo.initialize_gwo_hist_heur:
        params_gwo = replace(params_gwo, initial_wolf_hist_heur=bases_list_hist_heuristic)
    
    time_gwo = 0

    bases_list_gwo, time_gwo = optimize_gwo(date, params_case_study, params_fire, params_attack, params_sim_gwo, params_gwo, replication_id, path_extension)

    # Simulate with the optimized deployment

    bases_list_gwo = []

    if date[2] < 10:
        path_dir = f"Case_study/Results/GWO/HFI_prob/{path_extension}/{date[0]}_0{date[1]}_0{date[2]}_population{params_gwo.population_size}_maxiter{params_gwo.max_iter}_repsim{params_sim_gwo.replications}_repid{replication_id}"
    else:
        path_dir = f"Case_study/Results/GWO/HFI_prob/{path_extension}/{date[0]}_0{date[1]}_{date[2]}_population{params_gwo.population_size}_maxiter{params_gwo.max_iter}_repsim{params_sim_gwo.replications}_repid{replication_id}"

    with open(f"{path_dir}/Result.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            bases_list_gwo.append(row)
    bases_list_gwo = [int(item) for sublist in bases_list_gwo for item in sublist]
    
    
