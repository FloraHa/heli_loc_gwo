"""
This file contains the support function for the validation using optimization.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pyproj
import ast
import os
import ast
import re

from Code.Case_study.MCLP.mclp import optimize_MCLP


def read_base_data(path):
    opti_data_df = pd.read_csv(path, low_memory=False)
    data_df_bases = opti_data_df.drop_duplicates(subset=["DEST_FID"])

    crs_code = "EPSG:2138"

    transformer = pyproj.Transformer.from_crs(crs_code, "EPSG:4326", always_xy=True)
    lon_crs, lat_crs = transformer.transform(
        data_df_bases["DEST_X"].values, 
        data_df_bases["DEST_Y"].values
    )

    data_df_bases = data_df_bases.assign(Con_Lon=lon_crs, Con_Lat=lat_crs)

    data_df_bases.loc[:, 'Nom_IDBASE'] = data_df_bases['Nom_IDBASE'].str.replace(":", "", regex=False)

    return data_df_bases

def read_convert_solution(path):
    with open(path, "r") as f:
        dict_str = f.read()

    dict_str_fixed = dict_str.replace("np.int64(", "").replace(")", "")

    return ast.literal_eval(dict_str_fixed)

def read_optimization_results(date, folder_results="Case_study/Results/MCLP", path_extension=""):
    if date[1] == 6:
        month_string = "juin"
    if date[1] == 7:
        month_string = "juillet"

    if date[2] < 10:
        # Read the basis data that has been used for optimization
        data_df_bases = read_base_data(f"Data/output_tables/output_tables/FC_{date[0]}_{month_string}_Intensite_{date[0]}0{date[1]}0{date[2]}_ZPI_table.csv")
    else:
        data_df_bases = read_base_data(f"Data/output_tables/output_tables/FC_{date[0]}_{month_string}_Intensite_{date[0]}0{date[1]}{date[2]}_ZPI_table.csv")

    if date[2] < 10:
        # Read the optimization results
        used_facilities = read_convert_solution(f"{folder_results}/{path_extension}/{date[0]}_0{date[1]}_0{date[2]}/used_facilities.txt")
    else:
        used_facilities = read_convert_solution(f"{folder_results}/{path_extension}/{date[0]}_0{date[1]}_{date[2]}/used_facilities.txt")

    # used_facilities = read_convert_solution("Code/Case_study/MCLP/Test_results/processed/used_facilities.txt")

    # Merge the data into a single DataFrame
    selected_columns = ["DEST_FID", "DEST_X", "DEST_Y", "Nom_IDBASE", "Con_Lon", "Con_Lat"]
    unique_bases_df = data_df_bases[selected_columns].drop_duplicates(subset="DEST_FID")

    # Convert the dictionary to a DataFrame
    used_facilities_df = pd.DataFrame(list(used_facilities.items()), columns=['DEST_FID', 'Opt_result'])

    # Merge the two DataFrames on 'DEST_FID'
    result_df = pd.merge(unique_bases_df, used_facilities_df, on='DEST_FID', how='left')

    regional_data = gpd.read_file("Case_study/Processed_data/Regional_data.shp")

    filtered_result_df = result_df[result_df['Opt_result'] > 0]

    # Find the index of the selected facilities
    geometry = [Point(xy) for xy in zip(filtered_result_df['Con_Lon'], filtered_result_df['Con_Lat'])]
    bases_gdf = gpd.GeoDataFrame(filtered_result_df, geometry=geometry, crs=regional_data.crs)

    # Perform a spatial join to find which region each base is in
    joined_gdf = gpd.sjoin(bases_gdf, regional_data, how="left", predicate="within")

    # Initialize an empty list to store the indices
    index_list = []

    # Iterate over each row in the joined GeoDataFrame
    for index, row in joined_gdf.iterrows():
        n = int(row['Opt_result'])
        # Append the index of the regional_data to the list n times
        if not pd.isna(row['index_right']):  # Check if there is a match
            index_list.extend([row['index_right']] * n)

    return index_list

def check_and_optimize(num_servers, date, hfi_data_mclp, hfi_base, optimize_mclp=True, path_extension=""):
    # Determine the expected results path
    if date[2] < 10:
        results_path = f"Case_study/Results/MCLP/{path_extension}/{date[0]}_0{date[1]}_0{date[2]}/used_facilities.txt"
    else:
        results_path = f"Case_study/Results/MCLP/{path_extension}/{date[0]}_0{date[1]}_{date[2]}/used_facilities.txt"

    # Check if results exist
    if not os.path.exists(results_path) or optimize_mclp:
        print(f"Results not found for {date[0]}-{date[1]}-{date[2]}. Running optimization...")
        optimize_MCLP(num_servers, hfi_data_mclp, date, path_extension)
    else:
        print(f"Using existing results for {date[0]}-{date[1]}-{date[2]}.")

    # Read results (whether newly optimized or existing)
    if hfi_base == "preprocessed":
        bases_list_mclp = read_optimization_results(date)
    elif hfi_base == "raw":
        bases_list_mclp = read_facility_list_MCLP_raw(date)
    return bases_list_mclp


def read_facility_list_MCLP_raw(date, folder_results="Case_study/Results/MCLP", path_extension=""):
    """
    Reads the results from the MCLP with the raw data
    """
    if date[2] < 10:
        path = f"{folder_results}/{path_extension}/{date[0]}_0{date[1]}_0{date[2]}/used_facilities.txt"
    else:
        path = f"{folder_results}/{path_extension}/{date[0]}_0{date[1]}_{date[2]}/used_facilities.txt"

    # path = "Code/Case_study/MCLP/Test_results/raw/used_facilities.txt"

    with open(path, "r") as f:
        content = f.read()

    cleaned = re.sub(r'np\.int64\((\d+)\)', r'\1', content)

    # Safely evaluate the string to a dictionary
    covered_pixels_dict = ast.literal_eval(cleaned)

    # Extract keys with value 1.0
    used_facilities = [int(k) for k, v in covered_pixels_dict.items() if v == 1.0]

    return used_facilities