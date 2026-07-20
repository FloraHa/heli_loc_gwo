"""
This file reads in data.
"""

import csv
import geopandas as gpd
import numpy as np

def read_params(folder_path, sep=",", zone_inc_only=False):
    """
    This function reads in the data from the csv  and shape files.
    """

    districts = []
    if zone_inc_only==True:
        path_demand = f"{folder_path}/districts_demand.csv"
    else: 
        path_demand = f"{folder_path}/districts_demand_inc.csv"
    with open(path_demand, newline='') as district_demand:
        reader_dd = csv.reader(district_demand)
        next(reader_dd)
        for row in reader_dd:
            districts.append(int(row[0]))

    pot_locations = []
    with open(f"{folder_path}/districts_potential_locations.csv", newline='') as district_pot_loc:
        reader_pl = csv.reader(district_pot_loc)
        next(reader_pl)
        for row in reader_pl:
            pot_locations.append(int(row[0]))

    lambda_districts_ls = []
    with open(f"{folder_path}/demand_rate.csv", newline='') as demand_rates:
        reader_ld = csv.reader(demand_rates)
        next(reader_ld)
        for row in reader_ld:
            lambda_districts_ls.append(float(row[0]))

    lambda_districts = {}
    for i in districts:
        lambda_districts[i] = lambda_districts_ls[i]

    travel_times = np.loadtxt(
        f"{folder_path}/travel_times.csv",
        delimiter=sep,
        skiprows=1,
    )

    travel_times_across_zones = np.loadtxt(
        f"{folder_path}/travel_times_across_zones.csv",
        delimiter=sep,
        skiprows=1,
    )

    regional_data = gpd.read_file(f"{folder_path}/Regional_data.shp").to_crs(4326)
    regional_data['region_UID'] = regional_data['region_UID'].astype(int)
    fires = gpd.read_file(f"{folder_path}/fires_within_zpi.shp").to_crs(4326)

    return districts, lambda_districts, pot_locations, travel_times, travel_times_across_zones, regional_data, fires
