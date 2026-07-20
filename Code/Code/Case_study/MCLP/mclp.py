"""
This model is a first model to replicate the MCLP from Frédéric.
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import pulp
import csv
import os
from pulp import GUROBI

def optimize_MCLP(num_bases, data, date, path_extension):

    # print("Start optimization")

    model = pulp.LpProblem("Max_Coverage_Location_Problem", pulp.LpMaximize)

    covered_pixels = pulp.LpVariable.dicts("covered", data["pixels"], cat="Binary", lowBound=0, upBound=1)
    used_sites = pulp.LpVariable.dicts("site", data["facilities"], cat="Binary", lowBound=0, upBound=1)

    # Define the objective function
    model += pulp.lpSum(covered_pixels[pixel] * data["weight_pixels"][pixel] for pixel in data["pixels"])

    # Constraints
    for pixel in data["pixels"]:
        model += pulp.lpSum(used_sites[facility] for facility in data["pixel_covered_sites"][pixel]) >= covered_pixels[pixel]

    model += pulp.lpSum(used_sites[facility] for facility in data["facilities"]) <= num_bases

    if path_extension != "secondary_bases_adapted":
        for facility in data["facilities"]:
            if facility not in data["primary_facility_ids"]:
                 model += used_sites[facility] == 0

    status = model.solve(GUROBI(msg=False))

    # print(f"Status: {pulp.LpStatus[model.status]}")

    if pulp.LpStatus[model.status] == "Optimal":
        obj = pulp.value(model.objective)
        used_sites_result = {facility: used_sites[facility].varValue for facility in data["facilities"]}
        covered_pixels_result = {pixel: covered_pixels[pixel].varValue for pixel in data["pixels"]}

        print(f"Optimal Objective Value: {obj}")

        if date[2] < 10:
            path = f"Case_study/Results/MCLP/{path_extension}/{date[0]}_0{date[1]}_0{date[2]}"
        else:
            path = f"Case_study/Results/MCLP/{path_extension}/{date[0]}_0{date[1]}_{date[2]}"


        os.makedirs(path, exist_ok=True)
        with open(f"{path}/used_facilities.txt", "w") as f:
            f.write(str(used_sites_result))

        with open(f"{path}/covered_pixels.txt", "w") as f:
            f.write(str(covered_pixels_result))



