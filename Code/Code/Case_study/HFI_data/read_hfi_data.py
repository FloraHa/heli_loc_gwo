"""
This file contains the function to read HFI data.
"""

import pandas as pd
import geopandas as gpd
import fiona
from datetime import datetime
import warnings
from tqdm import tqdm
from shapely.geometry import Point

def calculate_weights(intensity):
    if int(intensity) <= 500: return 0
    elif int(intensity) <= 2000: return 1
    elif int(intensity) <= 4000: return 4
    else: return 8

def read_hfi_data_raw(date):
    """
    Read HFI data from a CSV file.
    """
    if date[1] == 6:
        month_string = "juin"
    elif date[1] == 7:
        month_string = "juillet"

    path = f"Data/donnees_florentina/donnees_florentina/{date[0]}_{month_string}.gdb"
    layers = fiona.listlayers(path)

    desired_date = datetime(date[0], date[1], date[2])

    desired_layer = desired_date.strftime('Intensite_%Y%m%d')


    if desired_layer in layers:
        hfi_data = gpd.read_file(path, layer=desired_layer).to_crs(epsg=4326)
    else:
        warnings.warn(f"No data found for {desired_date.date()}. Looking for the closest date with data.")

        # Extract dates from layer names and find the closest one
        layer_dates = []
        for layer in layers:
            try:
                # Assuming layer names are in the format 'Intensite_YYYYMMDD'
                layer_date_str = layer.split('_')[1]
                layer_date = datetime.strptime(layer_date_str, '%Y%m%d')
                layer_dates.append(layer_date)
            except (IndexError, ValueError):
                continue

        if not layer_dates:
            warnings.warn("No valid date layers found in the GDB file.")
            return gpd.GeoDataFrame()

        # Find the closest date
        closest_date = min(layer_dates, key=lambda date: abs(date - desired_date))
        closest_layer = closest_date.strftime('Intensite_%Y%m%d')

        warnings.warn(f"Using data from {closest_date.date()} instead.")
        hfi_data = gpd.read_file(path, layer=closest_layer).to_crs(epsg=4326)

    hfi_data['Intensite'] = hfi_data['Intensite'].apply(lambda x: pd.to_numeric(str(x).replace(',', '.'), errors='coerce'))
    hfi_data['Intensite'].fillna(0)

    # Assumption: If there are multiple entries for the same pixel, keep the one with the highest intensity
    hfi_data = hfi_data.sort_values("Intensite", ascending=False).drop_duplicates(subset="Quadr", keep="first")

    hfi_data["Weights"] = hfi_data["Intensite"].apply(calculate_weights)

    return hfi_data

def convert_raw_data_mclp(distance_df, coverage_range):
    """
    This function prepares the raw data for MCLP optimization.
    """
    pixels = distance_df["Quadr"].unique()
    facilities = distance_df["region_UID"].unique()
    primary_facility_ids = distance_df[distance_df["point_type"] == "Primary"]["region_UID"].unique()
    #primary_facility_ids = distance_df["region_UID"].unique()

    # Filter precomputed distances
    filtered_df = distance_df[
        (distance_df["distance_km"] <= coverage_range) &
        (distance_df["zone_compatible"])
    ]

    # Start with all pixels and assign empty lists
    pixel_covered_sites = {pixel: [] for pixel in distance_df["Quadr"].unique()}

    # Fill in only those with facilities in range
    covered = (
        filtered_df.groupby("Quadr")["region_UID"]
        .apply(list)
        .to_dict()
    )

    # Update the dictionary with actual covered facilities
    pixel_covered_sites.update(covered)
    
    # Calculate weights
    distance_df["weights"] = distance_df["Intensite"].apply(calculate_weights)
    weight_pixels = distance_df.set_index("Quadr")["weights"].to_dict()

    data = {}
    data["pixels"] = pixels
    data["facilities"] = facilities
    data["pixel_covered_sites"] = pixel_covered_sites
    data["weight_pixels"] = weight_pixels
    data["primary_facility_ids"] = primary_facility_ids

    return data

def read_file_hfi_preprocessed_data(date):
    """
    This file returns the preprocessed hfi data from the csv file (this is needed as input for the other functions)
    """
    if date[1] == 6:
        month_string = "juin"
    elif date[1] == 7:
        month_string = "juillet"

    if date[2] < 10:
        path = f"Data/output_tables/output_tables/FC_{date[0]}_{month_string}_Intensite_{date[0]}0{date[1]}0{date[2]}_ZPI_table.csv"
    else:
        path = f"Data/output_tables/output_tables/FC_{date[0]}_{month_string}_Intensite_{date[0]}0{date[1]}{date[2]}_ZPI_table.csv"

    hfi_data_from_csv = pd.read_csv(path)

    return hfi_data_from_csv

def convert_preprocessed_hfi_data(hfi_data_from_csv):
    """
    This file converts the preprocessed hfi from csv to a geo dataframe for plotting.
    """
    hfi_data = hfi_data_from_csv[["Intenst", "ORIG_X", "ORIG_Y"]]

    hfi_data = hfi_data.drop_duplicates(subset=['ORIG_X', 'ORIG_Y'])

    hfi_data['Intensite'] = hfi_data['Intenst'].apply(lambda x: pd.to_numeric(str(x).replace(',', '.'), errors='coerce'))
    hfi_data['Intensite'] = hfi_data['Intensite'].fillna(0)

    # Create a geometry column from ORIG_X and ORIG_Y
    geometry = [Point(xy) for xy in zip(hfi_data['ORIG_X'], hfi_data['ORIG_Y'])]

    # Convert to GeoDataFrame and set the CRS to EPSG:4326
    hfi_gdf = gpd.GeoDataFrame(hfi_data, geometry=geometry)
    hfi_gdf = hfi_gdf.set_crs(epsg=2138)
    hfi_gdf = hfi_gdf.to_crs(epsg=4326)

    hfi_gdf = hfi_gdf.drop(["Intenst", "ORIG_X", "ORIG_Y"], axis=1)

    hfi_gdf["Weights"] = hfi_gdf["Intensite"].apply(calculate_weights)

    return hfi_gdf

def convert_preprocessed_hfi_data_mclp(hfi_data_from_csv, coverage_range):
    """
    This function converts the preprocessed HFI data from a CSV file into a format suitable for the MCLP.
    """

    primary_bases_names = ["Baie-Comeau", "Broadback", "Chibougamau", "Chute-des-Passes", "Labrieville", "La Tuque", "Lac Joncas", "Lebel-sur-Quévillon",
                           "Manic-5", "Maniwaki", "Matagami", "Mudorchville", "NT Girardville", "Parent", "Rivière Bonnard", "Roberval",
                           "Sept-Îles", "Val-d'Or"]

    data_df = hfi_data_from_csv.dropna(subset=["ORIG_FI", "DEST_FID", "Nom_IDBASE", "Intenst", "LINK_DIST"])

    # Also drop everything with DEST_FID 46 (Saint-Bruno-de Guignes)
    data_df = data_df[data_df["DEST_FID"] != 46]

    facilities = data_df["DEST_FID"].unique()
    pixels = data_df["ORIG_FI"].unique()
    pixels = [pixel for pixel in pixels if not pd.isna(pixel)]

    primary_facility_ids = data_df[data_df["Nom_IDBASE"].isin(primary_bases_names)]["DEST_FID"].unique()

    pixel_covered_sites = {}
    for pixel in pixels:
        in_range = data_df[(data_df["ORIG_FI"] == pixel) & (data_df["LINK_DIST"] <= coverage_range)]
        pixel_covered_sites[pixel] = in_range["DEST_FID"].tolist()
        pixel_covered_sites[pixel] = [x for x in pixel_covered_sites[pixel] if x in facilities]

    data_df["weights"] = data_df["Intenst"].apply(calculate_weights)
    weight_pixels = data_df.set_index("ORIG_FI")["weights"].to_dict()

    data = {}
    data["pixels"] = pixels
    data["facilities"] = facilities
    data["pixel_covered_sites"] = pixel_covered_sites
    data["weight_pixels"] = weight_pixels
    data["primary_facility_ids"] = primary_facility_ids

    return data

def determine_color(intensite):
    if int(intensite) < 500:
        return 'royalblue'
    elif 500 <= int(intensite) < 2000:
        return 'limegreen'
    elif 2000 <= int(intensite) < 4000:
        return 'orange'
    else:
        return 'indianred'
    
def add_hfi(region_gdf, hfi_data, adj_factor):
    results = []
    
    # Ensure both GeoDataFrames have the same CRS
    hfi_data = hfi_data.to_crs(region_gdf.crs)

    # Iterate over each region
    for idx, region in tqdm(region_gdf.iterrows(), total=len(region_gdf)):
        # Filter hfi_data for the current region
        region_hfi_data = hfi_data[hfi_data.within(region.geometry)]

        # Extract the 'Weights' column
        weights = region_hfi_data['Weights']

        # Calculate the total number of weights
        total = len(weights)

        if total == 0:
            # If no valid weights, set all percentages to 0 and missing to 100
            pcts = {"W_0": 0.0, "W_1": 0.0, "W_4": 0.0, "W_8": 0.0, "Missing_prop": 100.0}
        else:
            # Count each weight category
            value_counts = weights.value_counts(normalize=True) * 100
            pcts = {
                "W_0": value_counts.get(0, 0.0),
                "W_1": value_counts.get(1, 0.0),
                "W_4": value_counts.get(4, 0.0),
                "W_8": value_counts.get(8, 0.0)
            }
            pcts["Missing_prop"] = 100 - sum(pcts.values())

        adjustment_factor = (
                pcts["W_0"] * adj_factor["Blue"] +
                pcts["W_1"] * adj_factor["Green"] +
                pcts["W_4"] * adj_factor["Orange"] +
                pcts["W_8"] * adj_factor["Red"]
            ) / 100

        # Store result
        results.append({
            "region_id": idx,
            "adjustment_factor": adjustment_factor,
            **pcts
        })

    # Create DataFrame for weight information
    weight_summary_df = pd.DataFrame(results)

    # Merge the weight information with the original GeoDataFrame
    region_gdf = region_gdf.join(weight_summary_df.set_index('region_id'), how='left')

    return region_gdf
