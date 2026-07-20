"""
This file reads in the FOP data and preprocesses it.
"""

import numpy as np
import rasterio
from rasterstats import zonal_stats

def read_fop_data(date, region_data, districts):
    """
    Reads in the FOP data for a given day.
    """
    day_string = f"{date[2]:02d}"
    month_string = f"{date[1]:02d}"
    date_string = f"{date[0]}{month_string}{day_string}"

    path_marginal_probability_fire_occurence = f"Data/fop_output/{date_string}/mPFi_MargProbUpperAtmosphericIndicesStrikeFire_{date_string}00_{date_string}.tif"
    path_unconditional_probability_heli_requirement = f"Data/fop_output/{date_string}/pLGFi_ProbLightningUpperAtmosphericIndicesCausedLargeFireGrid_{date_string}00_{date_string}.tif"

    path_marginal_probability_fire_occurence = f"Data/fop_output/{date_string}/mPFs_MargProbLightiningStrikeFire_{date_string}00_{date_string}.tif"
    path_unconditional_probability_heli_requirement = f"Data/fop_output/{date_string}/pLGFs_ProbLightningStrikeCausedLargeFireGrid_{date_string}00_{date_string}.tif"

    fop_lightening_path = f"Data/fop_output/{date_string}/mPFs_MargProbLightiningStrikeFire_{date_string}00_{date_string}.tif"

    # --- Project region_data to raster CRS ---
    with rasterio.open(path_marginal_probability_fire_occurence) as src:
        regions_proj = region_data.to_crs(src.crs)

    # --- Use zonal_stats (handles tiny overlaps properly) ---
    marge_stats = zonal_stats(
        regions_proj,
        path_marginal_probability_fire_occurence,
        stats=["sum"],
        nodata=-9999,
        geojson_out=False
    )

    uncond_stats = zonal_stats(
        regions_proj,
        path_unconditional_probability_heli_requirement,
        stats=["sum"],
        nodata=-9999,
        geojson_out=False
    )

    # Extract sums
    region_data["Forecast_fire_occurence"] = [s["sum"] if s["sum"] is not None else 0.0 for s in marge_stats]
    region_data["Forecast_helicopter_requirement"] = [s["sum"] if s["sum"] is not None else 0.0 for s in uncond_stats]

    # Conditional probability
    region_data["Forecast_cond_prob_heli_requirement"] = np.divide(
        region_data["Forecast_helicopter_requirement"],
        region_data["Forecast_fire_occurence"],
        out=np.zeros_like(region_data["Forecast_fire_occurence"]),
        where=region_data["Forecast_fire_occurence"] != 0
    )

    lambda_j = region_data.set_index("region_UID")["Forecast_fire_occurence"].to_dict()
    if districts is not None:
        lambda_j = {k: v for k, v in lambda_j.items() if k in districts}

    region_data = region_data.to_crs("EPSG:4326")

    return region_data, lambda_j, fop_lightening_path