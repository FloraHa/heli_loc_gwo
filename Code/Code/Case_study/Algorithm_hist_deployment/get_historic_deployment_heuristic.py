"""
This file contains the functions to calculate the historic deployment based on the heuristic.
"""

from shapely import Point
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd

def get_lighthening_prob(regional_data, fop_path_lightening, threshold_distance=113):
    site_data = regional_data[["region_UID", "lat_site", "lon_site"]]
    geometry = [Point(lon, lat) for lon, lat in zip(site_data["lon_site"], site_data["lat_site"])]
    gdf_site = gpd.GeoDataFrame(site_data, geometry=geometry, crs=regional_data.crs)
    gdf_site = gdf_site.to_crs(epsg=3857)
    gdf_site['geometry_buffer'] = gdf_site['geometry'].buffer(threshold_distance*1000)

    with rasterio.open(fop_path_lightening) as src:
        gdf_site_buffers = gdf_site.set_geometry('geometry_buffer').to_crs(src.crs)

        # Calculate the sum of raster values for each buffer
        lightening_prob = zonal_stats(
            vectors=gdf_site_buffers,
            raster=fop_path_lightening,
            stats=['sum'],
            geojson_out=False,
            nodata=-9999,
            affine=src.transform,
            all_touched=True
        )
    return lightening_prob

    
def get_positioning_historic(df_distance_hfi, regional_data, fop_path_lightening, threshold_distance=113, thresholds=[2001, 4001], threshold_prob_lightening=0.5):
    """
    """
    # Get the lightening prob
    lightening_prob = get_lighthening_prob(regional_data, fop_path_lightening, threshold_distance)

    primary_df = df_distance_hfi[df_distance_hfi['point_type'] == 'Primary']
    
    positioning = {}

    # Iterate over each region_UID
    for region in primary_df['region_UID'].unique():
        region_data = df_distance_hfi[df_distance_hfi['region_UID'] == region]
        # Filter Quadrants within the distance threshold
        within_threshold = region_data[region_data['distance_km'] <= threshold_distance]
        num_pixels = len(within_threshold)
        if not within_threshold.empty:
            # Calculate share of Quadrants with Intensite > threshold
            share_1 = len(within_threshold[within_threshold['Intensite'] > thresholds[0]])/num_pixels if num_pixels > 0 else 0
            share_2 = len(within_threshold[within_threshold['Intensite'] > thresholds[1]])/num_pixels if num_pixels > 0 else 0

            if share_2 >= 0.5 and float(lightening_prob[region]["sum"]) >= threshold_prob_lightening:
                helicopters = 2
            elif share_2 >= 0.5 and float(lightening_prob[region]["sum"]) < threshold_prob_lightening:
                helicopters = 1
            elif share_1 >= 0.4:
                helicopters = 1
            else:
                helicopters = 0
        
            if helicopters > 0:
                positioning[region] = {
                    "count": helicopters,
                    "share_1": share_1,
                    "share_2": share_2
                }
        
    return positioning

def remove_helicopters(positioning, x):
    """
    Remove up to x helicopters based on priority rules:
    1. Remove from regions with 2 helicopters (lowest share_2 first)
    2. If still need to remove more, remove from regions with 1 helicopter (lowest share_1 first)
    """

    positioning = positioning.copy()  # avoid mutating original

    # --- Step 1: Regions with 2 helicopters ---
    regions_two = [
        (region, data["share_2"])
        for region, data in positioning.items()
        if data["count"] == 2
    ]

    # Sort ascending ? lowest share_2 first
    regions_two.sort(key=lambda x: x[1])

    removed = 0

    for region, _ in regions_two:
        if removed >= x:
            break

        positioning[region]["count"] -= 1
        removed += 1

    # --- Step 2: Regions with 1 helicopter ---
    if removed < x:
        regions_one = [
            (region, data["share_1"])
            for region, data in positioning.items()
            if data["count"] == 1
        ]

        # Sort ascending ? lowest share_1 first
        regions_one.sort(key=lambda x: x[1])

        for region, _ in regions_one:
            if removed >= x:
                break

            positioning[region]["count"] -= 1
            removed += 1

    # --- Cleanup: remove regions with 0 helicopters ---
    positioning = {
        region: data
        for region, data in positioning.items()
        if data["count"] > 0
    }

    return positioning

def positioning_to_list(positioning):
    result = []
    for region, data in positioning.items():
        result.extend([region] * data["count"])
    return result

    
def count_close_bases(candidate, selected, threshold_distance_heli, travel_distance):
        count = 0
        for s in selected:
            dist = travel_distance[int(candidate["region_UID"])][int(s["region_UID"])]
            if dist <= threshold_distance_heli:
                count += 1
        return count

def try_select(row, selected, threshold_other_heli, travel_distance):
    close_count = count_close_bases(row, selected, threshold_other_heli, travel_distance)

    # default: no nearby bases allowed
    allowed_close = 0

    # relaxed rule
    if row["share_red"] >= 0.5 and row["lightning_prob"] >= 0.5:
        allowed_close = 1

    if close_count <= allowed_close:
        selected.append(row)    

def get_positioning_historic_secondary(df_distance_hfi, regional_data, fop_path_lightening, threshold_other_heli, travel_distances, threshold_distance=113, thresholds=[2001, 4001], threshold_prob_lightening=0.5):
    """
    This is the heuristic for the secondary bases
    """
    # Get the lightening prob
    lightening_prob = get_lighthening_prob(regional_data, fop_path_lightening, threshold_distance)
    
    records = []

    for region, g in df_distance_hfi.groupby("region_UID"):
        within = g[g["distance_km"] <= threshold_distance]
        num_pixels = len(within)

        if num_pixels == 0:
            continue

        share_red_yellow = (
            (within["Intensite"] > thresholds[0]).sum() / num_pixels
        )
        share_red = (
            (within["Intensite"] > thresholds[1]).sum() / num_pixels
        )

        records.append({
            "region_UID": int(region),
            "share_red": share_red,
            "share_red_yellow": share_red_yellow,
            "lightning_prob": float(lightening_prob[region]["sum"]),
        })

    df_regions = pd.DataFrame(records)

    selected = []

    stage1 = (
        df_regions[df_regions["share_red"] >= 0.5]
        .sort_values("share_red", ascending=False)
    )

    for _, row in stage1.iterrows():
        try_select(row, selected, threshold_other_heli, travel_distances)

    stage2 = (
        df_regions[
            (df_regions["share_red_yellow"] >= 0.4) &
            (~df_regions["region_UID"].isin([s["region_UID"] for s in selected]))
        ]
        .sort_values("share_red_yellow", ascending=False)
    )

    for _, row in stage2.iterrows():
        try_select(row, selected, threshold_other_heli, travel_distances)

    return [int(s["region_UID"]) for s in selected]