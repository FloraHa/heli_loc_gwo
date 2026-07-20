"""
This file stores the classes for the parameters.
"""

from dataclasses import dataclass
from Code.Case_study.Evaluation_Files_GWO_cs.eval_support import RegionCache

@dataclass
class ParamsGWO:
    replication_gwo: int
    population_size: int
    size_elite: int
    max_iter: int
    initialize_gwo_mclp: bool
    initialize_gwo_hist_heur: bool
    optimize_mclp: bool
    initial_wolf_mclp: None
    initial_wolf_hist_heur: None
    threshold: float
    
@dataclass
class ParamsFireBehavior:
    initial_area_distribution: dict
    ROS_distribution: dict
    HFI_factor_ROS: dict
    probabilities_heli_deploy_HFI: dict
    adjust_fact_lambda_HFI: dict
    fop: bool
    penalty: float
    method_update_ROS: str
    factor_arrival: float
    factor_my: float
    factor_ROS: float

@dataclass
class ParamsFireAttack:
    travel_speed: float
    base_speed_1: float
    base_speed_2: float
    setup_time: float
    dispatch_range: float

@dataclass
class ParamsSimulation:
    replications: int
    random_seed: int
    SIMTIME: int
    min_num_fires_sim: int

@dataclass
class ParamsCaseStudy:
    hfi_data_base: str
    hfi_data_mclp: dict
    max_helicopter_per_district: float
    districts: list
    lambda_j: dict
    pot_locations: list
    travel_distances: dict
    travel_distances_across_zones: dict
    regional_data: RegionCache
    num_helicopters: int
    zone_inc_only: bool
    threshold_other_heli: float
    secondary_bases: bool



