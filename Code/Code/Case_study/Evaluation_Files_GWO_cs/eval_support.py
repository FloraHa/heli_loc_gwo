"""
This file contains the support functions for the simulation.
"""

from itertools import product
from scipy.linalg import null_space
import numpy as np
from timeit import default_timer as timer
from collections import deque
from scipy.sparse import dok_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs, lsqr
import pandas as pd
from datetime import datetime

class Server:
    def __init__(self, location):
        self.location = location
        self.busy = False
        self.current_district = None

class RegionCache:
    def __init__(self, region_data):
        # Extract and normalize all needed arrays at initialization
        self.uids = region_data["region_UID"].to_numpy(dtype=int)
        self.uid_to_idx = {uid: i for i, uid in enumerate(self.uids)}

        # ---- HFI probabilities ----
        hfi_cols = ['W_0', 'W_1', 'W_4', 'W_8']
        self.hfi_probs = region_data[hfi_cols].to_numpy(dtype=float)
        self.hfi_probs /= self.hfi_probs.sum(axis=1, keepdims=True)
        self.hfi_cum = np.cumsum(self.hfi_probs, axis=1)
        self.hfi_categories = np.array([0, 1, 4, 8], dtype=int)

        # ---- Propagation index probabilities ----
        prop_cols = ['P_0', 'P_1', 'P_2', 'P_3', 'P_4', 'P_5']
        self.prop_probs = region_data[prop_cols].to_numpy(dtype=float)
        self.prop_probs /= self.prop_probs.sum(axis=1, keepdims=True)
        self.prop_cum = np.cumsum(self.prop_probs, axis=1)
        self.prop_categories = np.array(range(6), dtype=int)

        # ---- Other parameters ----
        self.heli_forecast = region_data['Forecast_cond_prob_heli_requirement'].to_numpy(dtype=float)

def get_generators_simulation(seed):
    """
    Create reproducible random number generators for the simulation.
    Returns a dictionary of RNGs.
    """
    master_rng = np.random.default_rng(seed)
    # Generate enough seeds
    rng_names = [
        "interarrival_rng", "service_time_rng", "server_choice_rng",
        "ros_rng", "initial_area_rng", "burn_index_rng",
        "propagation_index_rng", "hfi_rng", "heli_rng"
    ]
    master_seeds = master_rng.integers(0, 1_000_000, len(rng_names))
    
    # Create RNGs
    rngs = {name: np.random.default_rng(seed=s) for name, s in zip(rng_names, master_seeds)}
    return rngs
    
def get_penalty_simulator(seed):
    return np.random.default_rng(seed=seed+1)

def time_dependent_rate(current_time_minutes):

    # Time window for increased fire probability
    start_peak = (12-8)*60      # Start of the peak of fires in minutes from the start of the simulation, assumption: 8:00
    end_peak = (18-8)*60        # End of the peak of fires in minutes from the start of the simulation, assumption: 18:00

    # Scale the rate based on the adjusted time of day
    if start_peak <= current_time_minutes <= end_peak:
        # Increase the rate during peak hours
        return 2.0
    else:
        # Base rate during non-peak hours
        return 1.0

def dispatch_decision(servers, district, travel_distances, travel_distances_across_zones, positioning, travel_dist_th, server_choice_rng):
    """
    This function decides which server to dispatch
    """
    eligible_servers = []
    best_time = float('inf')

    for i, s in enumerate(servers):
        travel_dist = travel_distances[positioning[i]][district]
        if not s.busy:
            if travel_dist < travel_dist_th:
                if travel_dist < best_time:
                    best_time = travel_distances_across_zones[positioning[i]][district]
                    eligible_servers = [i]
                elif travel_dist == best_time:
                    eligible_servers.append(i)
            else:
                if travel_distances_across_zones[positioning[i]][district] < best_time:
                    best_time = travel_distances_across_zones[positioning[i]][district]


    if eligible_servers:
        return server_choice_rng.choice(eligible_servers), best_time
    else:
        return None, best_time
    
def compute_bases(initial_area):
    """
    This function computes the base dimensions of the fire ellipse.
    """
    base_a = np.sqrt(initial_area * 10000 / np.pi * 2)      # Distribution of the area is in ha (10,000 m²)
    base_b = base_a / 2
    return base_a, base_b

def compute_size_fire_at_arrival(time_passed, base_a=10, base_b=5, ROS=0.5):
    """
    This function calculates the size of the fire at arrival.
    """
    new_a = base_a + ROS * time_passed
    new_b = base_b + ROS/2 * time_passed

    return new_a, new_b

def compute_burned_area(time_passed, base_a=10, base_b=5, ROS=0.5):
    """
    This function calculates the size of the burned area as a function of the size of the fire at the arrival time.
    """
    a, b = compute_size_fire_at_arrival(time_passed, base_a, base_b, ROS)
    area_burned = np.pi * a * b
    return area_burned

def calculate_perimeter(time_passed, base_a=10, base_b=5, ROS=0.5):
    """
    This function calculates the perimeter of the fire.
    """
    a, b = compute_size_fire_at_arrival(time_passed, base_a, base_b, ROS)
    perimeter = np.pi * (a + b) * \
        (1 + 3 * ((a - b)**2/(a + b)**2) / \
            (10 + np.sqrt(4-3*((a - b)**2/(a + b)**2))))
    return perimeter

def compute_mu(travel_time, base_speed_1=1, base_speed_2=1, setup_time=2, base_a=10, base_b=5, ROS=0.5):
    """
    This function calculates the mu as a function of the travel time.
    """
    # Calculate the service time
    total_time_trajectory = 2*travel_time + setup_time                                                      # In minutes
    attack_time_1 = ((calculate_perimeter(travel_time, base_a, base_b, ROS))/base_speed_1)        # Attack speed in minutes
    attack_time_2 = ((calculate_perimeter(travel_time, base_a, base_b, ROS))/base_speed_2)
    service_time = (total_time_trajectory + attack_time_1 + attack_time_2)
    # Convert service time to service rate
    return 1 / service_time

def update_ROS(ROS, propagation_index, hfi_index, hfi_factor_ROS, method_update_ROS):
    """
    This function updates the Rate of Spread (ROS) based on the burn index.
    """
    if method_update_ROS == "propagation_index":
        if propagation_index == 'P_0':
            ROS *= 0.1
        elif propagation_index == 'P_1':
            ROS *= 1.0
        elif propagation_index == 'P_2':
            ROS *= 2.0
        elif propagation_index == 'P_3':
            ROS *= 3.0
        elif propagation_index == 'P_4':
            ROS *= 4.0
        elif propagation_index == 'P_5':
            ROS *= 5.0
    elif method_update_ROS == "hfi_index":
        if hfi_index == 0:
            ROS *= hfi_factor_ROS["Blue"]
        elif hfi_index == 1:
            ROS *= hfi_factor_ROS["Green"]
        elif hfi_index == 4:
            ROS *= hfi_factor_ROS["Yellow"]
        elif hfi_index == 8:
            ROS *= hfi_factor_ROS["Red"]
    else: 
        ROS = ROS
    return ROS


def sample_hfi(region_cache, rng, uid):
    idx = region_cache.uid_to_idx[uid]
    u = rng.random()
    cat_idx = np.searchsorted(region_cache.hfi_cum[idx], u)
    return region_cache.hfi_categories[cat_idx]


def sample_propagation(region_cache, rng, uid):
    idx = region_cache.uid_to_idx[uid]
    u = rng.random()
    cat_idx = np.searchsorted(region_cache.prop_cum[idx], u)
    return region_cache.prop_categories[cat_idx]

def get_heli_prob(region_cache, uid):
    return region_cache.heli_forecast[region_cache.uid_to_idx[uid]]

def get_requirement_helicopter_hfi(heli_rng, hfi, probabilities={"Blue": 0, "Green": 0.10, "Yellow": 0.5, "Red": 1}):
    """
    This function generates a random requirement for helicopters based on the HFI.
    """
    if hfi == 0:
        return heli_rng.random() < probabilities["Blue"]
    elif hfi == 1:
        return heli_rng.random() < probabilities["Green"]
    elif hfi == 4:
        return heli_rng.random() < probabilities["Yellow"]
    elif hfi == 8:
        return heli_rng.random() < probabilities["Red"]
    
def get_requirement_helicopter_fop(region_data, index, heli_rng, training=False, factor_training=1.0):
    """
    This function generates a random requirement for helicopters based on the FOP.
    """

    p = get_heli_prob(region_data, index)

    if training:
        p *= factor_training

    return heli_rng.random() < p

def update_mu(mu, hfi):
    """
    This function updates the mu based on the HFI data.
    """
    mu = mu * (1 + hfi)
    #mu = mu * 1
    return mu

def convert_to_minutes_since_8(time_str):
    try:
        fire_time = pd.to_datetime(time_str, format="%H:%M").time()
        reference_time = datetime.strptime("08:00", "%H:%M").time()
        fire_dt = datetime.combine(datetime.today(), fire_time)
        ref_dt = datetime.combine(datetime.today(), reference_time)
        delta = fire_dt - ref_dt
        return max(0, int(delta.total_seconds() // 60))
    except Exception:
        return None

