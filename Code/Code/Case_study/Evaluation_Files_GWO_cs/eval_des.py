"""
This file contains the simulation function for the case study.
"""

import heapq
import random
import numpy as np
from timeit import default_timer as timer
from collections import deque, defaultdict
from scipy.stats import lognorm
import geopandas as gpd
import pandas as pd

from Code.Case_study.Evaluation_Files_GWO_cs.eval_support import Server, get_generators_simulation, get_penalty_simulator, time_dependent_rate, dispatch_decision, \
        compute_bases, compute_size_fire_at_arrival, compute_burned_area, calculate_perimeter, compute_mu, \
        update_ROS, get_requirement_helicopter_hfi, get_requirement_helicopter_fop, update_mu, convert_to_minutes_since_8, sample_hfi, sample_propagation

def simulate(positioning, params_case_study, params_fire, params_attack, params_sim, random_seed,
             training=False, bool_print=False, historic_validation=False, historic_fires=None):
    
    start = timer()
    SIM_TIME = params_sim.SIMTIME


    # Initialize the dictionnary that stores the dispatches
    dispatch_dict = defaultdict(int)
    # Initialize the dictionnary that stores which fire was responded by which unit
    response_dict = defaultdict(int)
    # Initialize the dict for the calls per district
    total_calls_dict = defaultdict(int)

    # Initialize the total area burned
    total_area_burned_attacked = 0
    total_area_burned_escaped = 0

    # Initialize the number of fires
    num_fires = 0
    num_fires_within_range = 0
    distance_fires = 0
    response_time_fires = 0.0

    # Initialize the servers, the queue and the current time
    servers = [Server(positioning[i]) for i in range(len(positioning))]
    event_queue = []
    queue = deque()
    current_time = 0.0

    # Create the random number generators
    rngs = get_generators_simulation(random_seed)
    if not isinstance(params_fire.penalty, int):
        rng_penalty = get_penalty_simulator(random_seed)

    dist_ros = lognorm(s=params_fire.ROS_distribution["Parameters"]["s"], loc=params_fire.ROS_distribution["Parameters"]["loc"], scale=params_fire.ROS_distribution["Parameters"]["scale"])
    dist_initial_area = lognorm(s=params_fire.initial_area_distribution["Parameters"]["s"], loc=params_fire.initial_area_distribution["Parameters"]["loc"], scale=params_fire.initial_area_distribution["Parameters"]["scale"])

    # Initialize the stopping criterion
    if training:
        min_num_fires_sim = params_sim.min_num_fires_sim
    else:
        min_num_fires_sim = 0

    # Generate first arrivals
    if not historic_validation:
        for j, lam in params_case_study.lambda_j.items():
            scaled_rate = lam * time_dependent_rate(current_time)
            interarrival = rngs["interarrival_rng"].exponential(1/scaled_rate)
            heapq.heappush(event_queue, (current_time + interarrival, 'arrival', j))
    else:
        for arrival_time, district_id in historic_fires:
            heapq.heappush(event_queue, (arrival_time, 'arrival', int(district_id)))



    # State tracking
    state_time = defaultdict(float)
    state_time_bin = defaultdict(float)
    last_event_time = 0.0

    # Simulation loop
    while event_queue:
        
        time, event_type, data = heapq.heappop(event_queue)
        if time >= SIM_TIME and num_fires >= min_num_fires_sim:
            break  # Stop processing events beyond simulation time
        current_time = time

        # Track state
        server_status = tuple(1 if s.busy else 0 for s in servers)
        server_status_district = tuple(s.current_district+1 if s.busy else 0 for s in servers)
        queue_len = len(queue)
        extended_state = (server_status, queue_len)
        extended_state_district = (server_status_district, queue_len)
        state_time_bin[extended_state] += time - last_event_time
        state_time[extended_state_district] += time - last_event_time
        last_event_time = time

        if event_type == 'arrival': 
            district = data
            hfi = sample_hfi(params_case_study.regional_data, rngs["hfi_rng"], district)
            if params_fire.fop:
                heli_required = get_requirement_helicopter_fop(params_case_study.regional_data, district, rngs["heli_rng"], training=False, factor_training=1.0)
            else:
                heli_required = get_requirement_helicopter_hfi(rngs["heli_rng"], hfi, params_fire.probabilities_heli_deploy_HFI)

            if heli_required:
                num_fires += 1
                total_calls_dict[district] += 1
                server_idx, best_time = dispatch_decision(servers, district, params_case_study.travel_distances, params_case_study.travel_distances_across_zones, positioning, params_attack.dispatch_range, rngs["server_choice_rng"])
                
                if server_idx is not None:
                  distance_fires += best_time
                
                # Calculate the ROS
                ROS = dist_ros.rvs(random_state=rngs["ros_rng"])

                propagation_index = sample_propagation(params_case_study.regional_data, rngs["propagation_index_rng"], district)
                ROS = update_ROS(ROS, propagation_index, hfi, params_fire.HFI_factor_ROS, params_fire.method_update_ROS)
                
                ROS = ROS*params_fire.factor_ROS

                # Incorporate the HFI
                initial_area = dist_initial_area.rvs(random_state=rngs["initial_area_rng"])

                base_a, base_b = compute_bases(initial_area)

                if server_idx is not None:
                    num_fires_within_range += 1
                    servers[server_idx].busy = True
                    servers[server_idx].current_district = district

                    # Compute travel time and μ dynamically
                    travel_time = (params_case_study.travel_distances[positioning[server_idx]][district]/params_attack.travel_speed)*60         # Travel time in minutes
                    response_time_fires += travel_time
                    mu = compute_mu(travel_time, params_attack.base_speed_1, params_attack.base_speed_2, params_attack.setup_time, base_a, base_b, ROS)

                    mu = mu * params_fire.factor_my

                    # service_time = rngs["service_time_rng"].exponential(1/mu)
                    service_time = 1/mu
                    area_burned_attacked = compute_burned_area(travel_time, base_a, base_b, ROS)

                    total_area_burned_attacked += area_burned_attacked

                    heapq.heappush(event_queue, (time + service_time, 'departure', server_idx))
                    dispatch_dict[(positioning[server_idx], district)] += 1
                    response_dict[(district, positioning[server_idx])] += 1
                else: 

                    dispatch_dict[("Lost", -100)] += 1
                    response_dict[(district, "Lost")] += 1
                    if isinstance(params_fire.penalty, int):
                        area_burned_escaped = compute_burned_area(params_fire.penalty, base_a, base_b, ROS)
                        response_time_fires += params_fire.penalty
                        
                    else:
                        # Sample the penalty
                        penalty = rng_penalty.uniform(params_fire.penalty[0], params_fire.penalty[1])
                        area_burned_escaped = compute_burned_area(penalty, base_a, base_b, ROS)
                        response_time_fires += penalty

                    total_area_burned_escaped += area_burned_escaped


            # Schedule next arrival
            if not historic_validation:
                scaled_rate = params_case_study.lambda_j[district] * time_dependent_rate(current_time)
                interarrival = rngs["interarrival_rng"].exponential(1/scaled_rate)
                heapq.heappush(event_queue, (time + interarrival, 'arrival', district))

        elif event_type == 'departure':
            server_idx = data
            servers[server_idx].busy = False
            servers[server_idx].current_district = None

    # Final update
    server_status = tuple(1 if s.busy else 0 for s in servers)
    server_status_district = tuple(s.current_district+1 if s.busy else 0 for s in servers)
    queue_len = len(queue)
    extended_state = (server_status, queue_len)
    extended_state_district = (server_status_district, queue_len)
    state_time_bin[extended_state] += SIM_TIME - last_event_time
    state_time[extended_state_district] += SIM_TIME - last_event_time

    end = timer()

    total_area_burned = total_area_burned_attacked + total_area_burned_escaped

    results = {"area_burned_attacked_rep": total_area_burned_attacked/SIM_TIME,
               "area_burned_escaped_rep": total_area_burned_escaped/SIM_TIME,
               "area_burned_rep": total_area_burned/SIM_TIME,
               "distance_fires_rep": distance_fires,
               "num_fires_rep": num_fires,
               "num_fires_within_range_rep": num_fires_within_range,
               "sol_time_rep": end-start,
               "fire_counts": total_calls_dict,
               "dispatch_dict": dispatch_dict,
               "response_dict": response_dict,
               "response_times_all_fires": response_time_fires}

    return results


