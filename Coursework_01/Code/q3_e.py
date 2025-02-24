#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

import time
import numpy as np
import matplotlib.pyplot as plt
from common.scenarios import full_scenario
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':  

    theta_values = [0.1, 0.01, 0.001]
    max_steps_values = [1, 2, 5, 10, 20]

    results = []

    #Baseline policy solver
    airport_map, drawer_height = full_scenario()
    airport_environment = LowLevelEnvironment(airport_map)
    airport_environment.set_nominal_direction_probability(0.8)
    baseline_policy_solver = PolicyIterator(airport_environment)
    baseline_policy_solver.set_theta(0.0001)
    baseline_policy_solver.set_max_policy_evaluation_steps_per_iteration(50)
    baseline_policy_solver.initialize()
    v, _ = baseline_policy_solver.solve_policy()
    baseline_v = v._values

    for theta in theta_values:
        for max_steps in max_steps_values:
            # Get the map for the scenario
            airport_map, drawer_height = full_scenario()
    
            # Set up the environment for the robot driving around
            airport_environment = LowLevelEnvironment(airport_map)
            
            # Configure the process model
            airport_environment.set_nominal_direction_probability(0.8)
            
            policy_solver = PolicyIterator(airport_environment)

            policy_solver.set_theta(theta)
            policy_solver.set_max_policy_evaluation_steps_per_iteration(max_steps)

            # Set up initial state
            policy_solver.initialize()
                  
            start_time = time.time()
            v, pi = policy_solver.solve_policy()
            end_time = time.time()
            elapsed_time = end_time - start_time

            #mask out nan values
            mask = ~np.isnan(baseline_v) & ~np.isnan(v._values)
            l2_norm = np.linalg.norm(baseline_v[mask] - v._values[mask])

            results.append([theta, max_steps, elapsed_time, l2_norm])
    
    print(results)
    
    plt.figure(figsize=(10, 6))
    for theta in theta_values:
        times = [result[2] for result in results if result[0] == theta]
        plt.plot(max_steps_values, times, label=f'Theta={theta} (Time)')

    plt.xlabel('Max Policy Evaluation Steps per Iteration')
    plt.ylabel('Time to Solve Policy (seconds)')
    plt.title('Policy Iteration Performance')
    plt.legend()
    plt.grid(True)
    plt.show()