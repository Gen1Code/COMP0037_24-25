#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

import time
from common.scenarios import *
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_iterator import ValueIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    max_steps = [100, 20, 5]
    
    results = []
    
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Configure the process model
    airport_environment.set_nominal_direction_probability(0.8)
    
    for ms in max_steps:
        # Create the policy iterator
        policy_solver = PolicyIterator(airport_environment)
        policy_solver.set_max_policy_evaluation_steps_per_iteration(ms)
        
        # Set up initial state
        policy_solver.initialize()
            
        # Bind the drawer with the solver
        policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
        policy_solver.set_policy_drawer(policy_drawer)
        
        value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
        policy_solver.set_value_function_drawer(value_function_drawer)
            
        # Compute the solution
        start_time = time.time()
        v, pi = policy_solver.solve_policy()
        end_time = time.time()
        
        results.append(end_time - start_time)
        
        value_function_drawer.wait_for_key_press()
    
    #Value iteration
    policy_solver = ValueIterator(airport_environment)
    policy_solver.initialize()
        
    policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)
    
    value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_function_drawer)
        
    start_time = time.time()    
    v, pi = policy_solver.solve_policy()
    end_time = time.time()
    
    results.append(end_time-start_time)
    
    policy_drawer.save_screenshot("value_iterator_results.pdf")
    
    print(results)
    
    # Wait for a key press
    value_function_drawer.wait_for_key_press()
    