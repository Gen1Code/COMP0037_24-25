#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

import numpy as np

if __name__ == '__main__':
    airport_map, drawer_height = test_three_row_scenario()
    env = LowLevelEnvironment(airport_map)
    env.set_nominal_direction_probability(0.8)

    # Policy to evaluate
    pi = env.initial_policy()
    pi.set_epsilon(0)
    pi.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    pi.set_action(14, 2, LowLevelActionType.MOVE_DOWN)  
    
    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)  
    pe.evaluate()
    v_pe.update()  
    v_pe.update()  

    first_visit = True
    episodes = 10000
    
    # On policy MC predictor
    mcpp = OnPolicyMCPredictor(env)
    mcpp.set_target_policy(pi)
    mcpp.set_experience_replay_buffer_size(64)
    
    # Q1b: Experiment with this value
    mcpp.set_use_first_visit(first_visit)
    
    v_mcpp = ValueFunctionDrawer(mcpp.value_function(), drawer_height)
    
    # Off policy MC predictor
    mcop = OffPolicyMCPredictor(env)
    mcop.set_target_policy(pi)
    mcop.set_experience_replay_buffer_size(64)
    b = env.initial_policy()
    b.set_epsilon(0.2)
    mcop.set_behaviour_policy(b)
    
    # Q1b: Experiment with this value
    mcop.set_use_first_visit(first_visit)

    v_mcop = ValueFunctionDrawer(mcop.value_function(), drawer_height)
        
    for e in range(episodes):
        mcpp.evaluate()
        v_mcpp.update()
        mcop.evaluate()
        v_mcop.update()

    to_test = [mcop, mcpp]
    labels = ["Off Policy MC", "On Policy MC"]

    for i in range(2):
        test = to_test[i]
        current_values = np.array([test.value_function().value(x,y) for x in range(15) for y in range(3)])
        current_values_gt = np.array([pe.value_function().value(x,y) for x in range(15) for y in range(3)])
        diff = current_values_gt - current_values
        valid_diff = diff[~np.isnan(diff)] 

        rms_change = np.sqrt(np.mean(valid_diff**2))

        print(f"{"First Visit" if first_visit else "Every Visit"} - Episode {episodes} - {labels[i]} - RMS change: {rms_change:.4f}")    
     
    # Sample way to generate outputs    
    v_pe.save_screenshot("q1_b_truth_pe.pdf")

    if first_visit:
        v_mcop.save_screenshot(f"q1_b_mc-off-first-visit-{episodes}.pdf")
        v_mcpp.save_screenshot(f"q1_b_mc-on-first-visit-{episodes}.pdf")
    else:
        v_mcop.save_screenshot(f"q1_b_mc-off-every-visit-{episodes}.pdf")
        v_mcpp.save_screenshot(f"q1_b_mc-on-every-visit-{episodes}.pdf")