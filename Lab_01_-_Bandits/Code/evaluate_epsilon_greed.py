#!/usr/bin/env python3

'''
Created on 14 Jan 2022

@author: ucacsjj
'''

import matplotlib.pyplot as plt
import numpy as np

from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment
from bandits.epsilon_greedy_agent import EpsilonGreedyAgent
from bandits.damped_epsilon_greedy_agent import DampedEpsilonGreedyAgent
from bandits.performance_measures import compute_percentage_of_optimal_actions_selected
from bandits.performance_measures import compute_regret

if __name__ == '__main__':
       # Create bandit
    environment = BanditEnvironment(10)
    
    # Add some bandits. These use the actual values which generated
    # the plots in the examples in the lectures.
    environment.set_bandit(0, Bandit(0.1, 1))    
    environment.set_bandit(1, Bandit(-0.5, 1))    
    environment.set_bandit(2, Bandit(1.5, 1))
    environment.set_bandit(3, Bandit(0.5, 1))
    environment.set_bandit(4, Bandit(1.2, 1))
    environment.set_bandit(5, Bandit(-1.5, 1))
    environment.set_bandit(6, Bandit(-0.2, 1))
    environment.set_bandit(7, Bandit(-1, 1))
    environment.set_bandit(8, Bandit(0.5, 1))
    environment.set_bandit(9, Bandit(-0.5, 1))
    
    number_of_steps = 10000
    
    # Q5b:
    # Change values to see what happens
    epsilon = 0.05
    
    agent = DampedEpsilonGreedyAgent(environment, epsilon)
    
    # Step-by-step store of rewards
    reward_history = np.zeros(number_of_steps)
    action_history = np.zeros(number_of_steps)
    
    # Step through the agent and let it do its business
    for p in range(0, number_of_steps):
        action_history[p], reward_history[p] = agent.step()
        
    print(f'Mean reward={np.mean(reward_history)}')

    # Plot actions
    plt.figure(1)
    plt.plot(action_history)
    plt.xlabel('Sample number')
    plt.ylabel('Arm pulled')
    
    # Plot percentage correct action curves
    plt.figure(2)
    percentage_correct_actions = compute_percentage_of_optimal_actions_selected(environment, action_history)
    plt.plot(percentage_correct_actions)
    plt.xlabel('Sample number')
    plt.ylabel('Percentage optimal action')

    # Plot the regret curves
    plt.figure(3)
    regret = compute_regret(environment, reward_history)
    plt.plot(regret)
    plt.xlabel('Sample number')
    plt.ylabel('Regret')

    # This way means you have to close each figure separately, but you
    # can interactively explore the content.
    plt.show()
    
