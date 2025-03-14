'''
Created on 14 Jan 2022

@author: ucacsjj
'''

import numpy as np
import random

from .agent import Agent

class EpsilonGreedyAgent(Agent):
    
    def __init__(self, environment, epsilon):
        super().__init__(environment)
        self._epsilon = epsilon

    # Q5a:
    # Change the implementation to use the epsilon greedy algorithm
    def _choose_action(self):
        if random.random() <= self._epsilon:
            action = random.randint(0,self._environment.number_of_bandits()-1)
        else:
            average_q = np.divide(self.total_reward, self.number_of_pulls)
            action = np.where(average_q == np.amax(average_q))[0][0]
            
        return action
            
        
