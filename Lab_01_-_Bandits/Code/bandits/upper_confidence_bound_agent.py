'''
Created on 14 Jan 2022

@author: ucacsjj
'''

import math

import numpy as np

from .agent import Agent

class UpperConfidenceBoundAgent(Agent):

    def __init__(self, environment, c):
        super().__init__(environment)
        self._c = c

    # Q6a:
    # Implement UCB
    def _choose_action(self):
                
        average_q = np.divide(self.total_reward, self.number_of_pulls)
        
        best_action = 0
        best_score = -99
        
        for a in range(self._number_of_bandits):
            score = average_q[a] + self._c * math.sqrt(math.log(self.total_number_of_pulls)/self.number_of_pulls[a])
            if best_score < score:
                best_score = score
                best_action = a

        return best_action
