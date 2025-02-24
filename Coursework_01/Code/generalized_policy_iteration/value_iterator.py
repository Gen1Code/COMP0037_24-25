'''
Created on 29 Jan 2022

@author: ucacsjj
'''

from .dynamic_programming_base import DynamicProgrammingBase

# This class ipmlements the value iteration algorithm

class ValueIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the value iteration
        # algorithm is carried out is carried out.
        self._max_optimal_value_function_iterations = 2000
   
    # Method to change the maximum number of iterations
    def set_max_optimal_value_function_iterations(self, max_optimal_value_function_iterations):
        self._max_optimal_value_function_iterations = max_optimal_value_function_iterations

    #    
    def solve_policy(self):

        # Initialize the drawers
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        self._compute_optimal_value_function()
 
        self._extract_policy()
        
        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        return self._v, self._pi

    # Q3f:
    # Finish the implementation of the methods below.
    def _compute_optimal_value_function(self):
        """
        Implements Value Iteration Algorithm.
        Reference steps:
        1) v <- V(s)
        2) V(s) <- max_a Σ p(s', r | s, a) [r + γ V(s')]
        3) delta <- max(delta, |v - V(s)|)
        4) Terminate if delta < θ, otherwise continue iterations
        """

        while True:
            # Track the maximum change in the value function during this iteration
            delta = 0.0

            # Iterate through all states s ∈ S
            for x in range(self._environment.map().width()):
                for y in range(self._environment.map().height()):
                    # Skip if this cell is an obstruction
                    if self._environment.map().cell(x, y).is_obstruction():
                        continue

                    # 1) v <- V(s): Save the old value of the current state
                    v_old = self._v.value(x, y)

                    # 2) Compute the new V(s): calculate Bellman optimal equation for all actions
                    best_value = float('-inf')
                    for action in range(self._environment.available_actions().n):
                        # Get (s', r, p) distribution from environment for taking this action
                        next_states, rewards, probabilities = self._environment.next_state_and_reward_distribution(
                            (x, y), action
                        )

                        # Calculate Σ p(s', r | s, a) * [r + γ V(s')]
                        q_value = 0.0
                        for ns, reward, prob in zip(next_states, rewards, probabilities):
                            if ns is not None:
                                nx, ny = ns.coords()
                                q_value += prob * (reward + self._gamma * self._v.value(nx, ny))
                        
                        # Keep the maximum value across all possible actions
                        if q_value > best_value:
                            best_value = q_value

                    # Update the value function for current state
                    self._v.set_value(x, y, best_value)

                    # 3) Update delta: track the maximum change in this iteration
                    delta = max(delta, abs(v_old - best_value))

            # 4) Check convergence condition: terminate if changes are below threshold
            if delta < self._theta:
                break

    def _extract_policy(self):
        """
        Extracts the optimal policy from the computed value function.
        """
        for x in range(self._environment.map().width()):
            for y in range(self._environment.map().height()):
                state = (x, y)

                if self._environment.map().cell(x, y).is_obstruction():
                    continue  # Skip obstruction cells

                best_action = None
                best_value = float('-inf')

                # π(s) = argmax_a Σ p(s', r | s, a) [r + γV(s')]
                for action in range(self._environment.available_actions().n):
                    next_states, rewards, probabilities = self._environment.next_state_and_reward_distribution(state, action)

                    # Calculate expected value for current action
                    value_sum = sum(
                        prob * (reward + self._gamma * self._v.value(ns.coords()[0], ns.coords()[1]))
                        for ns, reward, prob in zip(next_states, rewards, probabilities) if ns is not None
                    )

                    # Update best action if current action yields higher value
                    if value_sum > best_value:
                        best_value = value_sum
                        best_action = action

                # Set the optimal action for current state
                self._pi.set_action(x, y, best_action)
