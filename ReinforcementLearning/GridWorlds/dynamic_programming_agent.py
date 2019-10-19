'''
Implementation of algorithm as described in chpt 4.3 of Richard Suttons
Reinforcement learning book
'''
import numpy as np



class DPPolicyAgent(object):

    def __init__(self,grid):
        '''

        :param grid: Grid object from gym environment
        '''

        self.V = np.zeros((grid.height,grid.width)) # Value function
        self.P = np.zeros((grid.height,grid.width)) # Policy function
        pass

    def get_action(self):
        pass

    def run_policy_evaluation(self):
        '''
        Evaluate the value of all states given the current policy
        while not converged:
            delta = 0
            for each s in S:
                v = V(s)
                action = policy(s)
                V(s) = Expected value given current state and action
                delta = max(delta, |v - V(s)|)
            end if delta < epsilon
        :return:
        '''

    def run_policy_improvement(self):
        '''
        Given the current value and policy. Improve policy
        for each s in S:
            a = policy(s)
            policy(s) = argmax_a E(s,a)
        :return:
        '''

    def train(self):
        '''
        Steps:
            1: initialize V(s) and policy(s) for all s

            while not converged:
                a: run policy improvement
                b: evaluate new policy
        :return:
        '''
        pass