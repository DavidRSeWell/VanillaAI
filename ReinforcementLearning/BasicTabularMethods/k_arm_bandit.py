##############################################################
# This module contains all functions for solving
# K arm bandit problems as described in Richard Suttons Book
##############################################################

import numpy as np

from scipy.stats import norm

def select_action(q_values,count_values,epsilon,num_levers,t,algorithm_type,c=1):

    action = None  # will be the index of the lever selected

    if algorithm_type == 'greedy':

        rand_uniform = np.random.uniform(0, 1)

        if rand_uniform <= epsilon:
            # take random action
            action = np.random.randint(0, num_levers)
        else:
            action = np.argmax(q_values)

    elif algorithm_type == 'UCB':

        ucb_values = []
        for i in range(len(q_values)):

            ucb_value = q_values[i] + c*np.sqrt( np.log(t) / float(count_values[i] ) )

            ucb_values.append(ucb_value)

        action = np.argmax(ucb_values)

    elif algorithm_type == 'greedy-2':
        pass

    else:
        print("Error incorrect algorithm type")

    return action

def run_e_greedy_bandit(lever_data,lever_meta,epochs=100,epsilon=0.1,init_q_value=0,algo_type='greedy'):
    '''
    This is a function for running the epsilon greedy bandit with different
    parameters.

    :param lever_data: Dictionary of the parameters used for each lever i.e (mean , std )
    :param epochs: number of iterations
    :param epsilon: for e-greedy function
    :param init_q_value: Initial guess for the value of each lever
    :param algo_type: UCB or greedy
    :return:
    '''

    num_levers = len(lever_data['data'])

    count_values = np.zeros(num_levers)

    q_values = np.ones(num_levers) * init_q_value

    avg_reward_data = [0]

    avg_reward = 0

    for i in range(1, epochs + 1):
        print("Epoch: " + str(i))

        print("Avg Reward: " + str(avg_reward))

        print("Reward Vector: " + str(q_values))

        action = select_action(q_values, count_values, epsilon, num_levers, i, algo_type, 1)

        mean_current_lever = lever_meta[action]['mean']

        std_current_lever = lever_meta[action]['std']

        reward = norm.rvs(loc=mean_current_lever, scale=std_current_lever)

        # update lever average and count
        count_values[action] += 1

        q_values[action] = q_values[action] + (1 / count_values[action]) * (reward - q_values[action])

        # Update overall average reward
        avg_reward = avg_reward + (1.0 / i) * (reward - avg_reward)

        avg_reward_data.append(avg_reward)

    return avg_reward_data, avg_reward, list(q_values), list(count_values)


