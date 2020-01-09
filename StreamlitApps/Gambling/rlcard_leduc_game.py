import streamlit as st
import rlcard

from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed

# This is necessary for importing from outisde the current directory while running streamlit
import sys
sys.path.append('.')

def main():

    # Make environment
    env = rlcard.make('blackjack')
    episode_num = 2

    # Set a global seed
    set_global_seed(0)

    # Set up agents
    agent_0 = RandomAgent(action_num=env.action_num)
    env.set_agents([agent_0])

    for episode in range(episode_num):

        # Generate data from the environment
        trajectories, _ = env.run(is_training=False)

        # Print out the trajectories
        print('\nEpisode {}'.format(episode))
        for ts in trajectories[0]:
            print(
                'State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.format(ts[0], ts[1], ts[2], ts[3], ts[4]))


if __name__ == '__main__':
    main()