import streamlit as st

import sys
sys.path.append('.')

from ReinforcementLearning.BasicTabularMethods.k_arm_bandit import run_e_greedy_bandit


def main():

    st.text('Multi arm bandit problem')
    st.sidebar.title('Choose model parameters')

    pass


if __name__ == '__main__':

    main()

