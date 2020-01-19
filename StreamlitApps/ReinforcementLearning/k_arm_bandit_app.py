import numpy as np
import os
import pandas as pd
import pickle
import plotly.graph_objs as go
import plotly.express as px
from scipy.stats import norm
import streamlit as st

# This is necessary for importing from outisde the current directory while running streamlit
import sys
sys.path.append('.')

from ReinforcementLearning.BasicTabularMethods.k_arm_bandit import run_e_greedy_bandit


def main():

    st.text('Multi arm bandit problem')

    ########################
    # SIDEBAR
    ########################

    st.sidebar.title('Choose model parameters')
    num_selection = st.sidebar.selectbox('Bandit Number', [1,2,3,4,5], index=1)
    algo_type = st.sidebar.selectbox('Algorithm', ['greedy','UCB'], index=1)
    num_epochs = st.sidebar.text_input('Iterations', 100)
    epsilon_value = st.sidebar.slider('Epsilon value',0.0,1.0,value=0.0,step=0.1)
    initial_guess = st.sidebar.text_input('Initial guess',15)
    clear_button = st.sidebar.button('Clear graph')

    DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/Data/k_arm_bandit/reward_graphs/'


    if clear_button:
        for file in os.listdir(DATA_PATH):
            os.remove(DATA_PATH + file)
    else:
        pass
    # get random mean
    violin_figure = go.Figure()
    violin_meta = []

    for i in range(num_selection):

        rand_mean = np.random.uniform(0, 5)
        rand_std = np.random.uniform(1,1)

        y_norm = norm.rvs(loc=rand_mean, size=1000)

        name = 'lever' + str(i) + ' mean: ' + str(np.round(rand_mean,3))

        tracei = {
            "type": "violin",
            "y": y_norm,
            "name":name
        }

        violin_figure.add_trace(tracei)
        violin_meta.append({'mean':rand_mean,'std':rand_std})

    avg_reward_data, avg_reward, errors,q_values, count_values = run_e_greedy_bandit(violin_figure,violin_meta,epochs=int(num_epochs),epsilon=epsilon_value,init_q_value=int(initial_guess))

    st.write(violin_figure)

    award_figure = go.Figure()

    for file in os.listdir(DATA_PATH):
        pass

    award_plot = go.Scatter(y=avg_reward_data,mode='lines')
    award_figure.add_trace(award_plot)
    #st.line_chart(avg_reward_data)

    st.write(award_figure)

    st.write('Estimated means by the algorithm')
    st.write(pd.DataFrame(q_values,columns=['lever #']))

    st.write('Distribution of the number of times the lever was pulled')
    st.write(pd.DataFrame(count_values,columns=['Count distribution']))

    # save off new file
    current_test_file = {'num_levers':int(num_selection),
                         'epsilon':epsilon_value,
                         'algo_type':algo_type,
                         'iterations':num_epochs,
                         'init_guess':initial_guess,
                         'final_avg':avg_reward_data[-1],
                         'erros':errors

                         }

    pickle.dump(current_test_file,DATA_PATH + 'blah')


    print('done running ')


if __name__ == '__main__':

    main()

