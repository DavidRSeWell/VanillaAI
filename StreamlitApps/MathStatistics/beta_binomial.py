import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import streamlit as st

from scipy.stats import beta as betafunc
from scipy.stats import betabinom
from scipy.special import comb

# This is necessary for importing from outisde the current directory while running streamlit
import sys
sys.path.append('.')

def sample_beta_binomial(n,a,b,size=None):

    p = np.random.beta(a, b, size=size)
    r = np.random.binomial(n, p)
    return r,p


def main():
    st.title('Beta Binomial fun in the sun')


    ########################
    # SIDEBAR
    ########################

    st.sidebar.title('Choose model parameters')
    num_trials = int(st.sidebar.text_input('Trials',100))
    alpha = float(st.sidebar.text_input('alpha',1))
    beta = float(st.sidebar.text_input('beta',1))

    st.title('Prior')

    #x = np.linspace(0,)

    #rv = betafunc(alpha,beta)

    beta_samples = np.random.beta(alpha,beta,num_trials)

    plt.hist(beta_samples)
    st.pyplot()
    plt.close()

    #beta_test = betabinom.pmf(num_trials,10,alpha,beta)

#    post, prior = sample_beta_binomial(num_trials,alpha,beta,size=num_trials)

 #   x = [i for i in range(num_trials)]

  #  rv = betabinom(num_trials, alpha, beta)

    beta_post_sample = np.random.beta(alpha + 25, beta + 25,num_trials)
    plt.hist(beta_post_sample)
    #plt.plot(x,rv.pmf(x))

    st.pyplot()





if __name__ == '__main__':
    main()