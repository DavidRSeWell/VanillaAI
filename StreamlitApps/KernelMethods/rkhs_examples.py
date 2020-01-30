import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

# This is necessary for importing from outisde the current directory while running streamlit
import sys
sys.path.append('.')

def k_polynomial(x, xp, d):
    return (np.dot(x, xp) + 1) ** d

def k_gaussian(x, xp, sigma):
    return np.exp(-np.sum((x - xp) ** 2) / (2 * (sigma ** 2)))

def k_tanh(x, xp, kappa, Theta):
    return np.tanh(kappa * np.dot(x, xp) + Theta)

def compute_kernel(k_func,X):

    n = X.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = k_func(X[i],X[j])

    return K

def gauss_kernel(X,gamma):

    X = np.reshape(X,(len(X),1))

    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = np.exp(-1.0*(mat_sq_dists) / ( 2*gamma**2 ) )

    return K

def sample_functions(m, N, kernel):
    '''

    :param m:
    :param N:
    :param kernel:
    :return:
    '''
    # Sample Functions
    X = np.arange(-m, m + 1)
    K = compute_kernel(kernel, X)
    return X, np.random.multivariate_normal(np.zeros((2 * m + 1)), K, (N))

def main():

    st.text('Viewing the RKHS of different reproducing kernels')

if __name__ == '__main__':

    run_stream_app = 0
    if run_stream_app == 1:
        main()

    run_kernel_test = 0
    if run_kernel_test:

        print('Running kernel test')

        X = np.linspace(-10,10,100)

        gamma = 1

        num_eigvectors = 10

        K = gauss_kernel(X,gamma=gamma)

        eigvals, eigvecs = eigh(K)

        for i in range(num_eigvectors):

            plt.plot(X,eigvecs[:,-1*(i + 1)])

            #plt.show()

            #print('showing')

        plt.title('Gaussian kernel first {} eigenvectors gamma = {}'.format(num_eigvectors,gamma))
        plt.show()

        print('Done running kernel test')

    run_sample_hilbert_functions = 1
    if run_sample_hilbert_functions:

        m = 10
        N = 10
        tau = 2
        kappa = 1
        theta = 0.5
        #X,f_samples = sample_functions(m,N,lambda x, y: k_polynomial(x, y, tau))
        X,f_samples = sample_functions(m,N,lambda x, y: k_tanh(x, y, kappa,theta))
        plt.title('Tanh kernel with kappa = {} and theta = {}'.format(kappa,theta))
        #x_plot = np.linspace()
        for i in range(N):
            plt.plot(X,f_samples[i])

        plt.show()
        print('Done running sampling')

        pass