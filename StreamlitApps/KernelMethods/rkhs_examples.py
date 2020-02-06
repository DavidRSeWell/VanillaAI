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

def compute_kernel(k_func,X,):

    n = X.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = k_func(X[i],X[j])

    return K

def center_kernel(K):
    '''
    Using the formulation from the book (same as class)

    K_c = K - I_m * K - K*I_m + I_m * K *I_m
    I_m is identity matrix
    :param K:
    :return:
    '''

    I = np.identity(K.shape[0])
    n = K.shape[0]
    IKI = K.sum() / float(n*n)

    mm = np.ones((n,n)) * (1.0 / n)

    # start with slow version for sanity check
    '''
    K_c_1 = np.zeros((K.shape))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            #K_c_1[i][j] = K - np.dot(I,K[:,i]) - np.dot(K[j,:],I) + IKI
            K_c_1[i][j] = K[i][j] - K[:,i].sum()/ n - K[j,:].sum() / n + IKI
    '''

    #K_c_1 = np.dot(np.dot((I - mm),K),(I - mm))
    K_c_1 = K - np.dot(mm,K) - np.dot(K,mm) + np.dot(np.dot(mm,K),mm)

    K[:11,:11] = K_c_1[:11,:11]



    #K_c = K - np.dot(I,K) - np.dot(K,I) + np.dot(np.dot(I,K),I)

    #diff = (K_c_1 - K_c).sum()

    return K

def gauss_kernel(X,gamma):

    X = np.reshape(X,(len(X),1))

    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = np.exp(-1.0*(mat_sq_dists) / ( 2*gamma**2 ) )

    return K

def sample_functions(m, N, K):
    '''

    :param m:
    :param N:
    :param kernel:
    :return:
    '''
    # Sample Functions
    #X = np.arange(-m, m + 1)
    #K = compute_kernel(kernel, X)
    return np.random.multivariate_normal(np.zeros((2*m + 1)), K, (N))

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

        K_c = center_kernel(K)

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
        tau = 10
        kappa = 1
        theta = 0.5

        #X,f_samples = sample_functions(m,N,lambda x, y: k_polynomial(x, y, tau))
        #X,f_samples = sample_functions(m,N,lambda x, y: k_polynomial(x, y, tau))
        #X,f_samples = sample_functions(m,N,lambda x, y: k_tanh(x, y, kappa,theta))
        X = np.arange(-m,m + 1)

        K = gauss_kernel(X,tau)

        K_c = center_kernel(K)

        f_samples = sample_functions(m,N,K_c)

        print('f sample mean = {}'.format(f_samples.mean()))
        mean_tot = 0
        for i in range(N):
            mean_tot += f_samples[i][:11].mean()

        mean_tot = mean_tot / N
        print('calc mean = {}'.format(mean_tot))
        plt.title('Gaussian kernel centered at 0 with tau = {}'.format(tau))
        #x_plot = np.linspace()
        #X = [i for i in range(-10,11,2)]
        for i in range(N):
            plt.plot(X,f_samples[i])

        plt.show()
        print('Done running sampling')

        pass