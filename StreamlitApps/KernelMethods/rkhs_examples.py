import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

# This is necessary for importing from outisde the current directory while running streamlit
import sys
sys.path.append('.')


def gauss_kernel(X,gamma):

    X = np.reshape(X,(len(X),1))

    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = np.exp(-1.0*(mat_sq_dists) / ( 2*gamma**2 ) )

    return K

def main():

    st.text('Viewing the RKHS of different reproducing kernels')

if __name__ == '__main__':

    run_stream_app = 0
    if run_stream_app == 1:
        main()


    run_kernel_test = 1
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