import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal as mvn

#LOCAL
from KernelMethods.Unsupervised.k_means import KMeans

class GMM:
    '''
    GMM algorithm overview:
        Init:
            - Means,Covariances and mixing coefficients (prior)

        2: E step - evaluate 'responsibilities'
        3: M step - Re estimate the parameters using the current responsibilites
        4: Evaluate the log likelihood - check convergence
    '''

    def __init__(self,X,K):

        self.X = X # data set
        self.K = K # number of Guassian ditributions to use
        self.N = self.X.shape[0]
        self.M = self.X.shape[1]
        self.mu = np.array([]) # array of means for each of the distributions K x 2
        self.covariance = np.zeros((self.K,self.M,self.M)) # array of covariance matrices for each of the gaussian distributions
        self.S = np.zeros((self.N,)) # list of the assignments for each of the data points
        self.pi = np.zeros((self.K,)) # list of the mixing coefficients
        self.gauss_list = [i for i in range(self.K)] # list of the current gaussian distributions in our mixture model
        self.gamma = np.zeros((X.shape[0],self.K))

    def calculate_k_gauss(self):
        '''
        :return:
        '''
        for k in range(self.K):
            mu = self.mu[k]
            cov = self.covariance[k]
            self.gauss_list[k] = mvn(mu,cov)

    def draw_ellipse(self,position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))
    def e_step(self):

        for n in range(self.X.shape[0]):
            X_n = self.X[n]
            gamma_n_k = np.zeros((self.K,))
            N_k = 0 # normalizing constant
            for k in range(self.K):
                gamma_n_k[k] = self.pi[k]*self.gauss_list[k].pdf(X_n)
                N_k += gamma_n_k[k]

            gamma_n_k = gamma_n_k / N_k

            self.gamma[n] = gamma_n_k

    def empirical_covariance(self):
        for k in range(self.K):
            X_k_n = self.X[np.where(self.S == k)]
            self.covariance[k] = np.cov(X_k_n,rowvar=False) # using numpy function for computing covariance columns are variables

    def init(self,type='kmeans',mu=None,S=None):
        '''
        initialize the cluster mean locations
        For this basic implemenation just pick
        2 random points
        :param type: What type of init to do. ('kmeans','random')
        :param mu: Mean from k means seeding
        :param S: list of assignment computed from kmeans seeding
        :return:
        '''


        if type == 'kmeans':

            self.mu = mu.copy()

            self.S = S.copy()
        else:

            c_indexes = np.random.randint(0, self.X.shape[0], self.K)

            self.mu = self.X[c_indexes].copy()

        # calculate empirical variance of each cluster as it currently stands
        self.empirical_covariance()

        # create our initial gauss dist
        self.calculate_k_gauss()

        # initialize the mixin parameters
        self.pi = np.ones((self.K,)) * 1.0 / self.K

        self.e_step()  # now that we have gauss distributions lets assign points

        self.plot_gmm() # show the current distributions and data

    def m_step(self):
        pass

    def plot_gmm(self):
        ax = plt.gca()

        ax.scatter(X[:, 0], X[:, 1], c=self.S, s=40, cmap='viridis', zorder=2)

        ax.axis('equal')

        w_factor = 0.2 / self.pi.max()

        for pos, covar, w in zip(self.mu, self.covariance, self.pi):
            self.draw_ellipse(pos, covar, alpha=w * w_factor)

        plt.show()

    def run(self):
        '''
        GMM algorithm overview:
        :return:
        '''
        pass

    def variance(self,X):
        '''

        :param X:
        :return:
        '''

        return ((X - X.mean(axis=0))**2).sum(axis=0) / X.shape[0]




if __name__ == '__main__':

    print('Running Vanilla GMM')

    K = 2 # num clusters
    R = 1 # number of runs
    MaxIters = 10 # num max iterations

    mean = [-1,0]
    cov = [[1,0],[0,1]]
    x1 = np.random.multivariate_normal(mean, cov, 100)

    mean = [1,0]
    x2 = np.random.multivariate_normal(mean, cov, 100)

    X = np.concatenate((x1,x2))

    # FIRST RUN KMEANS FOR SEEDING
    kmeans = KMeans(K,X)

    C , S = kmeans.run(R,MaxIters)

    gmm = GMM(X,K)

    gmm.init(type='kmeans',mu=C,S=S)

    print('Done running GMM')