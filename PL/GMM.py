import numpy as np

from scipy.stats import multivariate_normal as mvn


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

        self.mu = np.array([]) # array of means for each of the distributions K x 2
        self.covariance = np.array([]) # array of covariance matrices for each of the gaussian distributions
        self.S = np.array([]) # list of the assignments for each of the data points
        self.pi = np.array([]) # list of the mixing coefficients
        self.gauss_list = np.array([]) # list of the current gaussian distributions in our mixture model
        self.gamma = np.zeros((X.shape[0],self.K))

    def calculate_k_gauss(self):
        for k in range(self.K):
            mu = self.mu[k]
            cov = self.covariance[k]
            self.gauss_list[k] = mvn(mu,cov)

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
            self.covariance[k] = self.variance(X_k_n)

    def init(self):
        '''
        initialize the cluster mean locations
        For this basic implemenation just pick
        2 random points
        :return:
        '''

        c_indexes = np.random.randint(0, self.X.shape[0], self.K)

        self.mu = self.X[c_indexes].copy()

        self.empirical_covariance()

        self.S = np.zeros((self.X.shape[0],))

        #self.point_assignment()  # now that we have centroids lets assign points

    def m_step(self):
        pass

    def run(self):
        '''
        GMM algorithm overview:
        :return:
        '''
        pass

    def variance(self,X):

        return ((X - X.mean(axis=0))**2).sum(axis=0) / X.shape[0]




if __name__ == '__main__':

    #gmm = GMM()
    pass