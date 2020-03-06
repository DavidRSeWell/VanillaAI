import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal as mvn

#LOCAL
from UnsupervisedLearning.k_means import KMeans

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
        self.colors = ['b','g','r','c','m','y','k','w']
        self.color_map = {} # mapping from cluster index --> color
        self.covariance = np.zeros((self.K,self.M,self.M)) # array of covariance matrices for each of the gaussian distributions
        self.S = np.zeros((self.N,)) # list of the assignments for each of the data points
        self.pi = np.zeros((self.K,)) # list of the mixing coefficients
        self.gauss_list = [i for i in range(self.K)] # list of the current gaussian distributions in our mixture model
        self.gamma = np.zeros((self.N,self.K))

    def assign_points(self):
        '''
        Re assign points to most likely distribution
        :return:
        '''
        new_s = [np.argmax(self.gamma[n]) for n in range(self.N)]

        self.S = new_s

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

        # map cluster index to colors
        for k in range(self.K):
            self.color_map[k] = self.colors[k]

        # calculate empirical variance of each cluster as it currently stands
        self.empirical_covariance()

        # create our initial gauss dist
        self.calculate_k_gauss()

        # initialize the mixin parameters
        self.pi = np.ones((self.K,)) * 1.0 / self.K

        self.e_step()  # now that we have gauss distributions lets assign points

        ll = self.log_likelihood()

        self.plot_gmm(ll,'init') # show the current distributions and data

        return ll

    def log_likelihood(self):
        '''
        Calculate likelihood of data given parameters
        :return:
        '''
        ll = 0.0
        for n in range(self.N):

            ll_k = 0.0
            for k in range(self.K):
                ll_k += self.pi[k] * self.gauss_list[k].pdf(self.X[n])

            ll += np.log(ll_k)

        return ll

    def m_step(self):
        '''
        Update the model parameters
        mean, covariance, pi
        :return:
        '''

        for k in range(self.K):

            N_k = self.gamma[:,k].sum()

            mu_k = np.dot(self.gamma[:,k].T,self.X) / N_k # dot product with gamma_k = (n x 1)^T , X = n x m result = 1 x m

            mu_k = mu_k.reshape((self.M,1))

            cov_k = np.zeros((self.M,self.M))

            for n in range(self.N):
                x_n = np.reshape(self.X[n],(self.X[n].shape[0],1))
                conv_k_n =  self.gamma[n][k] * np.dot((x_n - mu_k),(x_n - mu_k).T)    #  (1 x 1) ( m x 1 ) ( 1 x m)
                cov_k = np.add(conv_k_n,cov_k)

            cov_k = cov_k / N_k

            self.pi[k] = N_k / self.N

            self.mu[k] = mu_k.reshape((self.M,))

            self.covariance[k] = cov_k

        self.calculate_k_gauss() # update our gaussian dist given the new parameters

    def plot_gmm(self,ll=None,iter=None):
        '''

        :param ll: log likelihood
        :param iter: current iterations
        :return:
        '''
        ax = plt.gca()

        colors = [self.color_map[s] for s in self.S]

        ax.scatter(self.X[:, 0], self.X[:, 1], s=40, color=colors, zorder=2)

        ax.axis('equal')

        w_factor = 0.2 / self.pi.max()

        for pos, covar, w in zip(self.mu, self.covariance, self.pi):
            self.draw_ellipse(pos, covar, alpha=w * w_factor)

        plt.title('GMM iter ={} log likelihood = {}'.format(iter,ll))

        plt.show()

    def run(self,runs,maxiters):
        '''
        GMM algorithm overview:
        :return:
        '''


        current_best_ll = -999999
        current_best_model = (None,None,None,None,None) # (mu,covariance,S,pi,gauss list,gamma)
        for run in range(runs):

            # re initialize
            # FIRST RUN KMEANS FOR SEEDING
            kmeans = KMeans(self.K, self.X)

            # run a single run of kmeans with max iterations at 5
            C, S = kmeans.run(1, 5)

            init_ll = self.init(type='kmeans', mu=C, S=S)

            ll_list = [init_ll]
            diff_ll = []
            current_ll = init_ll
            epsilon = 0.01 # how much to allow for delat_ll before converged
            for iter in range(maxiters):

                # E step
                self.e_step()

                self.m_step()

                ll = self.log_likelihood()

                ll_list.append(ll)

                print('LL at iter {} is {}'.format(iter,ll))

                if (iter % 10) == 0: # plot of x number of iterations
                    self.plot_gmm(ll,iter)

                self.assign_points() # used for plotting assign point to max Posterier

                diff_ll.append(np.abs(current_ll - ll))

                average_change = np.abs(np.array(diff_ll)[-5:].mean()) # for convergence checking
                print('average change = {}'.format(average_change))
                current_ll = ll

                if average_change <= epsilon:
                    print('Run {} converged! at iteration {}'.format(run,iter))
                    break

            final_ll = ll_list[-1]

            if final_ll < current_best_ll:

                current_best_model = (self.mu.copy(),self.covariance.copy(),self.S.copy(),self.pi.copy(),
                                      self.gauss_list.copy(),self.gamma.copy())


                current_best_ll = final_ll


    def variance(self,X):
        '''
        :param X:
        :return:
        '''
        return ((X - X.mean(axis=0))**2).sum(axis=0) / X.shape[0]




if __name__ == '__main__':

    print('Running Vanilla GMM')

    K = 3 # num gaussians
    R = 1 # number of runs
    MaxItersKmeans = 1 # num max iterations
    MaxIters = 50 # num max iterations

    gmm_data = pd.read_csv('/Users/befeltingu/PSUClasses/AdvancedML/GMM_dataset.txt',delim_whitespace=True).as_matrix()

    plt.scatter(gmm_data[:500, 0], gmm_data[:500, 1], color='b')
    plt.scatter(gmm_data[500:1000, 0], gmm_data[500:1000, 1], color='r')
    plt.scatter(gmm_data[1000:1500, 0], gmm_data[1000:1500, 1], color='g')
    plt.title('True clusters')
    plt.show()

    gmm = GMM(gmm_data,K)

    gmm.run(R,maxiters=MaxIters)

    print('Done running GMM')
