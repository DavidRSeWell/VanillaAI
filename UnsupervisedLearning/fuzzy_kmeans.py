import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class FuzzyKMeans:

    def __init__(self, num_clusters, data,m=2.0):
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.m = m # fuzzy ness hyperparamter

        self.K = num_clusters
        self.X = data

        self.C = None  # list of the means of each cluster
        self.W = None  # list of assignment for each data point

    def cluster_assignment(self):
        '''
        Update the mean of each of the centroids
        :return:
        '''
        for j, c_j in enumerate(self.C):

            W_j = (self.W[:,j])**self.m # get all the weights for the current centroid for each data point
            numerator = np.dot(W_j.T,self.X)
            denominator = W_j.sum()
            self.C[j] = numerator / denominator

    def init(self):
        '''
        initialize the cluster mean locations
        For this basic implemenation just pick
        2 random points
        :return:
        '''

        c_indexes = np.random.randint(0, self.X.shape[0], self.K)

        self.C = self.X[c_indexes].copy()

        #
        self.W = np.ones((self.X.shape[0],self.K)) / self.K

        self.point_assignment()  # now that we have centroids lets assign points

    def l2_loss(self):

        l2_loss = 0
        for j, c_j in enumerate(self.C):  # loop over each cluster
            for i, x in enumerate(self.X):
                l2_loss += (self.W[i,j]**2)*np.linalg.norm(x - c_j, 2)  # l2 norm for each data point

        return l2_loss

    def run(self, r, iters):
        '''
        Overview of algorithm
        2 steps
            1 assign each point to nearest centroid
            2 update clusters
        :return:
        '''

        best_loss = (0, 999999999999)  # tuple with (index of run, final loss)
        best_cluster = (None, None)  # tuple of (means, assignments)
        for r_i in range(r):  # do below for each run and track best result

            self.init()  # restart clusters each time
            print('Running run # = {}'.format(r_i))

            l2_loss = np.round(self.l2_loss(), 2)  # get initial loss
            self.plot(run=r_i, iter=0, l2_loss=l2_loss)

            for iter in range(1, iters + 1):
                self.point_assignment()
                self.cluster_assignment()
                l2_loss = np.round(self.l2_loss(), 2)
                if (iter % 10) == 0:
                    self.plot(run=r_i, iter=iter, l2_loss=l2_loss)

            if l2_loss < best_loss[1]:
                best_cluster = (self.C.copy(), self.W.copy())  # copy the best result
                best_loss = (r_i, l2_loss)

        self.plot(run='final', iter='final', l2_loss=best_loss[1])
        return best_cluster

    def plot(self, run=None, iter=None, l2_loss=None):
        '''
        A function for displaying the current state
        of the clusters. Display each cluster and its centroid
        :return:
        '''

        S = np.argmax(self.W,axis=1) # Assign the data point to the highest weight

        for i in range(self.K):
            data_cluster_i = self.X[np.where(S == i)]  # get the data with the current label
            centroid_i = self.C[i]
            color_i = self.colors[i]
            plt.scatter(data_cluster_i[:, 0], data_cluster_i[:, 1], color=color_i)
            plt.scatter(centroid_i[0], centroid_i[1], marker='X', s=200, color='black')

        plt.title('Clusters = {} run # = {} iteration ={} loss = {}'.format(self.K, run, iter, l2_loss))
        plt.show()

    def point_assignment(self):
        '''
        Assign each point in the data set to the closest centroid
        :return:
        '''

        # precompute distances from X - C for each x_i,C_j pair
        D = np.zeros((self.X.shape[0],self.K))
        for i , x in enumerate(self.X):
            for j , c_j in enumerate(self.C):
                D[i,j] = np.linalg.norm(self.X[i] - self.C[j])

        # starting with the slow version and will optimize later
        for i, x in enumerate(self.X):
            for j, c_j in enumerate(self.C):
                d_i_j = np.linalg.norm(self.X[i] - self.C[j])
                if d_i_j == 0.0: # If the point is the cluster then just assign the point a weight of one
                    self.W[i,j] = 1.0
                    continue

                denom = 0
                for k , c_k in enumerate(self.C):
                    d_i_k = np.linalg.norm(self.X[i] - self.C[k])
                    update = d_i_j / d_i_k
                    check = np.where(np.isnan(update))[0]
                    if len(check) > 0:
                        update = 0.0

                    update = (update) ** (2.0 / (self.m - 1.0))

                    denom += update

                self.W[i,j] = 1.0 / denom


if __name__ == '__main__':
    K = 3  # num clusters
    R = 1  # number of runs
    MaxIters = 20  # num max iterations

    #gmm_data = pd.read_csv('/Users/befeltingu/PSUClasses/AdvancedML/GMM_dataset.txt', delim_whitespace=True).as_matrix()
    gmm_data = pd.read_csv('/Users/befeltingu/downloads/cluster_dataset.txt', delim_whitespace=True).as_matrix()

    plt.scatter(gmm_data[:500, 0], gmm_data[:500, 1], color='b')
    plt.scatter(gmm_data[500:1000, 0], gmm_data[500:1000, 1], color='r')
    plt.scatter(gmm_data[1000:1500, 0], gmm_data[1000:1500, 1], color='g')
    plt.title('True clusters')
    plt.show()

    mean = [-1, 0]
    cov = [[1, 0], [0, 1]]
    x1 = np.random.multivariate_normal(mean, cov, 100)

    mean = [1, 0]
    x2 = np.random.multivariate_normal(mean, cov, 100)

    X = np.concatenate((x1, x2))

    k_means = FuzzyKMeans(K, gmm_data,m=2.0)

    k_means.run(R, MaxIters)

    print('Done running Fuzzy kmeans')
