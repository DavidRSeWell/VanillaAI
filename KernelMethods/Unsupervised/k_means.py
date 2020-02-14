import matplotlib.pyplot as plt
import numpy as np



class KMeans:
    def __init__(self,num_clusters,data):
        self.colors = ['b','g','r','c','m','y','k','w']

        self.K = num_clusters
        self.X = data

        self.C = None # list of the means of each cluster
        self.S = None # list of assignment for each data point


    def cluster_assignment(self):
        '''
        Update the mean of each of the centroids
        :return:
        '''

        for j,c_j in enumerate(self.C):
            data_j = self.X[np.where(self.S == j)]
            mean = np.mean(data_j,axis=0)
            self.C[j] = mean

    def init(self):
        '''
        initialize the cluster mean locations
        For this basic implemenation just pick
        2 random points
        :return:
        '''

        c_indexes = np.random.randint(0,self.X.shape[0],2)

        self.C = self.X[c_indexes].copy()

        self.S = np.zeros((self.X.shape[0],))

        self.point_assignment() # now that we have centroids lets assign points

    def run(self,iters):
        '''
        Overview of algorithm
        2 steps
            1 assign each point to nearest centroid
            2 update clusters
        :return:
        '''

        for iter in range(iters):
            self.point_assignment()
            self.cluster_assignment()
            self.plot()

    def plot(self):
        '''
        A function for displaying the current state
        of the clusters. Display each cluster and its centroid
        :return:
        '''

        for i in range(self.K):
            data_cluster_i = self.X[np.where(self.S == i)] # get the data with the current label
            centroid_i = self.C[i]
            color_i = self.colors[i]
            plt.scatter(data_cluster_i[:,0],data_cluster_i[:,1],color=color_i)
            plt.scatter(centroid_i[0],centroid_i[1],marker='X')

        plt.show()

    def point_assignment(self):
        '''
        Assign each point in the data set to the closest centroid
        :return:
        '''

        # starting with the slow version and will optimize later
        for i , x in enumerate(self.X):
            dist = 1000
            for j,c_j in enumerate(self.C):
                l2_dist = np.linalg.norm(x - c_j,2)
                if l2_dist < dist:
                    self.S[i] = j
                    dist = l2_dist


if __name__ == '__main__':

    K = 2 # num clusters
    Iters = 10 # num iters
    MaxIters = 100 # max iters

    mean = [-1,0]
    cov = [[1,0],[0,1]]
    x1 = np.random.multivariate_normal(mean, cov, 100)

    mean = [1,0]
    x2 = np.random.multivariate_normal(mean, cov, 100)

    X = np.concatenate((x1,x2))


    plt.scatter(x1[:,0],x1[:,1],color='blue')
    plt.scatter(x2[:,0],x2[:,1],color='red')


    k_means = KMeans(K,X)

    k_means.init()

    k_means.run(3)
    print('Done running K means')
