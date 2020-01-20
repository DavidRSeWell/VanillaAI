import numpy as np
import os
import urllib

from sklearn import datasets
from sklearn.datasets import fetch_mldata
from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home


def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)

class Perceptron:

    def __init__(self,targets,weights=np.zeros((1)),data=None):

        self.data = data
        self.targets = targets
        self.weights= weights
        self.X = None

    def activation(self,y_hat):

        # get max output

        max_index = np.argmax(y_hat)

        y_hat = np.zeros(y_hat.shape)

        y_hat.T[max_index] = 1

        return y_hat

    def E_in(self):
        '''
        Compute total in sample error
        :return:
        '''
        error = 0
        for j in range(self.X.shape[0]):

            x = self.X[j]

            x = np.reshape(x,(1,self.X.shape[1] + 1))

            y_hat = self.forward(x)

            t_j = self.Y[j]

            error += np.abs(y_hat - t_j)

        return error

    def forward(self,x):

        N = x.shape[1]

        #y_j = 0

        #for i in range(N):  # loop over the input nodes

        #    y_j += np.dot(self.weights[i][0], x[0][i])

        y_j = np.dot(x,self.weights)

        y_j = self.activation(y_j)

        return y_j

    def get_mse(self):

        pass

    def init_weights(self,n):
        '''
        Function to initilize the set of weight and biases
        for the network
        n = int. Length of input vector
        :return:
        '''
        W = np.random.random((n + 1,self.targets))
        self.weights = W

    def run(self,data,labels,iters=100,alpha=0.025,threshold=0.5):
        '''
        Naive means it is the most basic without any thought to optimization
        For iterations in iters
            for data in data_set:
                y = label
                y_hat = prediction
                error = error(y,y_hat)
                update(error,weight)
        :return:
        '''

        print('Running naive algorithm')

        print('Initializing weights')

        N = data.shape[1] # Length of the feature vector

        M = data.shape[0] # Length of the data set

        self.X = np.concatenate((data.copy(), -1.0*np.ones((M,1))),axis=1)

        self.Y = labels.copy()

        self.init_weights(N)

        #error = self.E_in()

        #print("Initial error {}".format(str(error)))

        for iter in range(iters):

            for m in range(M): # loop over the data

                x = self.X[m]

                x = np.reshape(x,(1,N + 1))

                y_hat = self.forward(x)

                t_j = np.zeros(y_hat.shape)

                t_j.T[self.Y[m]] = 1 # the jth indicator 1 or 0

                # error
                error = y_hat - t_j

                # update weights
                #for i in range(N + 1):

                #    self.weights[i,0] -= alpha*(np.dot(error,x[0][i]))

                self.weights -= alpha*(np.dot(x.T,error))

            #print('Error at iteration {} = {}'.format(str(iter),self.E_in()))

        print('Final Weights')
        #print(self.weights)



if __name__ == '__main__':

    #mnist = datasets.load_digits()
    #mnist = fetch_mldata('MNIST original') #data_home='/Users/befeltingu/PSUClasses/ML/')

    fetch_mnist()
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata("MNIST original")

    ###########################################################################
    # the dataset contains a set of 150 records under five attributes -
    # X = petal length, petal width, sepal length, sepal width
    # Y (target) = species (setosa,versicolor,virginica )
    ###########################################################################

    mnist_data = mnist['data'].copy()
    mnist_targets = mnist['target'].copy()

    mnist_data = mnist_data.reshape((len(mnist_data), -1))

    percept = Perceptron(targets=10)

    percept.run(mnist_data,mnist_targets,iters=100,alpha=0.025)

    print('Done running mnist perceptron')



