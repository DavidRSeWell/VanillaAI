import numpy as np


class Perceptron:

    def __init__(self,num_neurons=1,weights=np.zeros((1)),data=None):

        self.data = data
        self.num_neurons = num_neurons
        self.weights= weights
        self.X = None

    def activation(self,y_hat):

        return np.where(y_hat > 0,1,0)

    def E_in(self):
        '''
        Compute total in sample error
        :return:
        '''
        error = 0
        for j in range(self.X.shape[0]):

            x = self.X[j]

            x = np.reshape(x,(1,X.shape[1] + 1))

            y_hat = self.forward(x)

            t_j = self.Y[j]

            error += np.abs(y_hat - t_j)

        return error

    def forward(self,x):

        N = x.shape[1]

        y_j = 0

        for i in range(N):  # loop over the input nodes

            y_j += np.dot(self.weights[i][0], x[0][i])

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
        W = np.random.random((n + 1,self.num_neurons))
        self.weights = W

    def run_naive(self,data,labels,iters=100,alpha=0.025,threshold=0.5):
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

        error = self.E_in()

        print("Initial error {}".format(str(error)))

        for iter in range(iters):

            for m in range(M): # loop over the data

                x = self.X[m]

                x = np.reshape(x,(1,N + 1))

                y_hat = self.forward(x)

                t_j = self.Y[m] # the jth indicator 1 or 0

                # error
                error = y_hat - t_j

                # update weights
                for i in range(N + 1):

                    self.weights[i,0] -= alpha*(np.dot(error,x[0][i]))


            print('Error at iteration {} = {}'.format(str(iter),self.E_in()))

        print('Final Weights')
        print(self.weights)



if __name__ == '__main__':

    data = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    #data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])


    X = data[:,:2]
    Y = data[:,-1]

    perceptron = Perceptron()

    perceptron.run_naive(X,Y,iters=50)






