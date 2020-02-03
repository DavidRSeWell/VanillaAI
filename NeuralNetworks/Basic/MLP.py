import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

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

def sigmoid(X):

    return 1.0 / (1 + np.exp(-1.0*X))

class MLP:

    def __init__(self,X,Y,H,alpha,eta,label_count):
        '''

        :param X: Our full data set
        :param Y: Labels
        :param H: Size of hidden later
        :param alpha: learning rate
        '''
        self.alpha = alpha # momentum learning rate
        self.eta = eta # first order learning rate
        self.label_count = label_count
        self.H = H
        self.X = X
        self.Y = Y

        self.w1 = None # input to hidden weights
        self.w2 = None # hidden to outpu weights

        self.delta_w1 = None # Use for updating momentum
        self.delta_w2 = None

    def backward(self,yhat,target,a,x):
        '''
        Update the weights of the network
        starting from the output layer and working
        backward
        :param yhat: output layer 1 x label count
        :param target: integer for the correct label
        :param a: activation layer 1 x hidden size
        :return:
        '''

        #target_vector = np.zeros((1,self.label_count))

        #target_vector[0][target] = 1

        #o_error = np.dot(yhat - target_vector,yhat.T)

        #o_error = np.dot(o_error,(1.0 - yhat))

        o_error = (yhat - target)*yhat*(1.0 - yhat)

        h_error = a * (1.0 - a) * np.dot(o_error,self.w2.T)

        delta_w1 = self.eta*np.dot(x.T,h_error[:,:-1]) + self.alpha*self.delta_w1

        delta_w2 = self.eta*np.dot(a.T,o_error) + self.alpha*self.delta_w2

        self.w1 -= delta_w1

        self.w2 -= delta_w2

        self.delta_w1 = delta_w1.copy()

        self.delta_w2 = delta_w2.copy()

    def confusion_matrix(self):
        '''
        Build confusion matrix from test set
        :return:
        '''

        Y, h = self.forward(self.test_x)

        Y_guess = np.argmax(Y,1)

        con_mat = np.zeros((self.label_count,self.label_count))

        for i in range(self.label_count):
            for j in range(self.label_count):
                con_mat[i,j] = np.sum(np.where(self.test_y == i,1,0)*np.where(Y_guess == j,1,0))

        return con_mat


    def E_in(self):
        '''
        Compute total in sample error
        :return:
        '''

        Y, _ = self.forward(self.train_x)

        Y_guess = np.argmax(Y,1)

        diff = self.train_y - Y_guess

        diff = np.where(diff == 0.0,0,1)

        num_wrong = diff.sum()

        loss = (float(num_wrong) / self.train_y.shape[0])*100.0

        correct = ((self.train_y.shape[0] - num_wrong) / float(self.train_y.shape[0]))*100.0

        return loss , correct

    def E_out(self):
        '''
        Compute total in sample error
        :return:
        '''

        Y, _ = self.forward(self.test_x)

        Y_guess = np.argmax(Y, 1)

        diff = self.test_y - Y_guess

        diff = np.where(diff == 0.0, 0, 1)

        num_wrong = diff.sum()

        loss = (float(num_wrong) / self.test_y.shape[0]) * 100.0

        correct = ((self.test_y.shape[0] - num_wrong) / float(self.test_y.shape[0])) * 100.0

        return loss, correct

    def forward(self,x):
        '''
        Function for computing the forward pass of the nework
        :param x:
        :return:
        '''

        h = sigmoid(np.dot(x,self.w1))

        h = np.concatenate((h.copy(), 1.0*np.ones((len(h),1))),axis=1) # adding a bias term to the hidden layer

        o = sigmoid(np.dot(h,self.w2))

        return o , h

    def init_weights(self):

        n = self.X.shape[1]

        self.w1 = (np.random.random((n,self.H)) - 0.5) / 10.0

        self.w2 = (np.random.random((self.H + 1,self.label_count)) - 0.5) / 10.0 # add + 1 for the bias term

        self.delta_w1 = np.zeros(self.w1.shape)

        self.delta_w2 = np.zeros(self.w2.shape)

    def run(self,epochs,split_rule=0.80):
        '''
        Function for driving the algorithm
        :param epochs:
        :param split_rule: percent of data set to use for training
        :return:
        '''

        m = self.X.shape[0]
        n = self.X.shape[1]

        train_data_indexes = np.random.choice(m, int(m*split_rule), replace=False)

        test_data_indexes = [i for i in range(m) if i not in train_data_indexes]  # get whats left over for testing

        self.X = np.concatenate((self.X.copy(), 1.0*np.ones((m,1))),axis=1) # adding a bias term to the input

        self.train_x = self.X[train_data_indexes]
        self.train_y = self.Y[train_data_indexes]

        self.test_x = self.X[test_data_indexes]
        self.test_y = self.Y[test_data_indexes]

        # Create output vectors with value of 0.9 at the correct index and 0.1 otherwise
        train_targets = np.zeros((len(self.train_y), self.label_count))*0.1

        index_array = [i for i in range(len(self.train_y))]

        train_targets[index_array, self.train_y.astype(np.int)] = 0.9

        self.init_weights()

        train_loss, train_correct = self.E_in()

        test_loss, test_correct = self.E_out()

        train_corrects = [train_correct]
        test_corrects = [test_correct]

        for k in range(epochs):
            for i , x in enumerate(self.train_x):

                x = np.reshape(x,(1,n + 1))

                y_hat, a = self.forward(x)

                target = train_targets[i]

                self.backward(y_hat,target,a,x)


            train_loss, train_correct = self.E_in()
            test_loss, test_correct = self.E_out()

            train_corrects.append(train_correct)
            test_corrects.append(test_correct)

            print('%{} correct for iter {} '.format(str(test_correct),str(k)))

        return train_corrects , test_corrects



if __name__ == '__main__':



    fetch_mnist()
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata("MNIST original")

    ALPHA = 0
    ETA = 0.1
    HIDDEN_LAYER_SIZE = 100

    mnist_data = mnist['data'].copy()
    mnist_data = mnist_data.reshape((len(mnist_data), -1))


    mnist_data = mnist_data / 255.0

    mnist_targets = mnist['target'].copy()

    mnist_targets = mnist_targets.astype(np.int)

    num_targets = len(set(mnist_targets))

    mlp_perceptron = MLP(mnist_data,mnist_targets,HIDDEN_LAYER_SIZE,ETA,ALPHA,num_targets)

    train_corrects , test_corrects = mlp_perceptron.run(50,0.80)

    plt.plot(train_corrects,label='Training')
    plt.plot(test_corrects,label='Test')
    plt.title('alpha = {} momentum = {} hidden size = {}'.format(ETA,ALPHA,HIDDEN_LAYER_SIZE))
    plt.legend()
    plt.show()


    confusion_mat = mlp_perceptron.confusion_matrix()

    conf_mat_df = pd.DataFrame(np.round(confusion_mat, 5))

    ax = sns.heatmap(conf_mat_df, annot=True, fmt='g')

    bottom, top = ax.get_ylim()

    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.ylabel('Actuall Class')
    plt.xlabel('Predicted Class')
    plt.show()
    plt.close()
