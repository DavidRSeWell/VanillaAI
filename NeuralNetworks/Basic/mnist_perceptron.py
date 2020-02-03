import matplotlib.pyplot as plt
import numpy as np
import os
import urllib
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

class Perceptron:

    def __init__(self,training_data_x,training_data_y,
                 test_data_x,test_data_y,weights=np.zeros(1)):

        self.train_x = training_data_x
        self.train_y = training_data_y

        self.test_x = test_data_x
        self.test_y = test_data_y

        self.weights = weights

    def activation(self,y_hat):

        # get max output

        max_index = np.argmax(y_hat)

        y_hat = np.zeros(y_hat.shape)

        y_hat.T[max_index] = 1

        return y_hat

    def confusion_matrix(self,targets):
        '''
        Build confusion matrix from test set
        :return:
        '''

        Y = np.dot(self.test_x,self.weights)

        Y_guess = np.argmax(Y,1)

        con_mat = np.zeros((targets,targets))

        for i in range(targets):
            for j in range(targets):
                con_mat[i,j] = np.sum(np.where(self.test_y == i,1,0)*np.where(Y_guess == j,1,0))

        return con_mat


    def E_in(self):
        '''
        Compute total in sample error
        :return:
        '''

        Y = np.dot(self.train_x,self.weights)

        Y_guess = np.argmax(Y,1)

        diff = self.train_y - Y_guess

        diff = np.where(diff == 0.0,0,1)

        num_wrong = diff.sum()

        loss = (float(num_wrong) / self.train_y.shape[0])*100.0

        correct = ((self.train_y.shape[0] - num_wrong) / float(self.train_y.shape[0]))*100.0

        print('Train Num wrong = {}'.format(num_wrong))
        print('% train num loss = {}'.format(loss))

        return loss , correct

    def E_out(self):
        '''
        Compute total in sample error
        :return:
        '''

        Y = np.dot(self.test_x,self.weights)

        Y_guess = np.argmax(Y,1)

        diff = self.test_y - Y_guess

        diff = np.where(diff != 0,1,0)

        num_wrong = diff.sum()

        loss = (float(num_wrong) / self.test_y.shape[0])*100.0

        correct = ((self.test_y.shape[0] - num_wrong) / float(self.test_y.shape[0]))*100.0

        print('Test Num wrong = {}'.format(num_wrong))
        print('% Test Num loss = {}'.format(loss))

        return loss , correct

    def forward(self,x):

        y_j = np.dot(x,self.weights)

        y_j = np.where(y_j > 0,1,0)

        return y_j

    def init_weights(self,n):
        '''
        Function to initilize the set of weight and biases
        for the network
        n = int - Num targets
        :return:
        '''
        W = (np.random.random((self.train_x.shape[1],n)) - 0.5) / 10.0

        self.weights = W

    def run(self,iters=100,alpha=0.025):
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

        #N = data.shape[1] # Length of the feature vector

        M_train = self.train_x.shape[0] # Length of the data set
        M_test = self.test_x.shape[0] # Length of the data set

        self.train_x = np.concatenate((self.train_x.copy(), -1.0*np.ones((M_train,1))),axis=1)
        self.test_x = np.concatenate((self.test_x.copy(), -1.0*np.ones((M_test,1))),axis=1)

        targets = len(set(train_y))

        train_targets = np.zeros((M_train,targets))

        index_array = [i for i in range(M_train)]

        train_targets[index_array,self.train_y.astype(np.int)] = 1

        self.init_weights(targets)

        train_error,train_correct = self.E_in()
        test_error,test_correct = self.E_out()

        train_errors = [train_error]
        test_errors = [test_error]

        train_corrects = [train_correct]
        test_corrects = [test_correct]

        for iter in range(iters):

            for i , x in enumerate(self.train_x):

                x = np.reshape(x,(1,x.shape[0]))

                y_hat = self.forward(x)

                error = y_hat - train_targets[i].T

                self.weights -= alpha*np.dot(x.T,error)

            if (iter % 1 == 0):

                train_error,train_correct = self.E_in()

                test_error,test_correct = self.E_out()

                train_errors.append(train_error)

                test_errors.append(test_error)

                print('Error at iteration {} = {}'.format(str(iter),train_error))

                train_corrects.append(train_correct)

                test_corrects.append(test_correct)

        return train_errors, test_errors, train_corrects, test_corrects




if __name__ == '__main__':


    fetch_mnist()
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata("MNIST original")

    ALPHA = 0.1

    mnist_data = mnist['data'].copy()
    mnist_data = mnist_data.reshape((len(mnist_data), -1))


    mnist_data = mnist_data / 255.0

    mnist_targets = mnist['target'].copy()

    train_data_indexes = np.random.choice(len(mnist_data), 60000, replace=False)
    test_data_indexes = [i for i in range(len(mnist_data)) if
                         i not in train_data_indexes]  # get whats left over for testing

    train_x = mnist_data[train_data_indexes]
    train_y = mnist_targets[train_data_indexes]

    test_x = mnist_data[test_data_indexes]
    test_y = mnist_targets[test_data_indexes]

    percept = Perceptron(train_x,train_y,test_x,test_y)

    train_errors, test_errors, train_correct, test_correct = percept.run(iters=50,alpha=ALPHA)

    plt.plot([i for i in range(len(train_correct))],train_correct,label='training')

    plt.plot([i for i in range(len(test_correct))],test_correct,label='test')

    plt.legend()

    plt.title('MNIST Experiment alpha = {} best test accuracy = {}'.format(ALPHA,np.max(test_correct)))
    plt.ylabel('% correct')
    plt.xlabel('% epoch')

    plt.show()

    plt.close()

    print('Building confusion matrix')
    confusion_mat = percept.confusion_matrix(10)

    conf_mat_df = pd.DataFrame(np.round(confusion_mat,5))

    ax = sns.heatmap(conf_mat_df,annot=True,fmt='g')

    bottom, top = ax.get_ylim()

    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.ylabel('Actuall Class')
    plt.xlabel('Predicted Class')
    plt.show()
    plt.close()


    print('Done running mnist perceptron')



