################################################
# Module for running some basic
# binary classifiers using different kernel methods
################################################
import numpy as np
from sklearn import datasets


##########################
# Define kernels to use
##########################
def kernel_k1(x,y):
    return np.dot(x,y)

def kernel_k3(x,y):
    return ( (1+ np.dot(x,y) )**2)

def run_binary_iris_classifier(iris_data_x,iris_data_y,train_data_indexes,test_data_x,test_data_y):
    '''
    Example of running binary classifier on iris data set
    :return:
    '''


    ###########################
    # Create Classifier
    ###########################
    positive_indexes = [i for i in train_data_indexes if iris_data_y[i] == 1]
    negative_indexes = [i for i in train_data_indexes if iris_data_y[i] != 1]
    x_p = iris_data_x[positive_indexes]
    x_m = iris_data_x[negative_indexes]

    # calculate center of positive and negative vectors and get our offset constant b
    # b = 1/2
    k_pp = np.zeros((len(x_p), len(x_p)))
    k_mm = np.zeros((len(x_m), len(x_m)))

    for i in range(k_pp.shape[0]):
        for j in range(k_pp.shape[1]):
            k_pp[i][j] = np.dot(x_p[i], x_p[j])

    for i in range(k_mm.shape[0]):
        for j in range(k_mm.shape[1]):
            k_mm[i][j] = np.dot(x_m[i], x_m[j])

    b = (k_mm.sum() / len(k_mm) ** 2 - k_pp.sum() / len(k_pp) ** 2)

    b = b / 2.0

    data_x = np.concatenate((x_p, x_m))  # combine positive and negative vectors together

    alpha = [1.0 / len(x_p) for _ in range(len(x_p))] + [-1 / len(x_m) for _ in range(len(x_m))]

    ###################################
    # Evaluate the kernel on test set
    ###################################
    num_right = 0
    num_wrong = 0
    for i in range(test_data_x.shape[0]):

        x_test = test_data_x[i]
        y_test = test_data_y[i]
        k_xx = 0
        for j in range(data_x.shape[0]):  # loop over all points in data set
            k_xx += alpha[j] * kernel_k1(data_x[j], x_test)

        y_hat = k_xx + b
        # check result vs actual
        if y_hat * y_test > 0:
            num_right += 1
        else:
            num_wrong += 1

    print('Number right = {}'.format(num_right))
    print('Number wrong = {}'.format(num_wrong))
    print('Percent correct = {}'.format(float(num_right) / (num_right + num_wrong)))


if __name__ == '__main__':


    binary_iris_classifier = 0
    if binary_iris_classifier:

        iris_data = datasets.load_iris()

        ###########################################################################
        # the dataset contains a set of 150 records under five attributes -
        # X = petal length, petal width, sepal length, sepal width
        # Y (target) = species (setosa,versicolor,virginica )
        ###########################################################################

        iris_data_x = iris_data.data
        iris_data_y = iris_data.target

        # first start with setosa versus versicolor
        print('Running kernel classifier on Setosa vs Versicolor')

        # Random sample 80 data points for training
        # From the first 100 points in the data set grab 80 at random and without replacement
        train_data_indexes = np.random.choice(100, 80, replace=False)
        test_data_indexes = [i for i in range(100) if i not in train_data_indexes]  # get whats left over for testing

        train_data_x = iris_data_x[train_data_indexes]
        train_data_y = iris_data_y[train_data_indexes]

        test_data_x = iris_data_x[test_data_indexes]
        test_data_y = iris_data_y[test_data_indexes]

        # in keeping with the form of this simple binary classifier
        # I will break the data into labels with {1,-1}
        # I replace y labels with 0 to be -1
        train_data_y[np.where(train_data_y == 0)] = -1
        test_data_y[np.where(test_data_y == 0)] = -1

        run_binary_iris_classifier(iris_data_x, iris_data_y, train_data_indexes, test_data_x, test_data_y)


        # second run on Veriscolor vs virginica
        print('Running kernel classifier on Veriscolor vs virginica')

        #############################################
        # Run same code below for Veriscolor vs virginica
        #############################################

        # Random sample 80 data points for training
        # From the last 100 in the data set grab 80 at random and without replacement
        possible_indexes = [i for i in range(50, 150)]
        train_data_indexes = np.random.choice(possible_indexes, 80, replace=False)
        test_data_indexes = [i for i in range(50, 150) if
                             i not in train_data_indexes]  # get whats left over for testing

        train_data_x = iris_data_x[train_data_indexes]
        train_data_y = iris_data_y[train_data_indexes]

        test_data_x = iris_data_x[test_data_indexes]
        test_data_y = iris_data_y[test_data_indexes]

        # in keeping with the form of this simple binary classifier
        # I will break the data into labels with {1,-1}
        # I replace y labels with 0 to be -1
        train_data_y[np.where(train_data_y == 2)] = -1  # treat virginica as the negative example
        test_data_y[np.where(test_data_y == 2)] = -1

        run_binary_iris_classifier(iris_data_x, iris_data_y, train_data_indexes, test_data_x, test_data_y)