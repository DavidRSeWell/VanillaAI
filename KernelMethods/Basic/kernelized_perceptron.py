import matplotlib.pyplot as plt

from numpy.random import multivariate_normal
import numpy as np


def get_circle(radius, num_points=100):
    x_points = np.linspace(-radius, radius, num_points)
    circle_points_y = []
    circle_points_x = []
    for x_i in x_points:
        y_i = np.sqrt(radius ** 2 - x_i ** 2)
        circle_points_x.append(x_i)
        circle_points_x.append(x_i)
        circle_points_y.append(y_i)
        circle_points_y.append(-1.0 * y_i)

    return circle_points_y, circle_points_x

def kernel_k1(x,y):
    return np.dot(x,y)

def kernel_k2(x,y,p=2):

    k1 = kernel_k1(x,y)

    k2 = (1 + k1)**p
    return k2

def kernel_k3(x,y,a=1,b=1):

    return np.tanh(a*kernel_k1(x,y) + b)

def phi_I(x):
    return x



def phi_p(x):
    phi_x = [x[0]**2,x[1]**2,x[0]**2 + x[1]**2]

    return np.array(phi_x)

def run_kernalized_perceptron(X ,Y,kernel=kernel_k1,phi=phi_I):
    '''

    :param X:
    :param Y:
    :param kernel: kernel function to use
    :param phi: function: if using phi transformation
    :return:
    '''

    # init alpha to zero
    alphas = np.array([0 for _ in range(X.shape[0])])

    for iter in range(1): # for running more than one iteration. This didnt seem to matter

        print('Running iteration {}'.format(iter))

        for i in range(X.shape[0]):

            x_input = X[i]
            # if trying to use phi w / inner product
            #x = phi(x)

            # using loop for creating kernel
            k_mm = 0
            for j in range(X.shape[0]):

                k_mm += alphas[j] * kernel(X[j] ,x_input)

            y_hat = k_mm

            if (y_hat *Y[i] < 0) or i == 0: # if wrong or if its the first iteration

                alphas[i] = alphas[i] + Y[i] # update only the alpha of the index that we are trying to predict

    return alphas

def kernel_perceptron_check(X_new, X, Y, alphas,kernel,phi):
    num_wrong = 0
    num_right = 0

    for i in range(X_new.shape[0]):

        x = X_new[i]

        #x = phi(x)

        k_mm = 0
        for j in range(X.shape[0]):
            k_mm += alphas[j] * kernel(X[j],x)

        y_hat = k_mm

        # print('y hat = {}'.format(y_hat))
        # print('actual = {}'.format(Y[i]))

        if y_hat * Y[i] < 0:  # wrong
            num_wrong += 1
        else:
            num_right += 1

    print('Num wrong = {}'.format(num_wrong))
    print('Num right = {}'.format(num_right))
    print('%Correct = {} %'.format(float(num_right) / (num_wrong + num_right)))

if __name__ == '__main__':

    print('Running kernalized perceptron')

    ###################################################
    # Generate a circle
    ###################################################


    run_create_circle_data = 1
    if run_create_circle_data:

        circle_y, circle_x = get_circle(2)

        colors = ('blue', 'red')
        groups = ('+', '-')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(circle_x, circle_y, color='red', label='-')
        x_p = multivariate_normal([2, 0], [[0.5, 0], [0, 0.5]], size=200)
        x_p_1, x_p_2 = zip(*x_p)
        ax.scatter(x_p_1, x_p_2, color='blue', label='+')
        # ax.scatter(x_m_1,x_m_2,color='red',label='-')
        plt.title('Data')
        plt.legend()
        plt.show()
        x_m = np.array([circle_x, circle_y]).T


    run_create_norm_data = 0
    if run_create_norm_data:


        #####################
        # Generate some data
        #####################
        n_m = 100  # number minus
        n_p = 100  # number plus
        n = n_m + n_p
        # set mean and covariance for both samples
        mean_p = [2, 1]
        cov_p = [[2, 0], [0, 2]]
        mean_m = [-2, 1]
        cov_m = [[2, 0], [0, 2]]

        x_p = multivariate_normal(mean_p, cov_p, size=n_p)
        x_m = multivariate_normal(mean_m, cov_m, size=n_m)

        x_p_1, x_p_2 = zip(*x_p)
        x_m_1, x_m_2 = zip(*x_m)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x_p_1, x_p_2, color='blue', label='+')
        ax.scatter(x_m_1,x_m_2,color='red',label='-')
        plt.title('Data')
        plt.legend()
        plt.show()

    X_data = np.concatenate((x_p, x_m))

    Y_data = [1 for _ in range(len(x_p))] + [-1 for _ in range(len(x_m))]
    Y_data = np.array(Y_data)

    train_data_indexes = np.random.choice(len(X_data),int(len(X_data)*0.80), replace=False)
    test_data_indexes = [i for i in range(len(X_data)) if i not in train_data_indexes]  # get whats left over for testing

    train_data_x = X_data[train_data_indexes]
    train_data_y = Y_data[train_data_indexes]

    test_data_x = X_data[test_data_indexes]
    test_data_y = Y_data[test_data_indexes]

    alphas_trained = run_kernalized_perceptron(train_data_x, train_data_y,kernel_k2,phi=phi_I)

    kernel_perceptron_check(test_data_x,train_data_x,test_data_y,alphas_trained,kernel_k2,phi=phi_I)