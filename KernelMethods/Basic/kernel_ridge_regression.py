#############################
# Module for writing ridge regression
# as a kernel
#############################
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

from Space.BinarySpace import BinarySpace
from KernelMethods.Kernels import KernelsDict

from sklearn import datasets

def contour_lines(xmin,xmax,ymin,ymax,X,alphas,kernel,label1,label2):

    x_seq = np.linspace(xmin,xmax,20)

    y_seq = np.linspace(ymin,ymax,20)

    X1 , X2 = np.meshgrid(x_seq,y_seq)

    z = np.zeros((len(x_seq),len(x_seq)))

    for x_i in range(len(x_seq)):
        for x_j in range(len(y_seq)):

            x = np.array([x_seq[x_i],y_seq[x_j]])

            k_mm = 0
            for j in range(X.shape[0]):
                k_mm += alphas[j] * kernel(X[j], x)

            y_hat = k_mm

            # print('y hat = {}'.format(y_hat))
            # print('actual = {}'.format(Y[i]))

            label1_dis = np.abs(y_hat - label1)
            label2_dis = np.abs(y_hat - label2)
            guess = label2

            if label1_dis < label2_dis:
                guess = label1

            z[x_i][x_j] = y_hat

    fig = plt.figure(figsize=(6, 5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    cp = ax.contour(x_seq,y_seq,z)

    ax.clabel(cp, inline=True,
              fontsize=10)
    ax.set_title('Contour Plot')
    ax.set_xlabel('xi')
    ax.set_ylabel('yi')
    plt.show()

def kernel_ridge_regression(x_train,y_train,kernel,lambda_const = 2):

    kk = kernel(x_train,x_train)

    #step1 = lambda_const*np.diag(np.ones(kk.shape[0]))
    #step2 = kk + step1
    #step3 = np.linalg.inv(step2)

    alpha = (kk + lambda_const*np.diag(np.ones(kk.shape[0])))

    #step4 = np.matmul(step3,y_train)

    alpha = np.linalg.inv(alpha)

    alpha = np.matmul(alpha,y_train)

    print('Done running ridge regression')

    return alpha

def kernel_check(X_new, X, Y, alphas,kernel,label1,label2):
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

        label1_dis = np.abs(y_hat - label1)
        label2_dis = np.abs(y_hat - label2)
        guess = label2

        if label1_dis < label2_dis:
            guess = label1

        if guess != Y[i]:  # wrong
            num_wrong += 1
        else:
            num_right += 1

    print('Num wrong = {}'.format(num_wrong))
    print('Num right = {}'.format(num_right))
    print('%Correct = {} %'.format(float(num_right) / (num_wrong + num_right)))

    return num_wrong , num_right

def semiparametric_regression(X,Y,kernel,lamda):

    K = kernel(X,X)

    x2 = K + lamda*np.diag(np.ones(K.shape[0]))

    mat = np.block([[X,x2],[X,K]])

    mat_inv = np.linalg.pinv(mat)

    Y2 = np.block([[Y],[Y]])

    solution = np.matmul(mat_inv,Y2)

    theta_dim = X.shape[1] # X is n x d and theta is d x 1

    theta = solution[:theta_dim] # grab thetas

    alpha = solution[theta_dim:] # alpha solutions are whats left

    return theta,alpha


if __name__ == '__main__':

    run_binary_classification = 0
    if run_binary_classification == 1:

        space = BinarySpace()

        space.create_random_data(neg_data=False,mean=[2,1],covariance=[[0.1,0],[0,0.1]],size=100)

        #space.create_random_data(neg_data=True,mean=[1,2],covariance=[[0.01,0],[0,0.01]],size=100)

        space.create_binary_circle(size=50)

        space.make_matplotlib_plot()

        space.show_plt()

        x_m = np.array([space.neg_x_data, space.neg_y_data]).T

        x_p = np.array([space.pos_x_data, space.pos_y_data]).T

        X_data = np.concatenate((x_p,x_m))

        Y_data = [1 for _ in range(len(x_p))] + [-1 for _ in range(len(x_m))]

        Y_data = np.array(Y_data)

        train_data_indexes = np.random.choice(len(X_data),int(len(X_data)*0.80), replace=False)
        test_data_indexes = [i for i in range(len(X_data)) if i not in train_data_indexes]  # get whats left over for testing

        train_data_x = X_data[train_data_indexes]
        train_data_y = Y_data[train_data_indexes]

        test_data_x = X_data[test_data_indexes]
        test_data_y = Y_data[test_data_indexes]

        kernel_dict = KernelsDict()

        alpha_prime = kernel_ridge_regression(train_data_x,train_data_y,kernel_dict.kernel_k3,lambda_const=0.01)

        kernel_check(test_data_x,train_data_x,test_data_y,alpha_prime,kernel_dict.kernel_k3,1,-1)

        xmin = X_data[:,0].min() - 3
        xmax = X_data[:,0].max() + 3

        ymin = X_data[:,1].min() - 3
        ymax = X_data[:,1].max() + 3

        #contour_lines(xmin,xmax,ymin,ymax,train_data_x,alpha_prime,kernel_dict.kernel_k1,1,-1)
        space.make_contour_plot(xmin,xmax,ymin,ymax,alpha_prime,train_data_x,kernel_dict.kernel_k3,1,-1)

        #space.show_plt()

        print('Done running binary classifier')

    run_mnist_classifier = 0
    if run_mnist_classifier:
        #mnist = fetch_mldata('MNIST original', data_home='/Users/befeltingu/VanillaAI/DataSets/Image/')
        mnist = datasets.load_digits()

        ###########################################################################
        # the dataset contains a set of 150 records under five attributes -
        # X = petal length, petal width, sepal length, sepal width
        # Y (target) = species (setosa,versicolor,virginica )
        ###########################################################################

        mnist_data = mnist.images
        mnist_targets = mnist.target

        mnist_data = mnist_data.reshape((len(mnist_data), -1))

        # first start with setosa versus versicolor
        print('Running kernel classifier on Setosa vs Versicolor')

        combos = itertools.combinations(range(10), 2)

        lambdas = np.linspace(0,2,10)

        lambdas = [0.25]

        tot_accuracy = []

        for lambda_current in lambdas:

            tot_right = 0.0
            tot_wrong = 0.0

            s_time = time.time()

            combos = itertools.combinations(range(10), 2)

            mnist_result_df = pd.DataFrame(data=[],columns=['Image1','Image2','number train','number test','accuracy'])

            for combo in combos:

                print('------------------------')
                print('Runing label {} vs {}'.format(combo[0],combo[1]))

                sub_set_indexes = np.where((mnist_targets == combo[0]) | (mnist_targets == combo[1]))
                mnist_data_x = mnist_data[sub_set_indexes]
                mnist_data_y = mnist_targets[sub_set_indexes]

                n_train = int(np.floor(len(mnist_data_x) * 0.80))
                n_test = len(mnist_data_x) - n_train

                # Random sample 80 data points for training
                # From the first 100 points in the data set grab 80 at random and without replacement
                train_data_indexes = np.random.choice(len(mnist_data_x), n_train, replace=False)
                test_data_indexes = [i for i in range(len(mnist_data_x)) if i not in train_data_indexes]  # get whats left over for testing

                train_data_x = mnist_data_x[train_data_indexes]
                train_data_y = mnist_data_y[train_data_indexes]

                test_data_x = mnist_data_x[test_data_indexes]
                test_data_y = mnist_data_y[test_data_indexes]

                kernel_dict = KernelsDict()

                alpha_prime = kernel_ridge_regression(train_data_x,train_data_y,kernel_dict.kernel_k2,lambda_const=lambda_current)

                num_wrong , num_right = kernel_check(test_data_x,train_data_x,test_data_y,alpha_prime,kernel_dict.kernel_k2,combo[0],combo[1])

                print('------------------------')

                tot_right += num_right

                tot_wrong += num_wrong

                num_train = len(train_data_indexes)

                num_test = len(test_data_indexes)

                row_df = pd.DataFrame(data=[[combo[0],combo[1],num_train,num_test,tot_right / (tot_wrong + tot_right)]],columns=['Image1','Image2','number train','number test','accuracy'])

                mnist_result_df = mnist_result_df.append(row_df)

            e_time = time.time()

            print('MNIST Run time = {}'.format((e_time - s_time) / 60.0))

            print('Total correct = {}'.format(tot_right))

            print('Total wrong = {}'.format(tot_wrong))

            accuracy = tot_right / (tot_right + tot_wrong)

            tot_accuracy.append(accuracy)

            print('Accuracy = {}'.format(accuracy))

            mnist_sub = mnist_result_df[['Image1','Image2','accuracy']]

            #sns.heatmap(mnist_sub,annot=True)

            print('Done running mnist classifier')


        plt.plot(lambdas,tot_accuracy)

        plt.show()

    run_semiparametric_regression = 1
    if run_semiparametric_regression:

        data = pd.read_csv('/Users/befeltingu/VanillaAI/KernelMethods/Basic/hmw3-data1.csv')

        X = data[['x']].as_matrix()
        Y = data[['y']].as_matrix()

        kernels = KernelsDict()

        theta, alpha = semiparametric_regression(X,Y,kernels.kernel_k2,lamda=0.5)

        # now make predictions on data

        predictions = []
        for i in range(X.shape[0]):

            k_mm = 0
            for j in range(X.shape[0]):
                k_mm += alpha[j] * kernels.kernel_k2(X[j], X[i])

            y_hat = X[i]*theta + k_mm

            predictions.append(y_hat[0][0])


        plt.plot([_ for _ in range(len(predictions))],predictions)
        plt.scatter([_ for _ in range(len(predictions))],np.reshape(Y,(len(Y))),c='red')

        plt.show()









