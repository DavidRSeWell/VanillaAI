#############################
# Module for writing ridge regression
# as a kernel
#############################
from Space.BinarySpace import BinarySpace
from KernelMethods.Kernels import KernelsDict

import numpy as np

def kernel_ridge_regression(x_train,y_train,kernel,lambda_const = 2):

    kk = kernel(x_train,x_train)

    step1 = lambda_const*np.diag(np.ones(kk.shape[0]))
    step2 = kk + step1
    step3 = np.linalg.inv(step2)

    alpha = (kk + lambda_const*np.diag(np.ones(kk.shape[0])))

    step4 = np.matmul(step3,y_train)


    zeros = np.where(alpha == 0)[0].shape
    print('zeros {}'.format(zeros))
    alpha = np.linalg.inv(alpha)

    alpha = np.matmul(alpha,y_train)

    print('Done running ridge regression')

    return alpha



def kernel_check(X_new, X, Y, alphas,kernel):
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


    space = BinarySpace()

    space.create_random_data(neg_data=False,mean=[0,0],covariance=[[0.01,0],[0,0.01]],size=100)

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

    alpha_prime = kernel_ridge_regression(train_data_x,train_data_y,kernel_dict.kernel_k2,lambda_const=0.01)

    kernel_check(train_data_x,train_data_x,train_data_y,alpha_prime,kernel_dict.kernel_k2)
