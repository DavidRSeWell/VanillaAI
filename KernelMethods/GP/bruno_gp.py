import matplotlib.pyplot as plt
import numpy as np


def k_1(x1,x2):
    return np.dot(x1,x2)

def k_p(x1,x2,p=5):
    return k_1(x1,x2)**p

def k_gauss(x1,x2,sigma=4):
    return np.exp(-(x1-x2)**2/(2*sigma))

def kernel_mat(k, X,Y):
    '''
    allowing for N x M kernel for convience in GP problem
    :param k:
    :param X:
    :param Y:
    :return:
    '''

    N = X.shape[0]
    M = Y.shape[0]
    K = np.zeros((N, M))
    for i in range(0, N):
        for j in range(0, M):
            v = k(X[i], Y[j])
            K[i,j] = v
    return np.matrix(K)


def sample_gauss():
    '''
    Gauss likes to be sampled
    :return:
    '''



if __name__ == '__main__':

    print('Running brunos GP script in python')
    N = 5
    M = 300 # for prediction
    sigma2 = 2**2
    sigma2_obs = 0.5**2
    X = np.linspace(1,10,N)
    X_star = np.linspace(1,10,M)

    # K(x*,x*)
    K_s_s = kernel_mat(k_gauss,X_star,X_star)
    # K(x*,x)
    K_s_x = kernel_mat(k_gauss,X_star,X)
    # K(x,x)
    K_x_x = kernel_mat(k_gauss,X,X)

    noise = sigma2_obs*np.eye(N)
    # sample observations
    f = np.random.multivariate_normal(np.zeros((N,)), K_x_x, 1)
    y = np.random.multivariate_normal(f.reshape((N,)),noise)

    y = np.matrix(y.reshape((N,1)))

    K_x_x_inv = np.linalg.inv(K_x_x + noise)

    mean = K_s_x*K_x_x_inv*y
    cov = K_s_s - K_s_x*K_x_x_inv*K_s_x.T

    mean = np.array(mean)
    cov = np.array(cov)

    upper = mean + 2*np.diag(cov).reshape((mean.shape[0],1))
    lower = mean - 2*np.diag(cov).reshape((mean.shape[0],1))

    plt.scatter(X,np.reshape(np.array(y),(N,)))

    plt.plot(X_star,mean,color='red')
    plt.plot(X_star,upper,color='blue')
    plt.plot(X_star,lower,color='blue')
    plt.show()
    print('Done running GP ')
