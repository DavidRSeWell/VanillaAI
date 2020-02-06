import numpy as np
import matplotlib.pyplot as plt
import os

from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
from matplotlib import offsetbox
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn import datasets


def stepwise_kpca(X, gamma, n_components):
    """
    Implementation of a RBF kernel PCA.

    Arguments:
        X: A MxN dataset as NumPy array where the samples are stored as rows (M),
           and the attributes defined as columns (N).
        gamma: A free parameter (coefficient) for the RBF kernel.
        n_components: The number of components to be returned.

    """
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = np.exp(-gamma * mat_sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc

def david_kpca(X, gamma, n_components):
    """
    Implementation of a RBF kernel PCA.

    """
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    #sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    #mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    #K = np.exp(-gamma * mat_sq_dists)
    N = X.shape[0]
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            K[i][j] = (np.dot(X[i],X[j]))**4
            #K[i][j] = np.exp(np.dot(-1.0*X[i],X[j]) / gamma)
    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]

    one_n = np.ones((N, N)) / N

    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    alphas = eigvecs / np.sqrt(eigvals)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc

def embedding_plot(X, title):

    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(X[:, 0], X[:, 1], lw=0, s=40, c=y / 10.)

    shown_images = np.array([[1., 1.]])
    for i in range(X.shape[0]):
        if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < 1e-2: continue
        shown_images = np.r_[shown_images, [X[i]]]

        ax.add_artist(offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]))

    plt.xticks([]), plt.yticks([])
    plt.title(title)

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

if __name__ == '__main__':


    run_moon_demo = 0
    if run_moon_demo:

        from sklearn.datasets import make_moons

        X, y = make_moons(n_samples=100, random_state=123)

        plt.figure(figsize=(8, 6))

        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', alpha=0.5)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', alpha=0.5)

        plt.title('A nonlinear 2Ddataset')
        plt.ylabel('y coordinate')
        plt.xlabel('x coordinate')

        plt.show()

        X_pc = david_kpca(X, gamma=15, n_components=2)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pc[y == 0, 0], X_pc[y == 0, 1], color='red', alpha=0.5)
        plt.scatter(X_pc[y == 1, 0], X_pc[y == 1, 1], color='blue', alpha=0.5)

        plt.title('First 2 principal components after RBF Kernel PCA')
        plt.text(-0.18, 0.18, 'gamma = 15', fontsize=12)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

    run_mnist_example = 1
    if run_mnist_example:

#        digits = datasets.load_digits()

        fetch_mnist()
        from sklearn.datasets import fetch_mldata

        mnist = fetch_mldata("MNIST original")

        ALPHA = 0.1

        X = mnist['data'].copy()
        X = X.reshape((len(X), -1))

        X = X / 255.0

        y = mnist['target'].copy()

        #X = digits.data

        #y = digits.target

        digit_indexes = np.where((y== 1) | (y == 7))

        #sub_x = X[digit_indexes]

        #sub_y = y[digit_indexes]

        sub_indexes = np.random.choice(digit_indexes[0], 500, replace=False)

        sub_x = X[sub_indexes]

        sub_y = y[sub_indexes]

        n_samples, n_features = X.shape

        mnist_kpca = david_kpca(sub_x,1,8)
        #mnist_kpca = stepwise_kpca(X,60,2)
        #embedding_plot(mnist_kpca,'MNIST KPCA')

        for i in range(0,4):

            digit_1_1 = mnist_kpca[[sub_y == 1 ,2*i]]
            digit_1_2 = mnist_kpca[[sub_y == 1 ,2*i + 1]]
            digit_7_1 = mnist_kpca[[sub_y == 7, 2*i]]
            digit_7_2 = mnist_kpca[[sub_y == 7, 2*i + 1]]

            #x_1 = x_1[np.where((x_1 > -0.0001) & (x_1 < 0.0001))]
            #x_2 = x_2[np.where((x_2 > -0.0001) & (x_2 < 0.0001))]

            plt.scatter(digit_1_1,digit_1_2,color='red')
            plt.scatter(digit_7_1,digit_7_2,edgecolors='blue')
            plt.xlabel('PC{}'.format(str(2*i + 1)))
            plt.ylabel('PC{}'.format(str(2*i + 2)))
            plt.show()

        print('Done running the KPCA MNIST example')


