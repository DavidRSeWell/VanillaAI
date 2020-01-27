import numpy as np
import matplotlib.pyplot as plt

from matplotlib import offsetbox
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn import datasets

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

    K = (np.dot(X,X.T))**8

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    alphas = eigvecs / np.sqrt(eigvals)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((alphas[:, -i] for i in range(1, n_components + 1)))

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


if __name__ == '__main__':

    run_circles_toy_demo = 0
    if run_circles_toy_demo:

        from sklearn.datasets import make_circles

        circle_X, circle_y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

        plt.figure(figsize=(8, 6))

        plt.scatter(circle_X[circle_y == 0, 0], circle_X[circle_y == 0, 1], color='red', alpha=0.5)
        plt.scatter(circle_X[circle_y == 1, 0], circle_X[circle_y == 1, 1], color='blue', alpha=0.5)
        plt.title('Concentric circles')
        plt.ylabel('y coordinate')
        plt.xlabel('x coordinate')
        plt.show()

        print('Running Kernel PCA on example')

        kpca = david_kpca(circle_X,gamma=0.5,n_components=2)

        plt.figure(figsize=(8, 6))
        plt.scatter(kpca[circle_y == 0, 0], kpca[circle_y == 0, 1], color='red', alpha=0.5)
        plt.scatter(kpca[circle_y == 1, 0], kpca[circle_y == 1, 1], color='blue', alpha=0.5)

        # plt.text(-0.48, 0.35, 'gamma = 15', fontsize=12)
        plt.title('First 2 principal components after RBF Kernel PCA via scikit-learn')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

        print('Done running kernel PCA')

    run_mnist_example = 0
    if run_mnist_example:

        digits = datasets.load_digits()

        X = digits.data

        y = digits.target

        n_samples, n_features = X.shape

        mnist_kpca = david_kpca(X,1,2)

        embedding_plot(mnist_kpca,'MNIST KPCA')

        plt.show()

        print('Done running the KPCA MNIST example')


