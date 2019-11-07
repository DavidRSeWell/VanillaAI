#from scipy.stats import multivariate_normal
from numpy.random import multivariate_normal

from Space.util import get_circle

import matplotlib.pyplot as plt
import numpy as np

class BinarySpace(object):
    '''
    Creating and ploting data
    related to binary classification problems
    '''

    def __init__(self):

        self.neg_x_data = None
        self.neg_y_data = None

        self.pos_x_data = None
        self.pos_y_data = None

        self.plt_object = None # variable for housing
        self.plt_contour_plot = None


    def create_binary_circle(self,radius=2,size=100,neg_data = True):

        circle_y, circle_x = get_circle(radius,num_points=size)

        if neg_data:
            self.neg_x_data = circle_x
            self.neg_y_data = circle_y
        else:
            self.pos_x_data = circle_x
            self.pos_y_data = circle_y

    def create_random_data(self,neg_data = True,mean = [-1,1],covariance=[[1,0],[0,1]],size=100):

        #x_p = multivariate_normal(mean,covariance, size=size)

        x_p = multivariate_normal(mean,covariance, size=size)

        x_p_1, x_p_2 = zip(*x_p)

        if neg_data:
            self.neg_x_data = x_p_1
            self.neg_y_data = x_p_2
        else:
            self.pos_x_data = x_p_1
            self.pos_y_data = x_p_2

    def make_contour_plot(self,xmin,xmax,ymin,ymax,alphas,X,kernel,label1,label2):

        x_seq = np.linspace(xmin, xmax, 20)

        y_seq = np.linspace(ymin, ymax, 20)

        X1, X2 = np.meshgrid(x_seq, y_seq)

        z = np.zeros((len(x_seq), len(x_seq)))

        for x_i in range(len(x_seq)):
            for x_j in range(len(y_seq)):

                x = np.array([x_seq[x_i], y_seq[x_j]])

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

        cp = ax.contour(x_seq, y_seq, z,zorder=-1)

        ax.clabel(cp, inline=True,
                  fontsize=10)
        ax.set_title('Contour Plot')
        ax.set_xlabel('xi')
        ax.set_ylabel('yi')

        plt.show()




    def make_matplotlib_plot(self):

        colors = ('blue', 'red')
        groups = ('+', '-')

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(self.neg_x_data, self.neg_y_data, color='red', label='-',zorder=1)

        ax.scatter(self.pos_x_data, self.pos_y_data, color='blue', label='+',zorder=1)
        # ax.scatter(x_m_1,x_m_2,color='red',label='-')
        plt.title('Data')

        plt.legend()

        self.plt_object = fig

    def show_plt(self):

        plt.show()
        #self.plt_object.show()