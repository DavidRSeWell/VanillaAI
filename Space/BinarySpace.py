#from scipy.stats import multivariate_normal
from numpy.random import multivariate_normal

from Space.util import get_circle

import matplotlib.pyplot as plt

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

    def make_matplotlib_plot(self):

        colors = ('blue', 'red')
        groups = ('+', '-')

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(self.neg_x_data, self.neg_y_data, color='red', label='-')

        ax.scatter(self.pos_x_data, self.pos_y_data, color='blue', label='+')
        # ax.scatter(x_m_1,x_m_2,color='red',label='-')
        plt.title('Data')

        plt.legend()

        self.plt_object = plt

    def show_plt(self):

        self.plt_object.show()