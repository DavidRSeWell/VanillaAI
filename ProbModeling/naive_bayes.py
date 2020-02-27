import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from scipy.stats import norm


def create_spam_data(spam_file):
    '''
    Helper function to clean up the
    spam data into a more readable format
    '''

    spam_mat = np.array([])
    for i, row in enumerate(spam_file):

        row_i = np.array(row.split(','))
        row_i[-1] = int(row_i[-1])
        row_i = row_i.reshape((1, row_i.shape[0]))

        if i == 0:
            spam_mat = row_i.copy()

        else:
            spam_mat = np.append(spam_mat, row_i, axis=0)

    return spam_mat.astype(np.float)


class BinaryGaussianNB:

    def __init__(self,X,min_std=0.0001):

        self.X = X # including target value

        self.class_positive_dist = [] # list of gauss for each feature
        self.class_negative_dist = [] # list of gauss for each feature
        self.min_std = min_std # dont allow any of the feature distributions to have std below this
        self.pos_prior = None
        self.neg_prior = None
        self.test_x = None
        self.test_y = None
        self.train_x = None
        self.train_y = None

    def calculate_priors(self):
        '''
        Calculate the prob distributions
        for each one of the classes
        :return:
        '''
        positive_indexes = np.where(self.train_y == 1.0)[0]
        negative_indexes = np.where(self.train_y == 0.0)[0]

        self.pos_prior = len(positive_indexes) / len(self.train_y)
        self.neg_prior = len(negative_indexes) / len(self.train_y)


    def create_feature_dist(self):
        '''
        Create a list of distributions
        one for each feature / class in
        the training set
        :return:
        '''

        positive_indexes = np.where(self.train_y == 1.0)[0]
        negative_indexes = np.where(self.train_y == 0.0)[0]
        gauss_positive_list = []
        gauss_negative_list = []
        for j in range(self.train_x.shape[1]):

            # first look at the data points with a positive label
            j_pos_mean = self.train_x[positive_indexes,j].mean() # mean for the jth feature in the training set
            j_pos_std = self.train_x[positive_indexes,j].std() # std for the jth feature in the training set
            if j_pos_std < self.min_std:
                j_pos_std = self.min_std

            norm_j_pos = norm(loc=j_pos_mean,scale=j_pos_std)
            gauss_positive_list.append(norm_j_pos)

            # first look at the data points with a positive label
            j_neg_mean = self.train_x[negative_indexes, j].mean()  # mean for the jth feature in the training set
            j_neg_std = self.train_x[negative_indexes, j].std()  # std for the jth feature in the training set

            if j_neg_std < self.min_std:
                j_neg_std = self.min_std

            norm_j_neg = norm(loc=j_neg_mean, scale=j_neg_std)
            gauss_negative_list.append(norm_j_neg)

        self.class_positive_dist = gauss_positive_list
        self.class_negative_dist = gauss_negative_list

    def create_tt_set(self,per_train=0.50):
        '''
        Helper function to split train
        and test set.
        :param per_train: Percentage of data to use for training
        '''
        N = self.X.shape[0]
        num_train = int(N * per_train)
        x_indexes = [i for i in range(N)]
        train_indexes = np.random.choice(x_indexes, num_train, replace=False)
        test_indexes = [i for i in x_indexes if i not in train_indexes]

        X_train, y_train = self.X[train_indexes,:-1], self.X[train_indexes,-1]

        X_test, y_test = self.X[test_indexes,:-1], self.X[test_indexes,-1]

        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

    def predict(self,x):
        '''
        Make prediction on the point x
        using log of the likelihood
        :param x: Data point
        :return:
        '''

        # first get positive prediction
        pos_predict = np.log(self.pos_prior) # start from log of prior
        neg_predict = np.log(self.neg_prior)
        M = self.train_x.shape[1] # num features

        for j in range(M):

            pos_prob = self.class_positive_dist[j].pdf(x[j])
            neg_prob = self.class_negative_dist[j].pdf(x[j])

            pos_predict += np.log( pos_prob)
            neg_predict += np.log( neg_prob)

        if pos_predict >= neg_predict:
            return 1
        else:
            return 0


    def run(self):
        '''
        Algorithm:
            With the training set:
                For each feature, class: calc mean, variance and create gauss from that
            With the test set:
                for each x in test set:
                    calc arg max P(class) product P(x_i|class)

        '''

        # create distributions for each feature, class
        self.create_feature_dist()

        # calculate the class distributions i.e p(spam) p(not spam)
        self.calculate_priors()

        N = self.test_y.shape[0] # number of test data points

        tp = 0 # true positive
        tn = 0 # true negative
        fp = 0 # false positive
        fn = 0 # false negative

        for i , x in enumerate(self.test_x):

            y_hat = int(self.predict(x))

            y = int(self.test_y[i])

            diff = y_hat - y

            if diff == 0:
                if y == 0:
                    tp += 1
                else:
                    tn += 1
            elif diff == 1:
                fp += 1

            elif diff == -1:
                fn += 1


        accuracy = (tp + tn) / N
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        print('accuracy = {}'.format(accuracy))
        print('precision = {}'.format(precision))
        print('recall = {}'.format(recall))

        confusion_mat = np.array([[tn,fp],[fn,tp]])

        conf_mat_df = pd.DataFrame(confusion_mat)

        #ax = sns.heatmap(conf_mat_df, annot=True)

        print(conf_mat_df)

        plt.ylabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()



if __name__ == '__main__':

    ###################################
    # PUT PATH TO FILES FROM UBI SITE
    ###################################
    data_path = '/Users/befeltingu/PSUClasses/ML/spambase/spambase.data'
    names_path = '/Users/befeltingu/PSUClasses/ML/spambase/spambase.names'

    spam_file = open(data_path, 'r').readlines()
    feature_names = open(names_path, 'r').readlines()

    spam_mat = create_spam_data(spam_file)

    gauss_nb = BinaryGaussianNB(spam_mat)

    gauss_nb.create_tt_set()

    gauss_nb.run()