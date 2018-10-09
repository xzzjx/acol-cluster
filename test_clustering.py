#coding: utf-8
'''
test cluster
'''
from __future__ import division, print_function, unicode_literals
from sklearn.cluster import KMeans
import numpy as np
import argparse
import datasets
from keras.datasets import mnist

def cluster(X):
    '''
    cluster on X
    '''
    est = KMeans(n_clusters=10)
    y_pred = est.fit_predict(X)
    return y_pred

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    # D = max(y_pred.max(), y_true.max()) + 1
    # w = np.zeros((D, D), dtype=np.int64)
    # for i in range(y_pred.size):
    #     w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    # ind = linear_assignment(w.max() - w)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    def assign_label(y_true, y_pred):
        '''
        assign label to cluster
        '''
        correct_label = np.zeros_like(y_pred, dtype=np.int32)
        for cluster in range(y_pred.max()+1):
            label = np.bincount(y_true[y_pred==cluster]).argmax()
            correct_label[y_pred == cluster] = label
        return correct_label
    
    y_pred = assign_label(y_true, y_pred)
    # print(y_pred[:10])
    # print(y_true[:10])
    # compute ACC
    acc = np.sum(y_pred == y_true) / y_true.shape[0]
    print("acol cluster accuracy is", acc)
    return acc

def cluster_and_acc(train_X, train_y):
    y_pred = cluster(train_X)
    print(y_pred.shape)

    accuracy = acc(train_y, y_pred)
    print("acol cluster accuracy is", accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='svhn')
    args = parser.parse_args()
    if args.dataset == 'svhn':
        train_label, test_label, _ = datasets.load_svhn_labels()
        data_path = './latents/svhn_trainX.npy'
        train_X = np.load(data_path)
    elif args.dataset == 'mnist':
        (train_X, train_label), (test_X, test_label) = mnist.load_data()
        data_path = './latents/mnist_trainX.npy'
        train_X = np.load(data_path)
    elif args.dataset == 'mnist_original':
        (train_X, train_label), (test_X, test_label) = mnist.load_data()
        train_X = train_X.reshape([-1, train_X.shape[1]*train_X.shape[2]])
        print(train_X.shape)
    y_pred = cluster(train_X)
    print(y_pred.shape)

    accuracy = acc(train_label, y_pred)
    print("acol cluster accuracy is", accuracy)