#coding: utf-8

'''
主测试
'''

import training
from lalnet import define_cnn
from keras.optimizers import SGD
from keras.datasets import mnist
from transform import get_pseudos

nb_pseudos = 8
nb_clusters_per_pseudo = 20
c1 = 0.1
c2 = 1.0
c3 = 0.0
c4 = 0.000001

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)

'''
mnist
'''
(train_X, train_y), (test_X, test_y) = mnist.load_data()
input_shape = (train_X[0].shape[0], train_X.shape[1], 1) 
train_X = train_X.reshape([-1, input_shape[0], input_shape[1], input_shape[2]])

nb_epochs = 200
# nb_dpoints = 40
batch_size = 128
print(input_shape)
training.train_with_pseudos(nb_pseudos, nb_clusters_per_pseudo,
                                            define_cnn, (input_shape, nb_pseudos, (nb_clusters_per_pseudo,c1,c2,c3,c4)), 
                                            sgd, train_X, train_y, get_pseudos, nb_epochs, batch_size)
