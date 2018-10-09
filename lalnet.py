#coding: utf-8

from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from regularizer import activity_acol
from acolpooling import AcolPooling
import tensorflow as tf

def define_cnn(input_shape, nb_classes, acol_params):
    '''
    return cnn model
    '''
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Convolution2D(32, (3, 3), activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, (3, 3), activation='relu', border_mode='same'))
    model.add(Convolution2D(64, (3, 3), activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu', name='L-2'))
    model.add(Dropout(0.5))
    ks, c1, c2, c3, c4 = acol_params
    model.add(Dense(nb_classes*ks, activity_regularizer=activity_acol(c1, c2, c3, c4, ks), name='L-1'))
    '''
    Z = acol(Y^(L-2)*W^(L-1)+b^(L-1))
    Z is L-1_softmax layer's input 
    do cluster on Z
    '''
    # model.add(Activation(activation='softmax', name='L-1_activation'))
    model.add(Activation(tf.nn.softmax, name='L-1_softmax'))
    '''
    L-th layer is Linear layer to add softmax nodes of one parent
    '''
    model.add(AcolPooling(nb_classes, name='AcolPooling'))
    return model

