#coding: utf-8

from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from regularizer import activity_acol
from acolpooling import AcolPooling

def define_model(input_shape, nb_classes, acol_params, truncated=False):
    '''
    返回定义好的模型
    '''
    model = Sequential()
    '''
    前L-2层
    '''
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    '''
    在F=Y^(L-2)上做聚类
    '''
    if not truncated:
        # 对第L-2层的输出做acol regularizer
        ks, c1, c2, c3, c4 = acol_params
        '''
        L-1层的logit，并对其做acol
        '''
        model.add(Dense(nb_classes*ks, activity_regularizer=activity_acol(c1, c2, c3, c4, ks), name='L-1'))
        '''
        Z = acol(Y^(L-2)*W^(L-1)+b^(L-1))
        Z是softmax的输入，是L-1 layer的输出
        L-1层的softmax
        '''
        model.add(Activation('softmax', name='L-1_activation'))
        '''
        L层，linear层，将同一个parent的softmax nodes相加
        '''
        model.add(AcolPooling(nb_classes, name='AcolPooling'))
    return model

