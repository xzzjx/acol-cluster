#coding: utf-8

'''
实现ACOL Pooling
'''

from __future__ import division, print_function, unicode_literals
from keras.layers.core import Layer
from keras import backend as K
import numpy as np

def identity_vstack(shape):
        a = np.identity(shape[1])
        for i in range(1, shape[0]//shape[1]):
            a = np.concatenate((a, np.identity(shape[1])), axis=0)
        return K.variable(a)

class AcolPooling(Layer):
    '''
    将同一个parent的softmax node相加
    '''
    def __init__(self, units, name='AcolPooling', **kwargs):
        super(AcolPooling, self).__init__(**kwargs)
        self.units = units 
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                                        initializer=identity_vstack,
                                                        name='kernel', trainable=False)
        self.built = True
    
    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    

        