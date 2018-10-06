#coding: utf-8
'''
定义acol regularizer
'''

from keras.regularizers import Regularizer
from keras import backend as K 
def activity_acol(c1, c2, c3, c4, ks):
    return AcolRegularizer(c1, c2, c3, c4, ks)
class AcolRegularizer(Regularizer):
    '''
    定义Acol Regularizer
    https://github.com/keras-team/keras/blob/master/keras/regularizers.py
    '''
    def __init__(self, c1, c2, c3, c4, ks):
        self.c1 = K.variable(c1)
        self.c2 = K.variable(c2)
        self.c3 = K.variable(c3)
        self.c4 = K.variable(c4)
        self.ks = ks

    def __call__(self, x):
        '''
        x是层的输出
        '''
        Z = x # shape=(batch_size, nb_classes*ks)
        n = K.int_shape(Z)[1]
        B = K.reshape(Z*k.cast(Z>0., K.floatx()), (-1, self.ks, n//self.ks)) # B.shape=(batch_size, ks, nb_classes)

        # N = B.T*B, shape= n*n, n = nb_classes*ks
        # GAR项迫使矩阵N变为identity matrix
        # affinity惩罚矩阵N的非对角线元素，尽量使得一个样本分配只到一个softmax node的概率为1
        # balance项尽量使得矩阵N的对角线元素相等，避免最后塌缩到一个小于n的空间

        '''
        affinity: ks-1项intra-parent
        '''
        N = K.tf.tensordot(B, B, axes=[0, 0]) # N.shape=ks*np*ks*np 相当于B.T*B
        def calculate_partial_affinity_balance(i):
            N_partial = N[:, i, :, i]
            v = K.tf.linalg.diag_part(N_partial)
            V = K.dot(v.T, v)
            affinity = (K.sum(N_partial) - K.tf.trace(N_partial)) / ((self.ks-1)*K.tf.trace(N_partial)+K.epsilon())
            balance = (K.sum(V) - K.tf.trace(V)) / ((self.ks-1)*K.tf.trace(V)+K.epsilon())
            return affinity, balance
        partials = K.tf.scan(calculate_partial_affinity_balance, [K.tf.range(N.shape[1])])
        # def cond(i, np, k, N, affinities, balances):
        #     return K.tf.less(i, np)
        # def body(i, np, k, N, affinities, balances):
        #     N_partial = N[:, i, :, i] # shape=ks*ks
        #     # v = K.sum(N_partial, axis=0).reshape((1, np))
        #     v = K.tf.diagpart(N_partial).reshape((1, self.ks))
        #     V = K.dot(v.T, v)
        #     affinity = (K.sum(N_partial) - K.tf.trace(N_partial)) / ((self.ks-1)*K.tf.trace(N_partial)+K.epsilon())
        #     balance = (K.sum(V) - K.tf.trace(V)) / ((self.ks-1)*K.tf.trace(V)+K.epsilon())
        #     affinities = tf.concat(affinities, affinity)
        # i = K.tf.constant(1, dtype=K.tf.int32)
        # np = K.tf.constant(self.nb_classes, dtype=K.tf.int32)
        # k = K.tf.constant(self.ks, dtype=K.tf.int32)
        # partials = tf.while_loop(cond, body, [i, np, k, N, affinities, balances])
        affinity = K.means(partials[0])
        balance = K.means(partials[1])
        regularization = K.variable(0, dtype=K.floatx())
        if self.c1.get_value():
            regularization += self.c1*affinity
        if self.c2.get_value():
            regularization += self.c2*(1-balance)
        if self.c3.get_value():
            regularization += self.c3*balance
        if self.c4.get_value():
            regularization += K.sum(self.c4*K.square(Z))
        self.affinity = affinity
        self.balance = balance
        self.reg = regularization 
        return regularization 
    def get_config(self):
        return {'name': self.__class__.__name__,
                    'c1': self.c1.get_value(),
                    'c2': self.c2.get_value(),
                    'c3': self.c3.get_value(),
                    'c4': self.c4.get_value()}





