#coding: utf-8

'''
获取第L-1层的logit
'''
from keras.models import Model
import numpy as np 

def get_Z(define_model, model_params, optimizer, weight_path, latent_save_path, train_X):
    '''
    Z = Y^(L-2)*W^(L-1) + b^(L-1)
    '''
    model = define_model(*model_params)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    model.load_weights(weight_path)
    Z_model = Model(inputs=model.input,
                                    outputs=model.get_layer('L-1_softmax').input)
    latent_X = Z_model.predict(train_X)
    np.save(latent_save_path, latent_X)
    # return latent_X