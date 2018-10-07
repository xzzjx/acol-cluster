#coding: utf-8

'''
preprocessing data, train model, save model
'''
from __future__ import division, print_function, unicode_literals
import numpy as np 
import time 
from lalnet import define_model
from keras.models import load_model

def train_with_pseudos(nb_pseudos, nb_clusters_per_pseudo, define_model, model_params, optimizer,
                                        X_train, y_train,
                                        get_pseudos,
                                        nb_epochs, nb_dpoints, batch_size,
                                        validation_set_size=None,
                                        test_on_test_set=True, return_model=False,
                                        save_path=None):
    '''
    暂时没有test
    '''
    # nb_all_clusters = nb_clusters_per_pseudo * nb_pseudos
    
    nb_classes = len(np.unique(y_train))

    run_start = time.time()

    model = define_model(*model_params)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    # history = model.fit_pseudo(X_train, nb_pseudos)
    steps_per_epoch = X_train.shape[0] / batch_size
    history = model.fit_generator(pseudo_generator(get_pseudos, X_train, nb_classes, batch_size),
                                                     steps_per_epoch, nb_epochs)

    model.save_weights(save_path)
    run_end = time.time()
    print("total time is ", (run_end-run_start))
    print(history)


def pseudo_generator(get_pseudos, X_train, nb_classes, batch_size):
    count = 0
    list_x = []
    list_y = []
    for x in X_train:
        for y in range(nb_classes):
            px = get_pseudos(x, y)
            count += 1
            list_x.append(px)
            list_y.append(y)
            if count >= batch_size:
                yield (np.array(list_x), np.array(list_y))
                count = 0
                list_x = []
                list_y = []