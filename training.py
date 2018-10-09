#coding: utf-8

'''
preprocessing data, train model, save model
'''
from __future__ import division, print_function, unicode_literals
import numpy as np 
import time 
from keras.utils.np_utils import to_categorical
from test_clustering import cluster_and_acc, cluster, acc
from keras.models import Model
# from lalnet import define_cnn
# from keras.models import load_model

def train_with_pseudos(nb_pseudos, nb_clusters_per_pseudo, define_model, model_params, optimizer,
                                        X_train, y_train,
                                        get_pseudos,
                                        nb_epochs, batch_size,
                                        validation_set_size=None,
                                        test_on_test_set=True, return_model=False,
                                        save_path=None):
    '''
    暂时没有test
    '''
    # nb_all_clusters = nb_clusters_per_pseudo * nb_pseudos
    
    # nb_classes = len(np.unique(y_train))

    run_start = time.time()
    # print(*model_params)
    model = define_model(*model_params)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    # history = model.fit_pseudo(X_train, nb_pseudos)
    steps_per_epoch = X_train.shape[0] // batch_size
    n_dpoints = 20
    latent_model = Model(inputs=model.input, outputs=model.get_layer('L-1').output)
    for i in range(n_dpoints):
        epochs_per_dpoints = nb_epochs // n_dpoints
        # history = model.fit_generator(pseudo_generator(get_pseudos, X_train, nb_pseudos, batch_size),
                                                    #  steps_per_epoch, nb_epochs)
        history = model.fit_generator(pseudo_generator(get_pseudos, X_train, nb_pseudos, batch_size),
                                                     steps_per_epoch, epochs_per_dpoints)
        latent_model.set_weights(model.get_weights())
        latent_X = latent_model.predict(X_train)
        y_pred = cluster(latent_X)
        # y_pred = latent_model.predict_classes(X_train)
        acc(y_train, y_pred)
        # cluster_and_acc(latent_X, y_train)
        

    model.save_weights(save_path)
    run_end = time.time()
    print("total time is ", (run_end-run_start))
    print(history)


def pseudo_generator(get_pseudos, X_train, nb_classes, batch_size):
    count = 0
    list_x = []
    list_y = []
    # print(nb_classes)
    while True:
        for x in X_train:
            for y in range(nb_classes):
                px = get_pseudos(x, y)
                count += 1
                # if px is None:
                #     print(y)
                # print(px.shape)
                list_x.append(px)
                list_y.append(y)
                if count >= batch_size:
                    yield (np.array(list_x), to_categorical(np.array(list_y), nb_classes))
                    count = 0
                    list_x = []
                    list_y = []
