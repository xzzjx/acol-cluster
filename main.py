#coding: utf-8

'''
主测试
'''

import training
from lalnet import define_cnn
from keras.optimizers import SGD
from keras.datasets import mnist
from transform import get_pseudos
from get_representation import get_Z
import os
import datasets
import argparse



def preprocess_data(dataset):
    if dataset == 'mnist':
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        input_shape = (train_X[0].shape[0], train_X.shape[1], 1) 
        train_X = train_X.reshape([-1, input_shape[0], input_shape[1], input_shape[2]])
        test_X = test_X.reshape([-1, input_shape[0], input_shape[1], input_shape[2]])
    elif dataset == 'svhn':
        # data=np.concatenate((train_data, ext_data, test_data))
        data, label, n_train, n_test = datasets.load_svhn()
        train_X = data[:n_train]
        test_X = data[-n_test:]
        train_y = label[:n_train]
        test_y = label[-n_test:]
        input_shape = (32, 32, 3)
        train_X = train_X.reshape([-1, input_shape[0], input_shape[1], input_shape[2]])
        test_X = test_X.reshape([-1, input_shape[0], input_shape[1], input_shape[2]])
        
    print(input_shape)
    return input_shape, train_X, test_X, train_y, test_y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist')
    args = parser.parse_args()
    dataset = args.dataset
    input_shape, train_X, test_X, train_y, test_y = preprocess_data(dataset)

    nb_pseudos = 8
    nb_clusters_per_pseudo = 20
    c1 = 0.1
    c2 = 1.0
    c3 = 0.0
    c4 = 0.000001

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)

    nb_epochs = 5
    # nb_dpoints = 40
    batch_size = 128

    weights_dir = './weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    latents_dir = './latents'
    if not os.path.exists(latents_dir):
        os.makedirs(latents_dir)
    
    
    model_params = (input_shape, nb_pseudos, (nb_clusters_per_pseudo,c1,c2,c3,c4))
    weight_save_path = weights_dir+'/'+dataset+'_weights.h5'
    latent_trainX_save_path = latents_dir + '/'+ dataset + '_trainX.npy'
    latent_testX_save_path = latents_dir + '/' + dataset + '_testX.npy'
    

    training.train_with_pseudos(nb_pseudos, nb_clusters_per_pseudo,
                                                define_cnn, model_params, 
                                                sgd, train_X, train_y, get_pseudos, nb_epochs, batch_size, save_path=weight_save_path)

    
    get_Z(define_cnn, model_params, sgd, weight_save_path, latent_trainX_save_path, train_X)
    get_Z(define_cnn, model_params, sgd, weight_save_path, latent_testX_save_path, test_X)