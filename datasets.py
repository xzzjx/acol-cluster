from __future__ import division
import numpy as np

def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        for did in did_to_cat.keys():
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did_to_cat.has_key(did):
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000]
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print 'todense succeed'

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print 'permutation finished'

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], x.size / x.shape[0]))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 50.)  # normalize as it does in DEC paper
    print 'MNIST samples', x.shape
    return x, y, x_train.shape[0]


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        if not os.path.exists(data_path+'/usps_train.jf.gz'):
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
            os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)
        os.system('gunzip %s/usps_train.jf.gz' % data_path)
        os.system('gunzip %s/usps_test.jf.gz' % data_path)

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [map(float, line.split()) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [map(float, line.split()) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64')
    y = np.concatenate((labels_train, labels_test))
    print 'USPS samples', x.shape
    return x, y


def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print 'making reuters idf features'
        make_reuters_data(data_path)
        print 'reutersidf saved to ' + data_path
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy')).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], x.size / x.shape[0])).astype('float64')
    y = y.reshape((y.size,))
    print 'REUTERSIDF10K samples', x.shape
    return x, y

def image_whitening(data):
    import math
    assert len(np.shape(data)) == 4

    nb_pixels = np.shape(data)[1] * np.shape(data)[2] * np.shape(data)[3]

    mean = np.mean(data, axis=(1,2,3))

    ones = np.ones(np.shape(data)[1:4], dtype=np.float32)
    for i in xrange(len(data)):
        data[i, :, :, :] -= mean[i] *ones
    
    adj_std_var = np.maximum(np.ones(len(data), dtype=np.float32) / math.sqrt(nb_pixels), np.std(data, axis=(1,2,3)))

    for i in xrange(len(data)):
        data[i, :, :, :] = data[i, :, :, :] / adj_std_var[i]
    
    return data
def extract_svhn(data_path):
    from scipy.io import loadmat as loadmat
    with open(data_path, 'r') as file_obj:
        dict = loadmat(file_obj)
        data, labels = dict['X'], dict['y']

        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)

        data = data.transpose(3, 0, 1, 2)
        labels[labels == 10] = 0

        labels = labels.reshape(len(labels))

        return data, labels
def load_svhn(data_path='../DEC-keras-master/data/svhn/'):
    '''
    for reason of memory, not load ext data
    '''
    # get data
    file_names = ['train_32x32.mat', 'test_32x32.mat', 'extra_32x32.mat']
    train_data, train_labels = extract_svhn(data_path + file_names[0])
    test_data, test_labels = extract_svhn(data_path + file_names[1])
    # ext_data, ext_labels = extract_svhn(data_path + file_names[2])
    train_data = image_whitening(train_data)
    test_data = image_whitening(test_data)
    # ext_data = image_whitening(ext_data)

    # data = np.concatenate((train_data, ext_data, test_data))
    data = np.concatenate((train_data, test_data))
    data = data.reshape(data.shape[0], -1)
    # labels = np.concatenate((train_labels, ext_labels, test_labels))
    labels = np.concatenate((train_labels, test_labels))
    return data, labels, train_data.shape[0], test_data.shape[0]
