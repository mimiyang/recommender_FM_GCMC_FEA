import json
import os
import numpy as np
import h5py
import math
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from pyfm import pylibfm

def load_dataset(data_dir, dataset_name):

    X = []
    y = []
    train_X = []
    train_y = []
    test_X = []
    test_y = []

    if dataset_name == 'ml-1m':
        file_path = os.path.join(data_dir, dataset_name, 'ratings.dat')
        with open(file_path, 'r') as f:
            for line in f.readlines():
                uid, mid, rating, timestamp = line.split('::')
                X.append({'user' : uid, 'item' : mid})
                y.append(float(rating))
        y = np.array(y)
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.1)
        
    elif dataset_name == 'douban' or dataset_name == 'yahoo_music':
        file_path = os.path.join(data_dir, dataset_name, 'training_test_dataset.mat')
        db = h5py.File(file_path, 'r')
        
        M = np.asarray(db['M']).astype(np.float32).T
        Otraining = np.asarray(db['Otraining']).astype(np.float32).T
        Otest = np.asarray(db['Otest']).astype(np.float32).T
        
        for u, v in zip(np.where(Otraining)[0], np.where(Otraining)[1]):
            train_X.append({'user' : str(u), 'item' : str(v)})
            train_y.append(float(M[u, v]))
        for u, v in zip(np.where(Otest)[0], np.where(Otest)[1]):
            test_X.append({'user' : str(u), 'item' : str(v)})
            test_y.append(float(M[u, v]))
        
        train_y = np.array(train_y)
        test_y = np.array(test_y)


    return train_X, test_X, train_y, test_y

def main(para_dname):

    data_dir = 'data'
    output_dir = 'output'
    dataset_name = para_dname
    # dataset_name = 'ml-1m'
    # dataset_name = 'douban'
    # dataset_name = 'yahoo_music'

    train_X, test_X, train_y, test_y = load_dataset(data_dir, dataset_name)
    v = DictVectorizer()
    train_X = v.fit_transform(train_X)
    test_X = v.transform(test_X)

    fm = pylibfm.FM(num_factors = 10, num_iter = 1000, verbose = True, task = 'regression', initial_learning_rate = 0.001, learning_rate_schedule = 'optimal', validation_size = 0.1)
    train_loss, val_loss = fm.fit(train_X, train_y)
    train_loss = np.sqrt(np.array(train_loss))
    val_loss = np.sqrt(np.array(val_loss))
    np.save(os.path.join(output_dir, dataset_name + '_trloss'), train_loss)
    np.save(os.path.join(output_dir, dataset_name + '_valloss'), val_loss)
    preds = fm.predict(test_X)
    test_loss = math.sqrt(mean_squared_error(test_y, preds))
    print(preds)
    print('Test loss: %.5f' % test_loss)

    return 0

if __name__ == '__main__':

    main(sys.argv[1])

