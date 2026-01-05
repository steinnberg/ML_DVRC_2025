
# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                        MNIST Data loader
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/MIAI - https://fidle.cnrs.fr
# ------------------------------------------------------------------
# JL Parouty (Mars 2024)


import h5py
import os
import numpy as np
from hashlib import blake2b
import keras
import keras.datasets.mnist as mnist


# ------------------------------------------------------------------
#   A usefull class to manage our MNIST dataset
#   This class allows to manage datasets derived from the original MNIST
# ------------------------------------------------------------------


class MNIST():
    
    version = '0.1'
    
    def __init__(self):
        pass
   
    @classmethod
    def get_data(cls, normalize=True, expand=True, scale=1., train_prop=0.8, shuffle=True, seed=None):
        """
        Return original MNIST dataset
        args:
            normalize   : Normalize dataset or not (True)
            expand      : Reshape images as (28,28,1) instead (28,28) (True)
            scale      : Scale of dataset to use. 1. mean 100% (1.)
            train_prop : Ratio of train/test (0.8)
            shuffle    : Shuffle data if True (True)
            seed       : Random seed value. False mean no seed, None mean using /dev/urandom (None)
        returns:
            x_train,y_train,x_test,y_test
        """

        # ---- Seed
        #
        if seed is not False:
            np.random.seed(seed)
            print(f'Seeded ({seed})')

        # ---- Get data
        #
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print('Dataset loaded.')
        
        # ---- Concatenate
        #
        x_data = np.concatenate([x_train, x_test], axis=0)
        y_data = np.concatenate([y_train, y_test])
        print('Concatenated.')

        # ---- Shuffle
        #
        if shuffle:
            p = np.random.permutation(len(x_data))
            x_data, y_data = x_data[p], y_data[p]
            print('Shuffled.')     
        
        # ---- Rescale
        #
        n = int(scale*len(x_data))
        x_data, y_data = x_data[:n], y_data[:n]
        print(f'rescaled ({scale}).') 

        # ---- Normalization
        #
        if normalize:
            x_data = x_data.astype('float32') / 255.
            print('Normalized.')
            
        # ---- Reshape : (28,28) -> (28,28,1)
        #
        if expand:
            x_data = np.expand_dims(x_data, axis=-1)
            print('Reshaped.')

        # ---- Split
        #
        n=int(len(x_data)*train_prop)
        x_train, x_test = x_data[:n], x_data[n:]
        y_train, y_test = y_data[:n], y_data[n:]
        print(f'splited ({train_prop}).') 

        # ---- Hash
        #
        h = blake2b(digest_size=10)
        for a in [x_train,x_test, y_train,y_test]:
            h.update(a)
            
        # ---- About and return
        #
        print('x_train shape is  : ', x_train.shape)
        print('x_test  shape is  : ', x_test.shape)
        print('y_train shape is  : ', y_train.shape)
        print('y_test  shape is  : ', y_test.shape)
        print('Blake2b digest is : ', h.hexdigest())
        return  x_train,y_train, x_test,y_test
                
            
