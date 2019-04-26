# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:14:28 2019

@author: Kathy_Breen
"""

from keras.layers import Input, Masking, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.optimizers import Adagrad
import numpy as np
import pandas as pd
import os
import h5py
import pickle
import tensorflow as tf
import random as rn
from keras import backend as K
#from matplotlib import rcParams
#rcParams['font.family'] = 'serif'
#rcParams['font.sans-serif'] = ['Times New Roman']
#import matplotlib.pyplot as plt

#%% SETTINGS FOR REPRODUCIBLE RESULTS DURING DEVELOPMENT

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

#import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#%% set WD

root = 'D:\\DL_Seminar\\Week5'
os.chdir(root)

#%% read in hyperparameters

hyper_params = pd.read_csv('h_tuning.txt',sep='\t')

#%% read in data

# columns for target data:
# 0: is not an extreme value
# 1: is an extreme value (> mu + 4sigma)

with h5py.File('week5_precip.hdf5','r') as f:
    X_train = np.array(f["precip_train"])
    X_test = np.array(f["precip_test"])
    Y_train = np.array(f["targets_train"])
    Y_test = np.array(f["targets_test"])
    test_labels = np.array(f["test_labels"])
    
#%% variables/hyperparameters   
X_Nfeatures = X_train.shape[1]
Y_Nfeatures = Y_train.shape[1]

padding_val = -1.0
epochs = 250

HyperParamOut = {}

for config in hyper_params.iterrows(): 
    run = config[0]
    row = config[1]
    batch_size = int(row['batch_size'])
    do = row['do']
    Nlyr_MLP = int(row['Nlyr_MLP'])
    Nnodes_MLP = int(row['Nnodes_MLP'])
    print('RUN: ',run)
    print(row)
    
    main_input = Input(shape=(X_Nfeatures,),
                   dtype='float'
                   )
    x = BatchNormalization()(main_input)
    
    x = Masking(mask_value=padding_val,
                batch_input_shape=(batch_size,X_Nfeatures)
                )(main_input)
    
    x = Dense(Nnodes_MLP, activation='relu')(x)
    x = BatchNormalization()(x)
    Dropout(do)(x)
        
    if Nlyr_MLP > 1:
        for i in range(1,Nlyr_MLP):
            x = Dense(Nnodes_MLP, activation='relu')(x)
            x = BatchNormalization()(x)
            Dropout(do)(x)
    
    final_output = Dense(Y_Nfeatures, activation='softmax', name='final_output')(x)
    
    model = Model(inputs=[main_input], 
                  outputs=[final_output]
                  )
    
    adagrad = Adagrad(clipvalue=0.01)  
    model.compile(loss='categorical_crossentropy',
                  optimizer=adagrad,
                  metrics=['categorical_accuracy']
                  )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.8,  
                                  patience=15,
                                  verbose=0)
    
    early_stop = EarlyStopping(monitor='val_loss', 
                               min_delta=0.00001,  
                               patience=100, 
                               verbose=1)  
    
    csv_logger = CSVLogger('training_' + str(epochs) + 'LSTM-MLP.log')
    
    history = model.fit([X_train],[Y_train],
                            epochs=epochs,
                            verbose=0,
                            batch_size=batch_size,
                            validation_data=([X_test],[Y_test]),
#                            callbacks=[reduce_lr, early_stop, csv_logger]
                            callbacks=[reduce_lr, csv_logger]
                            )
    
    histdict = history.history
    
    y_prob = model.predict([X_test],batch_size=batch_size)  # get class probabilities
    # how often predictions have maximum in the same spot as true values
    test_acc = np.mean(np.equal(np.argmax(Y_test, axis=-1), np.argmax(y_prob, axis=-1)))
    print('Test Prediction Accuracy', test_acc)
    HyperParamOut[run] = {'test_acc':test_acc}
    
pickle.dump(HyperParamOut, open( "HyperParamOut.pkl", "wb" ))



























































































