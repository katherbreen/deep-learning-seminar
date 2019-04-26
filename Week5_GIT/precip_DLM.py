# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:14:28 2019

@author: Kathy_Breen
"""

from keras.layers import Input, Masking, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
import numpy as np
import os
import h5py
import tensorflow as tf
import random as rn
from keras import backend as K
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']
import matplotlib.pyplot as plt

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
epochs = 2000
batch_size = 500
do = 0.2
Nlyr_MLP = 25
Nnodes_MLP = 10

#%% NETWORK!!!!! ------------------------------------------------
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

model.compile(loss='binary_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy']
              )

reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                              factor=0.9,  
                              patience=15,
                              verbose=1)

early_stop = EarlyStopping(monitor='val_acc', 
                           min_delta=0.000001,  
                           patience=100, 
                           verbose=1)  

csv_logger = CSVLogger('training_' + str(epochs) + 'LSTM-MLP.log')

history = model.fit([X_train],[Y_train],
                        epochs=epochs,
                        verbose=1,
                        batch_size=batch_size,
                        validation_data=([X_test],[Y_test]),
#                        callbacks=[reduce_lr, early_stop, csv_logger]
                        callbacks=[reduce_lr, csv_logger]
#                        callbacks=[csv_logger]
                        )

histdict = history.history

y_prob = model.predict([X_test],batch_size=batch_size)  # get class probabilities
# how often predictions have maximum in the same spot as true values?
test_acc = np.mean(np.equal(np.argmax(Y_test, axis=-1), np.argmax(y_prob, axis=-1)))

#%% plot results

# accuracy
acc_train = history.history['acc']
acc_test = history.history['val_acc']
xplot = list(range(len(acc_train)))

fig = plt.figure(num=1, figsize=(8,6))
ax = fig.add_subplot(111)
train = ax.plot(xplot,acc_train,'b-',label='Train',linewidth=6)
test = ax.plot(xplot,acc_test,'r--',label='Test',linewidth=4)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
curves = train+test
labels = [c.get_label() for c in curves]
ax.legend(curves, labels, loc=0)
plt.tight_layout()
plt.title('Accuracy')
plt.savefig('accuracy.png')
plt.show()

# loss
loss_train = history.history['loss']
loss_test = history.history['val_loss']
xplot = list(range(len(loss_train)))

fig = plt.figure(num=2, figsize=(8,6))
ax = fig.add_subplot(111)
train = ax.plot(xplot,loss_train,'b-',label='Train',linewidth=6)
test = ax.plot(xplot,loss_test,'r--',label='Test',linewidth=4)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
curves = train+test
labels = [c.get_label() for c in curves]
ax.legend(curves, labels, loc=0)
plt.tight_layout()
plt.title('Loss')
plt.savefig('loss.png')
plt.show()

##%% VISUALIZATION - this was really hard to get to work.  The following two forums helped me figure out the tricky package install...
## https://stackoverflow.com/questions/40632486/dot-exe-not-found-in-path-pydot-on-python-windows-7
## https://www.codesofinterest.com/2017/02/visualizing-model-structures-in-keras.html
#from keras.utils import plot_model
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/release/bin'
#plot_model(model, to_file='modelAPI.png', show_shapes=True, show_layer_names=True)



























































































