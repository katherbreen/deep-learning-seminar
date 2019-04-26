# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:00:45 2019

@author: Kathy_Breen
"""

import os
import pandas as pd
import numpy as np
import h5py
from random import shuffle
from sklearn import preprocessing
from keras.utils import to_categorical
import datetime
from matplotlib import rcParams  # next 3 lines set font family for plotting
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['TImes New Roman']
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

root = 'D:\\DL_Seminar\\Week5'
os.chdir(root)

#%% read in NOAA precipitation data

pcp_file = '722280-13876.pcp'
precip_data = pd.read_csv(pcp_file,  # filename
                     sep='\t',  # delimiter
                     na_values=-99)  # identify -99 as NaN value

#%% identify extreme values

ts = precip_data['precip']  # ts == timeseries
ts = np.array(ts,ndmin=2).T
#ts = ts.reshape((ts.shape[0],1))
ts_mu = np.zeros(ts.shape)
ts_sigma = np.zeros(ts.shape) 
extreme_binary = np.zeros(ts.shape)

# calculate moving average/std dev fr SM same way it was done for input data
window_size = 90  # calc moving avg using a window of N days
stride = int(window_size*.25)  # overlap windows by 50%
window_beg_idx = list(range(0, len(ts), stride))
ts_ma = [np.mean(ts[i:i+window_size]) for i in window_beg_idx if i+window_size <= len(ts)]
sigma_mult = 4
ts_std = sigma_mult * [np.std(ts[i:i+window_size]) for i in window_beg_idx if i+window_size <= len(ts)]
for i,idx in enumerate(window_beg_idx):
#            print(i,i+window_size)
    if idx+window_size <= len(ts):
        ts_mu[idx:idx+window_size] = ts_ma[i]
        ts_sigma[idx:idx+window_size] = ts_std[i]
    else:
        ts_mu[idx:len(ts)] = ts_ma[len(ts_ma)-1]
        ts_sigma[idx:len(ts)] = ts_std[len(ts_std)-1]
        
for i,val in enumerate(ts):
    mu_plus_sigma = ts_mu[i] + ts_sigma[i]
    if val > mu_plus_sigma:
        extreme_binary[i] = 1

#%% shuffle, partition and preprocess data
    
# shuffle samples
ismpls = list(i for i in range(0,ts.shape[0]))
shuffle(ismpls)
ismpls = np.argsort(ismpls)
ts_shuffle = ts[ismpls,:]
extreme_binary_shuffle = extreme_binary[ismpls,:]

# partition X and Y into training (90%) and test (10%) sets
split_idx = int(round(ts.shape[0]*0.9))
ts_train = ts_shuffle[:split_idx,:]
ts_test = ts_shuffle[split_idx:,:]
extreme_binary_train = extreme_binary_shuffle[:split_idx,:]
extreme_binary_test = extreme_binary_shuffle[split_idx:,:]

# normalize inputs for each feature (column) across all samples (rows)
mask_value = -1  # after data is normalized, replace NaN idx with mask value
ts_norm_train = ts_train
train_nan_idx = np.isnan(ts_train)
ts_norm_train[train_nan_idx] = 0  # sklearn doesn't like nans, but will only normalize non-zero values
ts_norm_train = preprocessing.normalize(ts_norm_train,norm='l2',axis=0)
ts_norm_train[train_nan_idx] = mask_value

ts_norm_test = ts_test
test_nan_idx = np.isnan(ts_test)
ts_norm_test[test_nan_idx] = 0  # sklearn doesn't like nans, but will only normalize non-zero values
ts_norm_test = preprocessing.normalize(ts_norm_test,norm='l2',axis=0)
ts_norm_test[test_nan_idx] = mask_value

# convert to categorical data for classification algorithm
extreme_categorical_train = to_categorical(extreme_binary_train, num_classes=2)
extreme_categorical_test = to_categorical(extreme_binary_test, num_classes=2)

#%% Write output data to process with DLM    

with h5py.File('week5_precip.hdf5','w') as f:
    precip_train = f.create_dataset('precip_train',ts_norm_train.shape,data=ts_norm_train)
    precip_test = f.create_dataset('precip_test',ts_norm_test.shape,data=ts_norm_test)
    targets_train = f.create_dataset('targets_train',extreme_categorical_train.shape,data=extreme_categorical_train)
    targets_test = f.create_dataset('targets_test',extreme_categorical_test.shape,data=extreme_categorical_test)
    test_labels = f.create_dataset('test_labels',extreme_binary_test.shape,data=extreme_binary_test)
    
#%% visualize raw precipitation data
base = datetime.datetime.strptime("1988-01-01", "%Y-%m-%d")
date_lst = [base + datetime.timedelta(days=x) for x in range(len(ts))]

fig = plt.figure(num=1, figsize=(8,6))
ax = fig.add_subplot(111)
# plot precipitation
y = ax.plot(date_lst,ts,'b-', label=r'$NOAA_{precip}$', linewidth=2)
# plot existence of extreme values
for i,val in enumerate(extreme_binary):
    if val == 1:
        ev = ax.plot(date_lst[i],ts[i]+1,'c.',markerfacecolor='None',markersize=12,label=r'extreme value')

# plot moving average/std dev
ymu = ax.plot(date_lst,ts_mu,'r-',label=r'$\mu$', linewidth=2)
yps = ax.plot(date_lst,ts_mu+(ts_sigma),'m--',label=r'$\mu+'+str(sigma_mult)+'\sigma$', linewidth=2)

ax.set_ylim(bottom=0,top=180)

ax.set_xlim(left=date_lst[0],right=date_lst[len(date_lst)-1])
ax.set_xlabel('Time')
ax.set_ylabel(r'Precipitation (mm)')

curves = y + ymu + yps + ev
labels = [c.get_label() for c in curves]

ax.legend(curves, labels, loc=0, prop={'size': 10})
plt.tight_layout()
plt.title(r'NOAA Precip')
plt.savefig('precip_mu_sigma.png', format='png')
plt.show()