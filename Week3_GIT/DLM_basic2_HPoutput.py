# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:46:17 2019

@author: Kathy_Breen
"""

#%% IMPORT PACKAGES
import numpy as np
import pandas as pd
from matplotlib import rcParams  # next 3 lines set font family for plotting
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['TImes New Roman']
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import os
import h5py
import pickle
import seaborn as sns

# set working directory (change the following path to match your directory structure)
main = 'C:\\Users\\Kathy_Breen\\Documents\\DL_Seminar\\Week3'  # set directory path where this file is saved
os.chdir(main)  # make sure the Spyder is pointing to the correct 

#%% Read in *.hdf5 data sets

with h5py.File('X.hdf5','r') as f:
    X_test = np.array(f["X_test"])

with h5py.File('Y.hdf5','r') as f:
    Y_test = np.array(f["Y_test"])


#%% load, read output

hparams = pd.read_csv('h_tuning.txt',sep='\t')
rundict = pickle.load( open( "HyperParamOut.pkl", "rb" ))
Ymse_MLP = []

for run in rundict.keys():
    Ymse_MLP.append(rundict[run]['Y_mse'])
    print(rundict[run]['Y_mse'])

# find minimum loss for each permutation of parameters and plot
loss_min = np.min(Ymse_MLP)
min_idx = np.argmin(Ymse_MLP)

# define temporary histdict for plotting using the "best" run for each modeltype
histdict = rundict[min_idx]['histdict']

#%% PLOT OUTPUT

# define datasets to plot
loss_train = histdict['loss']
loss_test = histdict['val_loss']
xplot = list(range(len(loss_train)))
Ymse_MLP = np.array(Ymse_MLP)  # change to numpy array 

# plot
fig = plt.figure(num=1, figsize=(8,6))
ax1 = fig.add_subplot(111)
train = ax1.plot(xplot,np.sqrt(loss_train),'b-', label='Train', linewidth=4)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss (MSE)')
for tl in ax1.get_yticklabels():
    tl.set_color('b')
ax2 = ax1.twinx()
test = ax2.plot(xplot,np.sqrt(loss_test),'r-',label='Test',linewidth=4)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Validation Loss (MSE)')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
curves = train + test
labels = [c.get_label() for c in curves]
ax1.legend(curves, labels, loc=0)
plt.tight_layout()
plt.title(str(hparams.loc[min_idx,:]))
plt.savefig('HP_best_loss.png')
plt.show()

fig = plt.figure(num=2, figsize=(8,10))
ax1 = fig.add_subplot(211)
y_true = ax1.plot(X_test,Y_test,'ko',markersize=16,label=r'$Y_{true}$')
y_pred = ax1.plot(X_test,rundict[min_idx]['predict'],'*',color='#009191',markersize=10,label=r'$\hat{Y}_{main}$')
curves = y_true+y_pred
labels = [c.get_label() for c in curves]
ax1.legend(curves, labels, loc=0)
ax1.set_xlabel(r'$X$')
ax1.set_ylabel(r'$Y$')
ax2 = fig.add_subplot(212)
sns.distplot(Ymse_MLP,  # data to plot 
             hist=True,  # plot histogram
             kde=True,   # overlay kernel density function (PDF)
             ax=ax2,  # plot on the existing axis object created for this figure
             hist_kws={'edgecolor':'black'},  # set color to outline hist bins
             kde_kws={'linewidth': 4}  # use a thick line for the kde
             )
ax2.set_xlabel(r'$Y_{mse}$')
ax2.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig('HP_best_yyhat.png')
plt.show()

fig = plt.figure(num=3, figsize=(8,10))
ax1 = fig.add_subplot(211)
sns.distplot(Ymse_MLP[Ymse_MLP > 1],  # data to plot 
             hist=True,  # plot histogram
             bins=10,  # number of bins to use in histogram
             kde=True,   # overlay kernel density function (PDF)
             ax=ax1,  # plot on the existing axis object created for this figure
             hist_kws={'edgecolor':'black'},  # set color to outline hist bins
             kde_kws={'linewidth': 4}  # use a thick line for the kde
             )
ax1.set_title(r'$Y_{mse} > 1$')
ax1.get_yaxis().set_visible(False)
ax2 = fig.add_subplot(212)
sns.distplot(Ymse_MLP[Ymse_MLP < 1],  # data to plot 
             hist=True,  # plot histogram
             bins=10,  # number of bins to use in histogram
             kde=True,   # overlay kernel density function (PDF)
             ax=ax2,  # plot on the existing axis object created for this figure
             hist_kws={'edgecolor':'black'},  # set color to outline hist bins
             kde_kws={'linewidth': 4}  # use a thick line for the kde
             )
ax2.set_title(r'$Y_{mse} < 1$')
ax2.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig('HP_best_dist.png')
plt.show()

