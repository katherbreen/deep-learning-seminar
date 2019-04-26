# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:56:18 2019

@author: Kathy_Breen
"""

import os
import pandas as pd
import pickle

root = 'D:\\DL_Seminar\\Week5'
os.chdir(root)

HyperParamOut = pickle.load(open('HyperParamOut.pkl','rb'))

# define dictionary to store only the data I want (RMSE) to make into a dataframe later
df_dict = {
#        'run':[],
        'acc':[]
        }
for run in HyperParamOut:
#    df_dict['run'].append(int(run))
    df_dict['acc'].append(HyperParamOut[run]['test_acc'])
    
avg_acc = pd.DataFrame.from_dict(df_dict)

# get the indices for the top 10 best performers for each metric
N = 10
best_perf = avg_acc.nlargest(N,'acc')
