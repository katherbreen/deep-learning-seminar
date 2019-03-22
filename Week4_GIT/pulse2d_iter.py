
# coding: utf-8

# IMPORT PACKAGES

import pandas as pd
import numpy as np
import math  
import pickle 

#%% DEFINE FUNCTIONS


# FUNCTIONS
def time2node(x,Rt,Vw):
    t = (Rt*x)/Vw
    return t

def peakC(t,C0,A,Dx,Dy):
    Cmax = (C0*A)/(4*t*math.pi*math.sqrt(Dx*Dy)) #* 0.0001  # convert output value to percent
    return Cmax

def plumeDim(t,D):
    sigma3 = 3*math.sqrt(2*D*t)
    return sigma3


#%% read in input parameters
    
inputs = pd.read_csv('pulse2d_iterinput.txt',sep='\t')
inputs.head() # print out first few rows to console


#%% iteratively run the model for each system state initialization

datadict = {}
for row in range(inputs.shape[0]):
    Dx = inputs.loc[row,'Dx']
    Dy = inputs.loc[row,'Dy']
    Vw = inputs.loc[row,'Vw']
    C0 = inputs.loc[row,'C0']
    A = inputs.loc[row,'A']
    Rt = inputs.loc[row,'Rt']
    
    #Define model domain and interval size
    h = 5
    domain = np.linspace(h,100,20)
    
    # preallocate empty lists to store output
    t_out = []
    Cmax_out = []
    sigma3x_out = []
    sigma3y_out = []

    # Write for loop to iterate over model domain
    for x in domain:

        # Apply functions for each interation and store values
        t = time2node(x,Rt,Vw)  # calculate value
        t_out.append(t)  # store value
        Cmax = peakC(t,C0,A,Dx,Dy)
        Cmax_out.append(Cmax)
        sigma3x = plumeDim(t,Dx)
        sigma3x_out.append(sigma3x)
        sigma3y = plumeDim(t,Dy)
        sigma3y_out.append(sigma3y)
    
    # Write lists to dictionary then convert to dataframe...write to output file
    data = {'t': t_out,
        'Cmax': Cmax_out,
        'sigma3x': sigma3x_out,
        'sigma3y': sigma3y_out}
    df = pd.DataFrame.from_dict(data)
    datadict[row] = df

pickle.dump(datadict, open( "pulse2d_iteroutput.pkl", "wb" ))
