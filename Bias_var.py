# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:37:55 2022

@author: antho
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import sklearn.linear_model as skl
import pandas as pd
import numpy as np
from random import random, seed

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4 

def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)
	N = len(x)
	l = int((n+1)*(n+2)/2) # Number of elements in beta
	X = np.ones((N,l))
	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)
	return X


def noise(sigma_):
    #seed_noise=np.random.RandomState(314159)
    sigma=sigma_
    noise=np.random.normal(0,sigma)

    return noise
N=60
sigma=[]
complexity=[]
BVTof=[]
bias=[]
var=[]
error=[]

seed_x=np.random.RandomState(123456)
seed_y=np.random.RandomState(654321)
x=np.sort(seed_x.uniform(0,1,N))
y=np.sort(seed_y.uniform(0,1,N))

sigma_=10**(-3)
order=30
while sigma_ < 10000:
    sigma_=sigma_*1.5
    sigma.append(sigma_)


#performe OLS regression
ols=skl.LinearRegression()
noise_z=noise(1)
z = FrankeFunction(x,y)+noise_z
#bootstrap


#as a function of model complexity
for j in range(order):
    complexity.append(j)
    X = create_X(x,y,order)
    #scaling the data
    scaler = StandardScaler()
    scaler.fit(X) 
    scaler_ = StandardScaler()
    zr=np.reshape(z,(-1,1))
    scaler_.fit(zr)
    z_=scaler_.transform(zr)
    X_= scaler.transform(X)
    X_train,X_test,z_train,z_test=train_test_split(X_,z_,test_size=0.2)
    
    
    ols.fit(X_train,z_train)
    ols_pred=ols.predict(X_test)
    bias.append(np.mean( (z_test - np.mean(ols_pred, keepdims=True))**2 ))
    var.append(np.mean( np.var(ols_pred) ))
    error.append( np.mean( np.mean((z_test - ols_pred)**2) ))
    BVTof.append(np.mean(z_test-ols_pred)**2)
    
#plt.plot(complexity,BVTof,label="Bias Variance Trade off")
plt.plot(complexity,bias,label="Bias")
plt.plot(complexity,var,label="Variance")
#♦plt.plot(complexity,error,label="Error")
plt.xlabel("complexity")
#plt.xscale("log")
plt.legend()
plt.show()




