# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:48:36 2022

@author: antho
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,cross_validate
import sklearn.linear_model as skl
import pandas as pd
import numpy as np
from random import random, seed

def noise():
    seed_noise=np.random.RandomState(314159)
    noise=seed_noise.normal(0,1)

    return noise

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

	return term1 + term2 + term3 + term4 


seed_split=42
noise_z=noise()
N=70
seed_x=np.random.RandomState(123456)
seed_y=np.random.RandomState(654321)
x=np.sort(seed_x.uniform(0,1,N))
y=np.sort(seed_y.uniform(0,1,N))

z = FrankeFunction(x,y)+noise_z


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
lbd=1e4
nlambda=9
lbd=np.logspace(-4,4,nlambda)

complexity=[]


mse_train_ri,mse_test_ri=[],[]
mse_test_ri_,mse_train_ri_=[],[]
mse_test_cv5,mse_train_cv5=[],[]
mse_test_cv10,mse_train_cv10=[],[]


    #bootstrap

k=100
order=10

for i in range (order):
    complexity.append(i)
    
for l in range (nlambda):
   
    mse_test_ri_,mse_train_ri_=[],[]
    mse_test_cv5,mse_train_cv5=[],[]
    mse_test_cv10,mse_train_cv10=[],[]
    for j in range(order):
        mse_train_ri,mse_test_ri=[],[]
        X = create_X(x,y,order)
        #scaling the data
        scaler = StandardScaler()
        scaler.fit(X) 
        X_= scaler.transform(X)
        for i in range (k):

            X_train,X_test,z_train,z_test=train_test_split(X,z,test_size=0.2) 
            #ridge
        
            ridge=skl.Ridge(lbd[l]).fit(X_train,z_train)
    
            mse_train_ri.append(mean_squared_error(ridge.predict(X_train), z_train))
            mse_test_ri.append(mean_squared_error(ridge.predict(X_test), z_test))
    

       
        mse_train_ri_.append(np.mean(mse_train_ri))
        mse_test_ri_.append(np.mean(mse_test_ri))
    
    
    for n in range (order):
        X = create_X(x,y,n)
        #scaling the data
        scaler = StandardScaler()
        scaler.fit(X) 
        X_= scaler.transform(X)
        
        #performe ridge regression
        ridge=skl.Ridge(lbd[l]).fit(X_train,z_train)
        #cross validation with 5 and 10 fold
        scores10 = cross_validate(ridge, X_, z, cv=10,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
        scores5 = cross_validate(ridge, X_, z, cv=5,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
        mse_test_cv10.append(abs(np.mean(scores10['test_neg_mean_squared_error'])))
        mse_train_cv10.append(abs(np.mean(scores10['train_neg_mean_squared_error'])))
        mse_test_cv5.append(abs(np.mean(scores5['test_neg_mean_squared_error'])))
        mse_train_cv5.append(abs(np.mean(scores5['train_neg_mean_squared_error'])))
        
    plt.figure(1)
    plt.plot(complexity,mse_test_ri_, label='MSE test ridge bootstrap')
    plt.plot(complexity,mse_train_ri_, label='MSE train ridge bootstrap')
    plt.plot(complexity,mse_test_cv5, label='MSE test ridge cv5')
    plt.plot(complexity,mse_train_cv5, label='MSE train ridge cv5')
    plt.plot(complexity,mse_test_cv10, label='MSE test ridge cv10')
    plt.plot(complexity,mse_train_cv10, label='MSE train ridge cv 10')
    
    #plt.xscale('log')
    plt.xlabel ('complexity')
    plt.title('lambda= {},bootstrap={},Ndata={}'.format(lbd[l],k,N))
    plt.legend()
    plt.show()



    



