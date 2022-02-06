# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:36:31 2022

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



noise_z=noise()
N=1000
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

complexity=[]
r2_test,r2_train=np.zeros(10),np.zeros(10)
r2_test_,r2_train_=[],[]
mse_test,mse_train=np.zeros(10),np.zeros(10)
mse_test_,mse_train_=[],[]

mse_test_cv5,mse_train_cv5=[],[]
mse_test_cv10,mse_train_cv10=[],[]



    #bootstrap

k=100
order=10
for i in range (order):
    complexity.append(i)

for i in range (k):

    
    for n in range(order):
        X = create_X(x,y,n)
    
        X_train,X_test,z_train,z_test=train_test_split(X,z,test_size=0.2)
        
        beta=np.linalg.pinv(X_train.T@ X_train)@X_train.T@z_train
        z_tild=X_train@beta
        z_predict=X_test@beta
        
        R2_test=r2_score(z_test,z_predict)
        R2_train=r2_score(z_train,z_tild)
        MSE_test=mean_squared_error(z_test,z_predict)
        MSE_train=mean_squared_error(z_train,z_tild)
        
        r2_test_.append(R2_test)
        r2_train_.append(R2_train)
        mse_test_.append(MSE_test)
        mse_train_.append(MSE_train)

       
    r2_test=[r2_test+r2_test_ for r2_test,r2_test_ in zip(r2_test,r2_test_)]
    r2_train=[r2_test+r2_test_ for r2_test,r2_test_ in zip(r2_train,r2_train_)]
    mse_test=[r2_test+r2_test_ for r2_test,r2_test_ in zip(mse_test,mse_test_)]
    mse_train=[r2_test+r2_test_ for r2_test,r2_test_ in zip(mse_train,mse_train_)]

r2_test  =[i*(1/k) for i in r2_test]
r2_train =[i*(1/k) for i in r2_train]
mse_test =[i*(1/k) for i in mse_test]
mse_train=[i*(1/k) for i in mse_train]



for n in range (order):
    X = create_X(x,y,n)
    #scaling the data
    scaler = StandardScaler()
    scaler_ = StandardScaler()
    scaler.fit(X) 
    
    X_= scaler.transform(X)
    zr=np.reshape(z,(-1,1))
    scaler_.fit(zr)
    z_=scaler_.transform(zr)
    #performe OLS regression
    ols=skl.LinearRegression()
    #cross validation with 5 and 10 fold
    scores10 = cross_validate(ols, X_, z, cv=10,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
    scores5 = cross_validate(ols, X_, z, cv=5,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
    mse_test_cv10.append(abs(np.mean(scores10['test_neg_mean_squared_error'])))
    mse_train_cv10.append(abs(np.mean(scores10['train_neg_mean_squared_error'])))
    mse_test_cv5.append(abs(np.mean(scores5['test_neg_mean_squared_error'])))
    mse_train_cv5.append(abs(np.mean(scores5['train_neg_mean_squared_error'])))



plt.figure(1)
plt.plot(complexity,mse_test,label='MSE test bootstrap')
plt.plot(complexity,mse_train,label='MSE train bootstrap')
plt.plot(complexity,mse_test_cv5,label='MSE test cv 5')
plt.plot(complexity,mse_train_cv5,label='MSE train cv 5')
plt.plot(complexity,mse_test_cv10,label='MSE test cv 10')
plt.plot(complexity,mse_train_cv10,label='MSE train cv 10')
plt.title ("MSE for differents resampling methods")
plt.legend()
plt.savefig("cross_validation_complexity.png")




plt.show()
