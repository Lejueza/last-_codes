# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:03:16 2022

@author: antho
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
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
complexity=[]
r2_test,r2_train=[],[]
r2_test_,r2_train_=[],[]
mse_test,mse_train=[],[]
mse_test_,mse_train_=[],[]
mse_test_cv5,mse_train_cv5=[],[]
mse_test_cv10,mse_train_cv10=[],[]

seed_x=np.random.RandomState(123456)
seed_y=np.random.RandomState(654321)
N=[]
N_max=20

#fullfilling of the list with high density of values in interested region(N low)
while N_max <100 :
    N_max=int(N_max*1.1)
    N.append(N_max)



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
k=100


for l in range (nlambda):
   
    mse_test_ri_,mse_train_ri_=[],[]
    mse_test_cv5,mse_train_cv5=[],[]
    mse_test_cv10,mse_train_cv10=[],[]
    for j in range (len(N)):
        mse_test_ri,mse_train_ri=[],[]
        ndat=N[j]
        x=np.sort(seed_x.uniform(0,1,ndat))
        y=np.sort(seed_y.uniform(0,1,ndat))
        X = create_X(x,y,5)
        z = FrankeFunction(x,y)+noise_z
        
        
    

    #bootstrap
    
    

        for i in range (k):

        
            X_train,X_test,z_train,z_test=train_test_split(X,z,test_size=0.2)
            
            ridge=skl.Ridge(lbd[l]).fit(X_train,z_train)
    
            mse_train_ri.append(mean_squared_error(ridge.predict(X_train), z_train))
            mse_test_ri.append(mean_squared_error(ridge.predict(X_test), z_test))
            
        mse_train_ri_.append(np.mean(mse_train_ri))
        mse_test_ri_.append(np.mean(mse_test_ri))      
         
            
    for j in range (len(N)):
        ndat=N[j]
        x=np.sort(seed_x.uniform(0,1,ndat))
        y=np.sort(seed_y.uniform(0,1,ndat))
        z = FrankeFunction(x,y)+noise_z
        X = create_X(x,y,5)
    
        scaler = StandardScaler()
        scaler.fit(X) 
        X_= scaler.transform(X)
    
    
        ridge_cv=skl.Ridge(lbd[l]).fit(X_train,z_train)
        scores10 = cross_validate(ridge_cv, X_, z, cv=10,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
        scores5 = cross_validate(ridge_cv, X_, z, cv=5,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
        mse_test_cv10.append(abs(np.mean(scores10['test_neg_mean_squared_error'])))
        mse_train_cv10.append(abs(np.mean(scores10['train_neg_mean_squared_error'])))
        mse_test_cv5.append(abs(np.mean(scores5['test_neg_mean_squared_error'])))
        mse_train_cv5.append(abs(np.mean(scores5['train_neg_mean_squared_error'])))
    

    plt.figure(l)
    plt.plot(N,mse_test_ri_, label='MSE test ridge bootstrap')
    plt.plot(N,mse_train_ri_, label='MSE train ridge bootstrap')
    plt.plot(N,mse_test_cv5, label='MSE test ridge cv5')
    plt.plot(N,mse_train_cv5, label='MSE train ridge cv5')
    plt.plot(N,mse_test_cv10, label='MSE test ridge cv10')
    plt.plot(N,mse_train_cv10, label='MSE train ridge cv 10')
    
    #9plt.xscale('log')
    plt.xlabel ('Ndata')
    plt.title('lambda= {},bootstrap={}'.format(lbd[l],k))
    plt.legend()
    plt.savefig("cross_validation_Ndata_MSE_ridge.png")
