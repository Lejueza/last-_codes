# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:10:19 2022

@author: antho
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
import sklearn.linear_model as skl
import numpy as np
from random import random, seed

def noise(sigma_):
    seed_noise=np.random.RandomState(314159)
    sigma=sigma_
    noise=np.random.normal(0,sigma_)

    return noise

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

	return term1 + term2 + term3 + term4 


seed_x=np.random.RandomState(123456)
seed_y=np.random.RandomState(654321)
N_max=100

sigma=[]

complexity=[]
r2_test,r2_train=[],[]
r2_test_,r2_train_=[],[]
mse_test,mse_train=[],[]
mse_test_,mse_train_=[],[]
mse_test_cv5,mse_train_cv5=[],[]
mse_test_cv10,mse_train_cv10=[],[]

x=np.sort(seed_x.uniform(0,1,N_max))
y=np.sort(seed_y.uniform(0,1,N_max))

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

X = create_X(x,y,5)
sigma_=10**(-1)

while sigma_ < 1000:
    sigma_=sigma_*1.5
    sigma.append(sigma_)

    
for j in range (len(sigma)):
    mse_test_,mse_train_=[],[]
    r2_test_,r2_train_=[],[]

    noise_z=noise(sigma[j])
    z = FrankeFunction(x,y)+noise_z

    #bootstrap
    
    k=100
    order=6

    for i in range (k):

        
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

    r2_test_mean=sum(r2_test_)*1/k
    r2_train_mean=sum(r2_train_)*1/k
    mse_test_mean=sum(mse_test_)*1/k
    mse_train_mean=sum(mse_train_)*1/k
    
    r2_test.append(r2_test_mean)
    r2_train.append(r2_train_mean)
    mse_test.append(mse_test_mean)
    mse_train.append(mse_train_mean)

    
for j in range(len(sigma)):
    noise_z=noise(sigma[j])
    z = FrankeFunction(x,y)+noise_z
    #scaling the data
    scaler = StandardScaler()
    scaler.fit(X) 
    X_= scaler.transform(X)
    
    #performe OLS regression
    ols=skl.LinearRegression()
    scores10 = cross_validate(ols, X_, z, cv=10,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
    scores5 = cross_validate(ols, X_, z, cv=5,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
    mse_test_cv10.append(abs(np.mean(scores10['test_neg_mean_squared_error'])))
    mse_train_cv10.append(abs(np.mean(scores10['train_neg_mean_squared_error'])))
    mse_test_cv5.append(abs(np.mean(scores5['test_neg_mean_squared_error'])))
    mse_train_cv5.append(abs(np.mean(scores5['train_neg_mean_squared_error'])))
print(mse_test_cv10)
      
      
plt.figure(1)

plt.plot(sigma,mse_test_cv10,label='MSE test cv 10')

plt.xscale('log')
plt.xlabel('noise')
plt.ylabel('mse')
plt.legend()

plt.figure(2)
plt.plot(sigma,mse_test_cv5,label='MSE test cv 5')
#plt.plot(sigma,mse_train_cv5,label='MSE train cv 5')
plt.plot(sigma,mse_test_cv10,label='MSE test cv 10')
#plt.plot(sigma,mse_train_cv10,label='MSE train cv 10')
plt.plot(sigma,mse_test,label='MSE test bootstrap')
plt.xscale('log')
plt.xlabel('noise')
plt.ylabel('MSE')
plt.legend()
plt.savefig("cross_validation_Noise_MSE_test.png")
plt.show()

plt.figure(3)
#plt.plot(sigma,mse_test_cv5,label='MSE test cv 5')
plt.plot(sigma,mse_train_cv5,label='MSE train cv 5')
#plt.plot(sigma,mse_test_cv10,label='MSE test cv 10')
plt.plot(sigma,mse_train_cv10,label='MSE train cv 10')
plt.plot(sigma,mse_train,label='MSE train bootstrap')
plt.xscale('log')
plt.xlabel('noise')
plt.ylabel('MSE')
plt.legend()
plt.savefig("cross_validation_Noise_MSE_train.png")
plt.show()
