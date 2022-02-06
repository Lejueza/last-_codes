from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
import pandas as pd
import numpy as np
from random import random, seed



    
def noise(sigma_):
    #seed_noise=np.random.RandomState(314159)
    sigma=sigma_
    noise=np.random.normal(0,sigma)

    return noise

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

	return term1 + term2 + term3 + term4 


seed_split=42

N=100
seed_x=np.random.RandomState(123456)
seed_y=np.random.RandomState(654321)
x=np.sort(seed_x.uniform(0,1,N))
y=np.sort(seed_y.uniform(0,1,N))




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
sigma=[]





X = create_X(x,y,5)
sigma_=10**(-1)

while sigma_ < 1000:
    sigma_=sigma_*1.5
    sigma.append(sigma_)



    #bootstrap

k=100
order=5
X = create_X(x,y,order)
for l in range (nlambda):
   
    mse_test_ri_,mse_train_ri_=[],[]
    mse_test_cv5,mse_train_cv5=[],[]
    mse_test_cv10,mse_train_cv10=[],[]
    
    for j in range(len(sigma)):
        
        mse_train_ri,mse_test_ri=[],[]
        for i in range (k):
            noise_z=noise(sigma[j])
            z = FrankeFunction(x,y)+noise_z
            X_train,X_test,z_train,z_test=train_test_split(X,z,test_size=0.2) 
            #ridge
            
            ridge=skl.Ridge(lbd[l]).fit(X_train,z_train)
            
            mse_train_ri=(mean_squared_error(ridge.predict(X_train), z_train))
            mse_test_ri=(mean_squared_error(ridge.predict(X_test), z_test))
    

       
        mse_train_ri_.append(np.mean(mse_train_ri))
        mse_test_ri_.append(np.mean(mse_test_ri))


        
    for j in range(len(sigma)):
        noise_z=noise(sigma[j])
        z = FrankeFunction(x,y)+noise_z
        #scaling the data
        scaler = StandardScaler()
        scaler.fit(X) 
        X_= scaler.transform(X)
    
        #performe OLS regression
        ridge_cv=skl.Ridge(lbd[l]).fit(X_train,z_train)
        scores10 = cross_validate(ridge_cv, X_, z, cv=10,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
        scores5 = cross_validate(ridge_cv, X_, z, cv=5,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
        mse_test_cv10.append(abs(np.mean(scores10['test_neg_mean_squared_error'])))
        mse_train_cv10.append(abs(np.mean(scores10['train_neg_mean_squared_error'])))
        mse_test_cv5.append(abs(np.mean(scores5['test_neg_mean_squared_error'])))
        mse_train_cv5.append(abs(np.mean(scores5['train_neg_mean_squared_error'])))


    

    plt.figure(l)
    plt.plot(sigma,mse_test_ri_, label='MSE test ridge')
    plt.plot(sigma,mse_train_ri_, label='MSE train ridge')
    plt.plot(sigma,mse_test_cv5, label='MSE test ridge cv5')
    plt.plot(sigma,mse_train_cv5, label='MSE train ridge cv5')
    plt.plot(sigma,mse_test_cv10, label='MSE test ridge cv10')
    plt.plot(sigma,mse_train_cv10, label='MSE train ridge cv 10')
    plt.xscale('log')
    plt.xlabel ('Noise')
    plt.title('complexity={},bootstrap={},Ndata={},lmabda={}'.format(order,k,N,lbd[l]))
    plt.legend()
    plt.show()

