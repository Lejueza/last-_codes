from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
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
lbd=1e4
nlambda=100
lbd=np.logspace(-4,4,nlambda)



mse_test_ri_,mse_train_ri_=[],[]



    #bootstrap

k=100
order=5
X = create_X(x,y,order)
for l in range (nlambda):
    for i in range (k):

        X_train,X_test,z_train,z_test=train_test_split(X,z,test_size=0.2,random_state=seed_split) 
        #ridge
        
        ridge=skl.Ridge(lbd[l]).fit(X_train,z_train)
    
        mse_train_ri=(mean_squared_error(ridge.predict(X_train), z_train))
        mse_test_ri=(mean_squared_error(ridge.predict(X_test), z_test))
    

       
    mse_train_ri_.append(np.mean(mse_train_ri))
    mse_test_ri_.append(np.mean(mse_test_ri))



#calculation bias variance

    

plt.figure(1)
plt.plot(lbd,mse_test_ri_, label='MSE test ridge')
plt.plot(lbd,mse_train_ri_, label='MSE train ridge')
plt.xscale('log')
plt.xlabel ('lambda')
plt.title('complexity={},bootstrap={},Ndata={}'.format(order,k,N))
plt.legend()

plt.show()

