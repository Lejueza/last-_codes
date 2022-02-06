# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Importations
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as skl
from sklearn.datasets import load_boston

"""
Data from Boston
"""

global boston_data
boston_data = load_boston()
boston_df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_df['MEDV'] = boston_data.target
#print(boston_df)
"""
Creation of the matrix
"""

columns_names = boston_df.columns.array
boston_array = boston_df.index.array

corr_matrix = boston_df.corr().round(1)
sns.heatmap(data=corr_matrix,annot=True)
col_name=(corr_matrix.head(0))
mean_=[]


for i in range(14):
    mean_.append(np.mean(boston_df.iloc[:,i]))
#print(mean_)

#variable without influence:
"""
chas
PTRATIO,RM,ZN-CRIM
RM,RAD,TAX,B,CRIM -ZN
PTRATIO,RM-INOX
B,TAX,RAD,DIS,AGE,NOX,ZN,CRIM-RM
B,PTRATIO,RM-AGE
MEDV,B,PTRATIO,RM-DIS
RM,ZN-RAD
RM,ZN-TAX
B,DIS,AGE,NOX,CRIM-PTRATIO
MEDV,PTRATIO,DIS,AGE,RM,ZN
B,DIS-MEDV


"""


val=corr_matrix.iloc[10,0]# return val(11th line, 1st col)
#print(val)


def multiplicity(n=1,N=1) : # n = number of features ; N = multiplicity

	# Initialisation
	count_1 = 0
	Matrix = np.zeros((N,1,n))

	# Begining of the big loop

	while count_1 < N:
			
		count_2 = 0 	# counter of position

		# fullfill the individuals values

		if count_1 == 0:
			for a in range(n):
				Matrix[count_1,0,a] = a

		elif count_1 == 1:
			x1 = 0
			x2 = 1
			L = np.array((x1,x2))
			print(L)
			a = 0

			while x2 < n :   
				Matrix[count_1,0]=L
				x2 += 1
				a += 1

		count_1 += 1 # counter of layers
	return Matrix

def creat_X():
    X=boston_df
    for j in range(14):
        
        for  i in range(14):
            X["pr"+str(i)+str(j)]=X.iloc[:,j]*X.iloc[:,i]
    return X
#%%


X=boston_df.drop(columns=["MEDV"])
#X=boston_df.drop(columns=["CHAS"]) #remove the CHAS colum because low correlation
Y=boston_df["MEDV"]

poly = PolynomialFeatures(2)
X = poly.fit_transform(X)
#"Y = poly.fit_transform(Y)
print(X)
print(Y)
#%%

#bootstrap for indentifie the better lamda
bootstrap=100
lbd=[0.001,0.01,0.1,1,10,100]

mse_min=[]
for i in range (bootstrap):
    mseteri=[]
    for i in range (6):
      
        X_train,X_test,MEDV_train,MEDV_test=train_test_split(X,Y,test_size=0.2)
        ridge=skl.Ridge(lbd[i]).fit(X_train,MEDV_train)
        #scores10 = cross_validate(ridge, X, Y, cv=10,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
        #mse_test_cv10=(abs(np.mean(scores10['test_neg_mean_squared_error'])))
        lasso=skl.Lasso().fit(X_train,MEDV_train)
        mse_train_ri = mean_squared_error(ridge.predict(X_train),MEDV_train)
        mse_test_ri = mean_squared_error(ridge.predict(X_test),MEDV_test)
        mseteri.append(mse_test_ri)
        mse_train_la = mean_squared_error(lasso.predict(X_train),MEDV_train)
        mse_test_la = mean_squared_error(lasso.predict(X_test),MEDV_test)
    mini=min(mseteri)
    mse_min.append(mini)
print("lambda optimized= {}".format(np.mean(mse_min)))
plt.plot(lbd,mseteri,label="mse_test_ridge")
plt.xscale("log")
plt.xlabel("lambda")
plt.ylabel("mse test ridge")


"""
    print("mse train ridge {}".format(mse_train_ri))
    print("mse test  ridge {}".format(mse_test_ri))
    print("mse train lasso {}".format(mse_train_la))
    print("mse test  lasso {}".format(mse_test_la))
"""

