# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:06:53 2019

@author: SurendarRajasekaran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#Categorgise the data col - State
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([("encoder",OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(ct.fit_transform(X),dtype=np.float)

#Avoid Dummy variable trap
X=X[:,1:]

#Split the training set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size=0.2,random_state=0)

#LinearRegression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train,Y_Train)

#Manual Backward Elimination for the dataset

"""import statsmodels.api as sm

#y== b0 + b1x1 + b2x2 + b3x3 + b4D1
#b0 is a constant np.ones((50,1)), So just assigning some dummy constant say "1" to all 50 rows before X values, So 1 + dataset X
X=np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
X_Opt = X [:, [0,1,2,3,4,5]]
len(X_Opt[0])
regressor_OLS = sm.OLS(endog=Y, exog=X_Opt).fit()
regressor_OLS.summary()
#Remove the highest P value which is found from above summary, So Col X1 or [1]=0.953 it is highest so remove the variable in the next step
X_Opt = X [:, [0,2,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_Opt).fit()
regressor_OLS.summary()
#Remove the highest P value which is found from above summary, So Col X1 or [2]=0.962 it is highest so remove the variable in the next step
X_Opt = X [:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_Opt).fit()
regressor_OLS.summary()
#Remove the highest P value which is found from above summary, So Col X2 or [4]=0.602 it is highest so remove the variable in the next step
X_Opt = X [:, [0,3,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_Opt).fit()
regressor_OLS.summary()
#Remove the highest P value which is found from above summary, So Col X2 or [5]=0.060 it is highest so remove the variable in the next step
X_Opt = X [:, [0,3]]
regressor_OLS = sm.OLS(endog=Y, exog=X_Opt).fit()
regressor_OLS.summary()"""

#Automated backward elimination

import statsmodels.api as sm
X=np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars): 
        print (i)
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                print (j)
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
                    #print (x)
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)