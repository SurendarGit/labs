# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:20:25 2019

@author: SurendarRajasekaran
"""
#Importing the packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,1]

#Spliting the training and test data set
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=1/3,random_state=0)

#Regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train,Y_Train)

#Predict
Y_Pred = regressor.predict(X_Test)

#Visualizing the training set results
plt.scatter(X_Train,Y_Train, color='red')
plt.plot(X_Train, regressor.predict(X_Train), color='blue')
plt.title('Salary Vs Experience - Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set results
plt.scatter(X_Test,Y_Test, color='pink')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Salary Vs Experience - Test Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()




