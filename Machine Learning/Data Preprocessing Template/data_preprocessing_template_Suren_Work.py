# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 08:04:13 2019

@author: SurendarRajasekaran
"""

#Dataset Preprocessing

#Impoting Libraties
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset and segregating the independant (Col-0,1,2) and dependant (Col-3) values
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #X= Col-0,1,2
Y = dataset.iloc[:, 3].values #Y= Col-3

#Taking care of Missing Data with deprecated version Imputer
#Old method as per video
"""from sklearn.preprocessing import Imputer #Sklearn - socket learn
imputerObj = Imputer(missing_values='NaN', strategy='mean', axis=0) # Obj initiaition with the required action like mean,median....
imputerObj = imputerObj.fit(X[:,1:3]) #Fitting or filling the empty values with the mean value and assign the same to the object
X[:,1:3] = imputerObj.transform(X[:,1:3]) # Transformed the imputer object to actual column."""

#Taking care of Missing Data with new class SimpleImputer
from sklearn.impute import SimpleImputer
impObj = SimpleImputer(missing_values=np.nan, strategy='mean')
#impObj = impObj.fit(X1[:,1:3])
X[:,1:3] = impObj.fit_transform(X[:,1:3])


#Categorical of data
#Encoding Categorical data of independant Variable with deprecated version of Categorical_features
#Old method as per video
#0=France, 2= Spain, 1=Germany
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()"""

#new Methods using Column Transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([("encoder", OneHotEncoder(),[0])],remainder="passthrough")
X=np.array(ct.fit_transform(X), dtype=np.float)

#Encoding Categorical data of dependant Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Splitting of training and testing data set - It will split in to 4 var, Size 0.2 = 20%
#and random state=0 will provide the same result for any time run
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Scaling the data - Fit and transform is only required for training set and the same will reflected
# in Test set
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_Train=sc_X.fit_transform(X_Train)
X_Test=sc_X.transform(X_Test)

sc_y = StandardScaler()
Y_Train = sc_y.fit_transform(Y_Train.reshape(-1,1))
Y_Test=sc_y.transform(Y_Test.reshape(-1,1))

