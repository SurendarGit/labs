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

#Splitting of training and testing data set - It will split in to 4 var, Size 0.2 = 20%
#and random state=0 will provide the same result for any time run
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Scaling the data - Fit and transform is only required for training set and the same will reflected
# in Test set
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_Train=sc_X.fit_transform(X_Train)
X_Test=sc_X.transform(X_Test)

sc_y = StandardScaler()
Y_Train = sc_y.fit_transform(Y_Train.reshape(-1,1))
Y_Test=sc_y.transform(Y_Test.reshape(-1,1))"""

