# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:28:07 2019

@author: SurendarRajasekaran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

# Checking the prediction using simple linear regression
from sklearn.linear_model import LinearRegression
line_reg=LinearRegression()
line_reg.fit(X,Y)

plt.scatter(X,Y, color="red")
plt.plot(X, line_reg.predict(X), color="blue")

from sklearn.preprocessing import PolynomialFeatures

Poly_reg = PolynomialFeatures(degree = 4)
# Fit the polynomial regression to X and and then assign X_Poly to the linear regression for the prediction
X_Poly = Poly_reg.fit_transform(X)
Poly_reg.fit(X_Poly,Y)
line_reg2 = LinearRegression()
line_reg2.fit(X_Poly,Y)

# Predicting the salary for the salary who have experience of 6.5 is 16 lakhs and predicting the employee salary is truth or not
line_reg2.predict(Poly_reg.fit_transform([[6.5]]))

# Visualising the Polynomial Regression results
plt.scatter(X,Y, color="red")
plt.plot(X, line_reg2.predict(Poly_reg.fit_transform(X)), color="blue")
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y, color="red")
plt.plot(X_grid, line_reg2.predict(Poly_reg.fit_transform(X_grid)), color="blue")
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




