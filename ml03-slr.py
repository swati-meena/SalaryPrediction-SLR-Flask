#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 	
import matplotlib.pyplot as plt
import pandas as pd	
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
dataset = pd.read_csv(r'/home/hp/Downloads/datasets/Salary_Data.csv')

X= dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

comparison = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparison)

# Visualize the training set
plt.scatter(X_train, y_train, color='red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set
plt.scatter(X_test, y_test, color='red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print('Slope of the dataset : ', m_slope)


c_intercept = regressor.intercept_
print('Intercept of the dataset : ', c_intercept)


future_pred =(m_slope * 12) + c_intercept
print(future_pred)

future_pred =(m_slope * 20) + c_intercept
print(future_pred)


# Check model performance
bias = regressor.score(X_train, y_train)
variance = regressor.score(X_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")




# Save the trained model to disk
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")


import os 
print(os.getcwd())

