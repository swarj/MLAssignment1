import pandas as pd
import numpy as np

import io
from sklearn.model_selection import train_test_split
import requests

url = "https://raw.githubusercontent.com/swarj/MLAssignment1/main/Concrete_Data.xlsx%20-%20Sheet1.csv"

df = pd.read_csv(url)
train, test = train_test_split(df, test_size = 0.2)
x = train.iloc[0:,0:8]
x['8'] = 1
x = np.array(x)
y = np.array(train.iloc[0:,8:])
x_test = test.iloc[0:, 0:8]
x_test['8'] = 1
x_test = np.array(x_test)
y_test = np.array(test.iloc[0:,8:])

def cost(df,params):
  total_cost = 0
  for i in range(len(df)):
    total_cost += (1/len(df)) * ((df[i]*params).sum() - y[i])**2
  return total_cost


def gd(df, params, lrate, iter_value):
  for i in range (iter_value): #more iterations, closer to mean value
    slopes = np.zeros(9) # a slope for each parameter
    for j in range(len(df)): # number of training data points
      for k in range(9): #pass through parameters
        slopes[k] += (1/len(df)) * ((df[j] * params).sum() - y[j])*df[j][k] #avg of number of rows
    params = params - lrate * slopes
    print(cost(df,params))
  return params

def predict(X, params):
  y_predicted = np.dot(X, params)
  return y_predicted

params = np.zeros(9)
lrate = 0.0000001
iter_value = 10000
params = gd(x,params, lrate, iter_value)
print(params)
print(predict(x_test, params))
print(y_test)
