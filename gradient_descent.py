import numpy as np
import pandas as pd
import io
import requests
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/swarj/MLAssignment1/main/Concrete_Data.xlsx%20-%20Sheet1.csv"
df = pd.read_csv(url)
train, test = train_test_split(df, test_size = 0.2)    #split the data
x_train = np.array(train.iloc[0:, 0:8])                #load x_train data into an array
y_train = np.array(train.iloc[0:, 8:])                 #load the output training data into an array
x_test = np.array(test.iloc[0:, 0:8])

def cost(x,y,params):
    total = 0
    for i in range(len(x)):
        total = (1/len(x)) * ((x[i]* params).sum() - y[i])**2
    return total

def gradient_descent(x, y, params, lr, iterations):
    for i in range(iterations):
        attr = np.zeros(x.shape[1])
        attr = attr.astype('float64')
        for j in range(len(x)):
            for k in range(x.shape[1]):
                attr[k] += (1/len(x)) * ((x[j]*params).sum() - y[j]) * x[j][k]
        params = params - lr * attr
    return params

def predict(X, params):
    y_predicted = np.dot(X, params)
    return y_predicted

params = np.zeros(x_train.shape[1])
l_rate = 0.1
it = 100
params = gradient_descent(x_train, y_train, params, l_rate, it)
print(params)
