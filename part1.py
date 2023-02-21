import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/marcoan01/MLA/main/ale.csv"
data = pd.read_csv(url)
# Features
x1 = data['anchor_ratio']
x2 = data['trans_range']
x3 = data['node_density']
y = data['ale']

#normalized
x1 = (x1 - x1.mean()) / x1.std()
x2 = (x2 - x2.mean()) / x2.std()
x3 = (x3 - x3.mean()) / x3.std()

# add m0 variable to matrix
x = np.c_[x1,x2,x3, np.ones(x1.shape[0])]
train_size = int(0.8 * len(x))
X_train, X_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#variables
lrate = 0.01
iter = 2000
np.random.seed(501)                                                                     #random starter values for slopes
weights = np.random.rand(4)                                                             # 3 params + 1s
n = y.size                                                                              # number of examples in the dataset

#Gradient Descent Implementation
def gradient_descent(x, y, weights, iter, lrate):
    prev_weights = [weights]
    prev_cost = []
    for i in range(iter):
      prediction = np.dot(x, weights)                                                   #w0 + w1x1+w2x2 + w3x3
      dv = prediction - y                                                               #how far the prediction is off the line from the actual
      cost = 1 / (2 * n) * np.dot(dv.T, dv)                                             #cost function 1(2n) * MSE
      prev_cost.append(cost)                                                            #update costs based on new cost amount
      derivative = (1 / n) * lrate * np.dot(x.T, dv)                                    #derivative equation to change weights, dotproduct does summation
      weights = weights - derivative                                                    #adjust weight
      prev_weights.append(weights)                                                      #update weights save previous ones
    return prev_weights, prev_cost

#predicts y values based on the x_test and the parameters obtained by the model
def predict(x_t, weights):
    y_predicted = np.dot(x_t, weights)
    return y_predicted

prev_weights, prev_cost = gradient_descent(X_train,y_train,weights,iter, lrate)
print("\nPredicted set of values: " + str(predict(X_test, prev_weights[-1])))
print("\n\nActual set of values: " + str(np.array(y_test)))
print("\nMinimum cost for the model: " + str(prev_cost[-1]))
