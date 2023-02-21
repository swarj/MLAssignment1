Dataset used was obtained from: https://archive.ics.uci.edu/ml/datasets/Average+Localization+Error+%28ALE%29+in+sensor+node+localization+process+in+WSNs
Dataset is hosted in: https://raw.githubusercontent.com/marcoan01/MLA/main/ale.csv

Part 1 - Linear Regression using Gradient Descent
For this section of the document, the code can be run using the file: part1.py
These are the libraries and packages needed to run this code:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

Part 2 - Linear Regression using ML libraries

For this section of the document, the code can be run using the file: part2.py

These are the libraries and packages needed to run this code:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

This code uses the Linear_Model package in order to construct the model for the given data. After such data is split, and normalized, the data is trained using the .fit() by sending x_train and y_train.

