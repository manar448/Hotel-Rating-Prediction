# test file
from datetime import time

import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, r2_score
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from preprocessing_Data import *
from sklearn.model_selection import train_test_split
from main import data_test
import time

start_time_test = time.time()
X = data_test.iloc[:, :-1]
Y = data_test.iloc[:, -1]

# split Review_Date
splitDate = split_date(data_test)

# split Tags
splitTags = split_Tags(splitDate)

# fill null in training set
data_fill_null = fill_Null_test(splitTags)

# label encoder to convert data to numerical
X, Y = Feature_Encoder(data_fill_null)

X = Standardization(X)

# load Saved the selected feature names to a file
with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)
X = X[selected_features]

###########################################################################
# Models
# multi linear regression
# Load the saved model from a file
start_T_linear = time.time()
with open('multi linear regression.pkl', 'rb') as f:
    model = pickle.load(f)
y_pred = model.predict(X)
mse = mean_squared_error(Y, y_pred)
print(f"Mean squared error: {mse}")

# calculate the accuarcy of regression
from sklearn.metrics import r2_score

r2 = r2_score(Y, y_pred)
acc_linear = round(r2, 2) * 100
print(f"accuracy of multiple  linear model = {acc_linear}% ")
end_T_linear = time.time()
time_predict_of_linear = (end_T_linear - start_T_linear) / 60
####################################################################
# polynomial regression model
start_T_poly = time.time()
with open('polynomial regression.pkl', 'rb') as f:
    model = pickle.load(f)
with open('poly_features.pkl', 'rb') as f:
    poly = pickle.load(f)

x_test_poly = poly.transform(X)
y_pred = model.predict(x_test_poly)

mse = mean_squared_error(Y, y_pred)
r2 = r2_score(Y, y_pred)
acc_poly = round(r2, 2) * 100

print('*' * 50)
print("Mean Squared Error of polynomial :", mse)
print("Accuracy of polynomial:", acc_poly, '%')
end_T_poly = time.time()
time_predict_of_poly = (end_T_poly - start_T_poly) / 60
######################################################################
# lasso regression
start_T_lasso = time.time()
with open('lasso regression.pkl', 'rb') as f:
    lasso = pickle.load(f)

# make predictions on the testing data
y_pred = lasso.predict(X)

# evaluate the performance of the model
mse = mean_squared_error(Y, y_pred)
print('*' * 50)
print("Mean squared error of lasso regression:", mse)
# accuracy of lasso
r2 = r2_score(Y, y_pred)
acc_lasso = round(r2, 2) * 100
print(f"accuracy of lasso  model = {acc_lasso}% ")
end_T_lasso = time.time()
time_predict_of_lasso = (end_T_lasso - start_T_lasso) / 60
###################################################################
# ridge regression model
start_T_redge = time.time()
with open('ridge regression.pkl', 'rb') as f:
    ridge = pickle.load(f)

# make predictions on the testing data
y_pred = ridge.predict(X)

# evaluate the performance of the model
mse = mean_squared_error(Y, y_pred)
print('*' * 50)
print("Mean squared error of ridge model:", mse)
# accuracy of lasso
r2 = r2_score(Y, y_pred)
acc_ridge = round(r2, 2) * 100
print(f"accuracy of ridge model = {acc_ridge}% ")
end_T_redge = time.time()
time_predict_of_redge = (end_T_redge - start_T_redge) / 60
###################################################################
# the ElasticNet regression model
start_T_elastice= time.time()
with open('ElasticNet regression.pkl', 'rb') as f:
    elasticnet = pickle.load(f)

# make predictions on the testing data
y_pred = elasticnet.predict(X)

# evaluate the performance of the model
mse = mean_squared_error(Y, y_pred)
print('*' * 50)
print("Mean squared error of ElasticNet model:", mse)
# accuracy of ElasticNet regression model
r2 = r2_score(Y, y_pred)
acc_elas = round(r2, 2) * 100
print(f"accuracy of ElasticNet  model = {acc_elas}% ")
end_T_elastice= time.time()
time_predict_of_elastice = (end_T_elastice - start_T_elastice) / 60
########################################################################
# Random Forest object
start_T_random = time.time()
with open('Random Forest object.pkl', 'rb') as f:
    rf = pickle.load(f)

y_pred = rf.predict(X)
# evaluate the performance of the model
mse = mean_squared_error(Y, y_pred)
print('*' * 50)
print("Mean squared error of RandomForestRegressor model:", mse)
# accuracy of ElasticNet regression model
r2 = r2_score(Y, y_pred)
acc_random = round(r2, 2) * 100
print(f"accuracy of RandomForestRegressor  model = {acc_random}% ")
end_T_random = time.time()
time_predict_of_random = (end_T_random - start_T_random) / 60
#######################################################
end_time_test = time.time()
time_test = (end_time_test - start_time_test) / 60
print('-' * 20)
print("time of test ==> ", time_test, "minutes")
#########################################################################
# summarize total accuracy of all models
All_accuracy = {'Linear': acc_linear,  'poly': acc_poly,
                'lasso': acc_lasso, 'ridge': acc_ridge, 'elastic': acc_elas,
                'random': acc_random}
accuracy = list(All_accuracy.keys())
values = list(All_accuracy.values())

fig = plt.figure(figsize=(10, 8))

# creating the bar plot
plt.bar(accuracy, values, color='maroon', width=0.2)

plt.xlabel("Accuracy of each model")
plt.ylabel("%")
plt.title("Summarize Total Accuracy")
plt.show()
##########################################################################
# time of train
All_time = {'Linear': time_predict_of_linear,  'Poly': time_predict_of_poly,
            'Lasso': time_predict_of_lasso, 'Redge': time_predict_of_redge, 'Elastic': time_predict_of_elastice,
            'Random': time_predict_of_random, 'Total Time Test': time_test}
Time = list(All_time.keys())
values = list(All_time.values())

fig = plt.figure(figsize=(10, 8))

# creating the bar plot
plt.bar(Time, values, color='maroon', width=0.2)

plt.xlabel("Time of Test")
plt.ylabel("time in minutes")
plt.title("Total Time Test")
plt.show()