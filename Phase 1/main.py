# train model
from datetime import time

import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, r2_score
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from preprocessing_Data import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time

start_time_train = time.time()
hotel_data = pd.read_csv('hotel-tas-test-regression.csv', na_values=["No Negative", "No Positive"])
# hotel_data = pd.read_csv('hotel-regression-dataset.csv', na_values=["No Negative", "No Positive"])
X = hotel_data.iloc[:, :-1]
Y = hotel_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# training data that learing models
data = pd.concat([X_train, y_train], axis=1)
# testing set that will be hidden
data_test = pd.concat([X_test, y_test], axis=1)
# split Review_Date
splitDate = split_date(data)

# split Tags
splitTags = split_Tags(splitDate)

# fill null in training set
data_fill_null = fill_Null_train(splitTags)

# label encoder to convert data to numerical
X, Y = Feature_Encoder(data_fill_null)

# Feature Scaling
X = Standardization(X)

# data after scaling
dff = pd.DataFrame(X)
dff1 = pd.DataFrame(Y)
# Reset the indices of the two data frames
dff = dff.reset_index(drop=True)
dff1 = dff1.reset_index(drop=True)
frame = [dff, dff1]
df_merged = pd.concat(frame, axis=1)


# data after outliers
data_cleaned = z_score(df_merged)
# print("length of data after outlier: ", len(data_cleaned))
X = data_cleaned.iloc[:, :-1]
Y = data_cleaned.iloc[:, -1]
# plot_boxplot(data_cleaned, 22)

# Feature Selection from chi2
fs = SelectKBest(score_func=f_classif, k=7)
# fs = SelectKBest(score_func=mutual_info_classif, k=4)
fs.fit(X, Y.values.ravel())
selected_features = X.columns[fs.get_support()]
print("selected features ==> ", selected_features)
X = X[selected_features]
with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

####################################################################################################################
# Models
# multi linear regression
start_fT_linear = time.time()
model = LinearRegression()
model.fit(X, Y)
# Save the  multi linear regression model to a file
with open('multi linear regression.pkl', 'wb') as f:
    pickle.dump(model, f)
end_fT_linear = time.time()
time_fit_of_linear = (end_fT_linear - start_fT_linear) / 60
###############################
# polynomial regression model
start_fT_poly = time.time()
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(X)

# fit the transformed input data to a linear regression model
model = LinearRegression()
model.fit(x_poly, Y)

# Save the  polynomial regression model to a file
with open('polynomial regression.pkl', 'wb') as f:
    pickle.dump(model, f)
# save poly feature
with open('poly_features.pkl', 'wb') as f:
    pickle.dump(poly, f)
end_fT_poly = time.time()
time_fit_of_poly = (end_fT_poly - start_fT_poly) / 60
##################################################
# lasso regression
start_fT_lasso = time.time()
lasso = Lasso(alpha=0.1)
lasso.fit(X, Y)

with open('lasso regression.pkl', 'wb') as f:
    pickle.dump(lasso, f)
end_fT_lasso = time.time()
time_fit_of_lasso = (end_fT_lasso - start_fT_lasso) / 60
#####################################
# ridge regression model
# create and train the Ridge regression model
start_fT_redge = time.time()
ridge = Ridge(alpha=0.1)
ridge.fit(X, Y)

with open('ridge regression.pkl', 'wb') as f:
    pickle.dump(ridge, f)
end_fT_redge = time.time()
time_fit_of_redge = (end_fT_redge - start_fT_redge) / 60
########################################
# the ElasticNet regression model
start_fT_elastice= time.time()
elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5)
elasticnet.fit(X, Y)

with open('ElasticNet regression.pkl', 'wb') as f:
    pickle.dump(elasticnet, f)
end_fT_elastice= time.time()
time_fit_of_elastice = (end_fT_elastice - start_fT_elastice) / 60
########################################################
from sklearn.ensemble import RandomForestRegressor

# Random Forest object
# rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
start_fT_random = time.time()
rf = RandomForestRegressor(n_estimators=100, max_depth=8)
rf.fit(X, Y)

with open('Random Forest object.pkl', 'wb') as f:
    pickle.dump(rf, f)
end_fT_random = time.time()
time_fit_of_random = (end_fT_random - start_fT_random) / 60
#####################################################
end_time_train = time.time()
time_train = (end_time_train - start_time_train) / 60
# time of train
All_time = {'Linear': time_fit_of_linear, 'polnomial': time_fit_of_poly,
            'KNN': time_fit_of_lasso, 'Decision': time_fit_of_redge, 'Random': time_fit_of_elastice,
            'SVM': time_fit_of_random, 'Total Time Train': time_train}
Time = list(All_time.keys())
values = list(All_time.values())

fig = plt.figure(figsize=(10, 5))
# creating the bar plot
plt.bar(Time, values, color='maroon', width=0.2)

plt.xlabel("Time of Train")
plt.ylabel("time in minutes")
plt.title("Total Time Train")

print("time of train ==> ", time_train, "minutes")
print("classification train is done")
print("-------------------------------------------------------------------")
