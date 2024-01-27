import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from preprocessing_Data import *
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
hotel_data = pd.read_csv('hotel-classification-dataset.csv', na_values=["No Negative", "No Positive"])
X = hotel_data.iloc[:, :-1]
Y = hotel_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# training data that learning models
data = pd.concat([X_train, y_train], axis=1)
# testing set that will be hidden
data_test = pd.concat([X_test, y_test], axis=1)

# fill null in training set
data_fill_null = fill_Null_train_classi(data)

# split Review_Date
splitDate = split_date(data_fill_null)

# split Tags
data_processed = split_Tags(splitDate)

# label encoder to convert data to numerical
X, Y = Feature_Encoder_classi(data_processed)

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

# BoxPlot
# print("length of data: ", len(df_merged))
# plot_boxplot(df_merged, 22)

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
print(selected_features)
X = X[selected_features]
with open('selected_features_classi.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

# -----------------------------------------------------------------------------------------------

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

XX = data_test.iloc[:, :-1]
YY = data_test.iloc[:, -1]

# fill null in training set
data_fill_null = fill_Null_test_classi(data_test)

# split Review_Date
splitDate = split_date(data_fill_null)

# split Tags
data_processed = split_Tags(splitDate)

# label encoder to convert data to numerical
XX, YY = Feature_Encoder_classi(data_processed)

XX = Standardization(X)

# load Saved the selected feature names to a file
with open('selected_features_classi.pkl', 'rb') as f:
    selected_features = pickle.load(f)
XX = XX[selected_features]

####################################################################################################################
# Models
# SVM model
# svm_model = SVC(C=.1, kernel='linear', gamma=1)
svm_model = SVC(kernel='linear', random_state=0)
svm_model.fit(X, Y)
with open('SVM model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)


# SVM model
with open('SVM model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
Y_pred = svm_model .predict(XX)
# Accuracy = accuracy_score(YY, Y_pred)

# print('Accuracy of SVM model: ', Accuracy*100, '%')

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator=svm_model, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X, Y)

accuracy = grid_search.best_score_

print('acc', accuracy)
print(grid_search.best_params_)
