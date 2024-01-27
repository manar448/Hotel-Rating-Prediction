import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from preprocessing_Data import *
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import time

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import warnings
import random

random.seed(42)
warnings.filterwarnings("ignore")

start_time_train = time.time()

hotel_data = pd.read_csv('hotel-tas-test-classification.csv', na_values=["No Negative", "No Positive"])
# hotel_data = pd.read_csv('hotel-classification-dataset.csv', na_values=["No Negative", "No Positive"])
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
data_cleaned = Outliers(df_merged)
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


####################################################################################################################
# Models
# logistic regression model
start_fT_logistic = time.time()
Logistic = LogisticRegression(random_state=0, C=0.1, penalty='none')
Logistic.fit(X, Y)

# Save the  logistic regression model to a file
with open('logistic regression model.pkl', 'wb') as f:
    pickle.dump(Logistic, f)

# grid      #"C": np.logspace(-3, 3, 7),
grid_log = {
    "C": np.logspace(-3, 3, 7),
    "penalty": ["l1", "l2"]
}  # l1 lasso l2 ridge
# grid={"C":np.logspace(-3,3,20), "penalty":["l2"]}
# grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
logistic = LogisticRegression()
# logreg_cv = GridSearchCV(logistic, grid, cv=10)
logreg_cv = GridSearchCV(logistic, grid_log, cv=5)
logreg_cv.fit(X, Y)

# Save the  logistic regression model after use gride search to a file
with open('logistic regression model by grid.pkl', 'wb') as f:
    pickle.dump(logreg_cv, f)
end_fT_logistic = time.time()
time_fit_of_logitic = (end_fT_logistic - start_fT_logistic) / 60
########################################
# naive bayes model
start_fT_naive = time.time()
naive_bayes = GaussianNB()
naive_bayes.fit(X, Y)
# Save naive bayes model to a file
with open('naive bayes model.pkl', 'wb') as f:
    pickle.dump(naive_bayes, f)
end_fT_naive = time.time()
time_fit_of_naive = (end_fT_naive - start_fT_naive) / 60
#######################################
# KNN model
knn = KNeighborsClassifier(n_neighbors=11, p=2, metric='manhattan', weights='distance')
start_fT_Knn = time.time()
knnn = KNeighborsClassifier()
knnn.fit(X, Y)
with open('KNN model.pkl', 'wb') as f:
    pickle.dump(knnn, f)

# grid
# Define the hyperparameter grid to search over    #[3, 5, 7]
param_grid_Knn = {'n_neighbors': [7, 9, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}

# Create a KNN classifier object
knn = KNeighborsClassifier()

# Create a GridSearchCV object to search over the hyperparameter grid
grid_search_KNN = GridSearchCV(knn, param_grid_Knn, cv=5)

# Fit the GridSearchCV object to the data
grid_search_KNN.fit(X, Y)

# Save the  logistic regression model after use gride search to a file
with open('KNN model by grid.pkl', 'wb') as f:
    pickle.dump(grid_search_KNN, f)
end_fT_Knn = time.time()
time_fit_of_Knn = (end_fT_Knn - start_fT_Knn) / 60
#######################################
# Desicion Tree Model
start_fT_tree = time.time()
DT = DecisionTreeClassifier(max_depth=8, min_samples_leaf=1, min_samples_split=2)
DT.fit(X, Y)
with open('Decision Tree model.pkl', 'wb') as f:
    pickle.dump(DT, f)

# grid
param_grid_DT = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}

# Create a decision tree classifier
DT_grid = DecisionTreeClassifier()

# Perform the grid search to find the optimal hyperparameters
grid_search_DT = GridSearchCV(DT_grid, param_grid=param_grid_DT, cv=5)
grid_search_DT.fit(X, Y)

with open('Decision Tree model grid.pkl', 'wb') as f:
    pickle.dump(grid_search_DT, f)

end_fT_tree = time.time()
time_fit_of_DT = (end_fT_tree - start_fT_tree) / 60
#######################################
# Random forest model
start_fT_Random = time.time()
regressor = RandomForestClassifier(n_estimators=100, max_depth=12)
regressor = RandomForestClassifier()
regressor.fit(X, Y)
with open('Random forest model.pkl', 'wb') as f:
    pickle.dump(regressor, f)
end_fT_Random = time.time()
time_fit_of_Random = (end_fT_Random - start_fT_Random) / 60
# #######################################
# SVM model
# kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']
start_fT_svm = time.time()
svm_model = SVC(C=10, kernel='linear', gamma=1)
svm_model.fit(X, Y)
with open('SVM model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
end_fT_svm = time.time()
time_fit_of_svm = (end_fT_svm - start_fT_svm) / 60
# #######################################
# Gradient Boosting model
start_fT_gradient = time.time()
gbc = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=100, max_features=5)
gbc.fit(X, Y)
with open('Gradient Boosting model.pkl', 'wb') as f:
    pickle.dump(gbc, f)
end_fT_gradient = time.time()
time_fit_of_gradient = (end_fT_gradient - start_fT_gradient) / 60
# ###############################################
end_time_train = time.time()
time_train = (end_time_train - start_time_train) / 60
# time of train
All_time = {'Logistic': time_fit_of_logitic, 'Naive': time_fit_of_naive,
            'KNN': time_fit_of_Knn, 'Decision': time_fit_of_DT, 'Random': time_fit_of_Random,
            'SVM': time_fit_of_svm, 'Gradient': time_fit_of_gradient, 'Total Time Train': time_train}
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