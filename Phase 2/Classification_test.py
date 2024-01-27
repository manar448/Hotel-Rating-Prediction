import pandas as pd
from sklearn.model_selection import GridSearchCV

from preprocessing_Data import *
from Classification_train import data_test
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn import metrics
import time
from matplotlib import pyplot as plt
import random
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import warnings

random.seed(42)
warnings.filterwarnings("ignore")

start_time_test = time.time()
X = data_test.iloc[:, :-1]
Y = data_test.iloc[:, -1]

# fill null in training set
data_fill_null = fill_Null_test_classi(data_test)

# split Review_Date
splitDate = split_date(data_fill_null)

# split Tags
data_processed = split_Tags(splitDate)

# label encoder to convert data to numerical
X, Y = Feature_Encoder_classi(data_processed)

X = Standardization(X)

# load Saved the selected feature names to a file
with open('selected_features_classi.pkl', 'rb') as f:
    selected_features = pickle.load(f)
X = X[selected_features]

###########################################################################
# Models
# logistic regression model
start_T_log = time.time()
with open('logistic regression model.pkl', 'rb') as f:
    Logistic = pickle.load(f)
y_pred_log = Logistic.predict(X)
acc_of_logistic = accuracy_score(y_pred_log, Y) * 100
print('Accuracy of logistic regression model: ', acc_of_logistic, '%')

with open('logistic regression model by grid.pkl', 'rb') as f:
    logreg_cv = pickle.load(f)

y_pred_log_G = logreg_cv.best_estimator_.predict(X)
acc_log_grid = accuracy_score(Y, y_pred_log_G) * 100

print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
print("log Accuracy with the best model:", acc_log_grid, '%')
end_T_log = time.time()
time_predict_of_logistic = (end_T_log - start_T_log) / 60
print("-------------------------------------------------------------------")

########################################
# naive bayes model
start_T_naive = time.time()
with open('naive bayes model.pkl', 'rb') as f:
    naive_bayes = pickle.load(f)
y_pred = naive_bayes.predict(X)
# cv_scores = cross_val_score(model, X, Y, cv=5)
#
# print(model, ' mean accuracy: ', round(cv_scores.mean() * 100, 3), '% std: ', round(cv_scores.var() * 100, 3), '%')
accuracy_naive = metrics.accuracy_score(y_pred, Y)
print('Accuracy of naive bayes model: ', accuracy_naive * 100, '%')
print("-------------------------------------------------------------------")
end_T_naive = time.time()
time_predict_of_naive = (end_T_naive - start_T_naive) / 60
#######################################
# KNN model
start_T_Knn = time.time()
with open('KNN model.pkl', 'rb') as f:
    knnn = pickle.load(f)

y_pred_Kn = knnn.predict(X)
accuracy_Knn = accuracy_score(y_pred_Kn, Y) * 100
print('accuracy of KNN model = ', accuracy_Knn, '%')

with open('KNN model by grid.pkl', 'rb') as f:
    grid_search_KNN = pickle.load(f)

# Get the best KNN model from the GridSearchCV object
y_pred_KN_G = grid_search_KNN.best_estimator_.predict(X)
acc_Knn_grid = accuracy_score(Y, y_pred_KN_G) * 100
print("Tuned hyper parameters (best parameters):", grid_search_KNN.best_params_)
print("Accuracy of KNN with the best model :", acc_Knn_grid, '%')
end_T_Knn = time.time()
Time_predict_of_Knn = (end_T_Knn - start_T_Knn) / 60
print("-------------------------------------------------------------------")
#######################################
# Desicion Tree Model
start_T_tree = time.time()
with open('Decision Tree model.pkl', 'rb') as f:
    DT = pickle.load(f)
y_pred_T = DT.predict(X)
accuracy_DT = accuracy_score(Y, y_pred_T) * 100

print("Accuracy of Decision Tree model:", accuracy_DT, '%')
# grid
with open('Decision Tree model grid.pkl', 'rb') as f:
    grid_search_DT = pickle.load(f)

y_pred_DT_G = grid_search_DT.best_estimator_.predict(X)
accuracy_DT_grid = accuracy_score(Y, y_pred_DT_G) * 100
print("Tuned hyper parameters (best parameters):", grid_search_DT.best_params_)
print("Accuracy of Decision Tree with the best model :", accuracy_DT_grid)

end_T_tree = time.time()
time_predict_of_DT = (end_T_tree - start_T_tree) / 60
print("-------------------------------------------------------------------")
# #######################################
# Random forest model
start_T_random = time.time()
with open('Random forest model.pkl', 'rb') as f:
    regressor = pickle.load(f)
Y_pred = regressor.predict(X)
acc_random = accuracy_score(Y_pred, Y)
print('-' * 20)
print('accuracy of RandomForestRegressor model = ', acc_random * 100, '%')
print("-------------------------------------------------------------------")
end_T_random = time.time()
time_predict_of_random = (end_T_random - start_T_random) / 60
# #######################################
# SVM model
start_T_svm = time.time()
with open('SVM model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
Y_pred = svm_model.predict(X)
accuracy_svm = accuracy_score(Y_pred, Y)
print('-' * 20)
print('Accuracy of SVM model: ', accuracy_svm * 100, '%')
print("-------------------------------------------------------------------")
end_T_svm = time.time()
time_predict_of_svm = (end_T_svm - start_T_svm) / 60
# #######################################
# Gradient Boosting model
start_T_gradient = time.time()
with open('Gradient Boosting model.pkl', 'rb') as f:
    gbc = pickle.load(f)
Y_pred = gbc.predict(X)
acc_gradient = accuracy_score(Y_pred, Y)
print('-' * 20)
print('Accuracy of Gradient Boosting Classifier', acc_gradient * 100, '%')
print("-------------------------------------------------------------------")
end_T_gradient = time.time()
time_predict_of_gradient = (end_T_gradient - start_T_gradient) / 60
# #######################################
# bar graph of accuracy with grid search of each model
# set width of bar
barWidth = 0.1
fig = plt.subplots(figsize=(8, 9))

# set height of bar
b1 = [acc_of_logistic, accuracy_Knn, accuracy_DT]
b2 = [acc_log_grid, acc_Knn_grid, accuracy_DT_grid]

# Set position of bar on X axis
br1 = np.arange(len(b1))
br2 = [x + barWidth for x in br1]

# Make the plot
plt.bar(br1, b1, width=barWidth,
        edgecolor='grey', label='accuracy')
plt.bar(br2, b2, color='grey', width=barWidth,
        edgecolor='grey', label='accuracy after update')

# Adding Xticks
plt.xlabel('Accuracy', fontweight='bold', fontsize=15)
plt.ylabel('Summarize Accuracy with Grid search', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(b1))],
           ['Logistic Regression', 'KNN', 'Decision Tree'])

plt.legend()
plt.show()
##################################################################
end_time_test = time.time()
time_test = (end_time_test - start_time_test) / 60
print('-' * 20)
print("time of test ==> ", time_test, "minutes")
#########################################################################
# summarize total accuracy of all models
All_accuracy = {'Logistic': acc_of_logistic,  'Naive': accuracy_naive,
                'Knn': accuracy_Knn, 'Decision': accuracy_DT, 'Random': acc_random,
                'SVM': accuracy_svm, 'Gradient':acc_gradient}
accuracy = list(All_accuracy.keys())
values = list(All_accuracy.values())

fig = plt.figure(figsize=(10, 8))

# creating the bar plot
plt.bar(accuracy, values, color='maroon', width=0.2)

plt.xlabel("Accuracy of each model")
plt.ylabel("%")
plt.title("Summarize Total Accuracy")
##########################################################################
# time of train
All_time = {'Logistic': time_predict_of_logistic,  'Naive': time_predict_of_naive,
            'Knn': Time_predict_of_Knn, 'Decision': time_predict_of_DT, 'Random': time_predict_of_random,
            'SVM': time_predict_of_svm, 'Gradient': time_predict_of_gradient, 'Total Time Test': time_test}
Time = list(All_time.keys())
values = list(All_time.values())

fig = plt.figure(figsize=(10, 8))

# creating the bar plot
plt.bar(Time, values, color='maroon', width=0.2)

plt.xlabel("Time of Test")
plt.ylabel("time in minutes")
plt.title("Total Time Test")
