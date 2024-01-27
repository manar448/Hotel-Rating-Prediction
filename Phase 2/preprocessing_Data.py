from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import pickle

def fill_Null_train_classi(data):
    # Negative_Review, Positive_Review, Review_Total_Positive_Word_Counts,
    # Total_Number_of_Reviews_Reviewer_Has_Given, Tags, days_since_review,
    # lat, lng, Reviewer_Score ----> contains a missing values
    mask_F = data.isnull().any(axis=0)

    # missing values in each row (any => if this row contain a missing value return true)
    mask_R = data.isnull().any(axis=1)
    # percentage of rows,features with nan => (num_of_rows_with_nan / total_num_of_rows)
    P_rows_with_nan = mask_R.sum() / len(data)
    P_features_with_nan = mask_F.sum() / len(data)
    # percentage in each column with nan
    P_in_each_column_with_nan = data.isnull().sum() / len(data)
    # if (P_rows_with_nan < 0.07) drop all rows with nan values
    if P_rows_with_nan < 0.07:
        data = data[~mask_R]
    else:
        X3 = data['Average_Score'].mean()
        X8 = data['Total_Number_of_Reviews'].mean()
        X14 = data['lat'].mean()
        X15 = data['lng'].mean()
        # most frequently value in 'Reviewer_Score' column
        # X16 = data['Reviewer_Score'].mode()[0]
        X16 = 'Low_Reviewer_Score'
        # 'Intermediate_Reviewer_Score' 'High_Reviewer_Score' 'Low_Reviewer_Score'
        # data with filling missing values
        data = data.fillna({'Hotel_Address': 'no address', 'Additional_Number_of_Scoring': 0.0,
                            'Review_Date': "00/00/0000", 'Average_Score': X3, 'Hotel_Name': 'no name',
                            'Reviewer_Nationality': 'no info', 'Negative_Review': 'positive',
                            'Review_Total_Negative_Word_Counts': 0.0, 'Total_Number_of_Reviews': X8,
                            'Positive_Review': 'negative', 'Review_Total_Positive_Word_Counts': 0.0,
                            'Total_Number_of_Reviews_Reviewer_Has_Given': 0.0,
                            'Tags': "[' no trip ', ' no Couple  ', ' no Room ', ' Stayed 0 ', 'not Submitted']",
                            'days_since_review': '0 days', 'lat': X14, 'lng': X15, 'Reviewer_Score': X16})
    # value that will restore in file
    values = {'Average_Score': X3, 'Total_Number_of_Reviews': X8, 'lat': X14, 'lng': X15, 'Reviewer_Score': X16}
    # pickle
    with open('valuesOfClassi', mode='bw') as f:
        pickle.dump(obj=values, file=f)
    return data


#########################################################################
def fill_Null_test_classi(data):
    # load store values from file
    with open('valuesOfClassi', mode='br') as f:
        values = pickle.load(f)
    # Negative_Review, Positive_Review, Review_Total_Positive_Word_Counts,
    # Total_Number_of_Reviews_Reviewer_Has_Given, Tags, days_since_review,
    # lat, lng, Reviewer_Score ----> contains a missing values
    mask_F = data.isnull().any(axis=0)

    # missing values in each row (any => if this row contain a missing value return true)
    mask_R = data.isnull().any(axis=1)
    # percentage of rows,features with nan => (num_of_rows_with_nan / total_num_of_rows)
    P_rows_with_nan = mask_R.sum() / len(data)
    P_features_with_nan = mask_F.sum() / len(data)
    # percentage in each column with nan
    P_in_each_column_with_nan = data.isnull().sum() / len(data)
    # if (P_rows_with_nan < 0.07) drop all rows with nan values
    if P_rows_with_nan < 0.07:
        data = data[~mask_R]
    else:
        X3 = data['Average_Score'].mean()
        X8 = data['Total_Number_of_Reviews'].mean()
        X14 = data['lat'].mean()
        X15 = data['lng'].mean()
        # most frequently value in 'Reviewer_Score' column
        # X16 = data['Reviewer_Score'].mode()[0]
        X16 = 'Low_Reviewer_Score'
        # data with filling missing values
        data = data.fillna({'Hotel_Address': 'no address', 'Additional_Number_of_Scoring': 0.0,
                            'Review_Date': "00/00/0000", 'Average_Score': X3, 'Hotel_Name': 'no name',
                            'Reviewer_Nationality': 'no info', 'Negative_Review': 'positive',
                            'Review_Total_Negative_Word_Counts': 0.0, 'Total_Number_of_Reviews': X8,
                            'Positive_Review': 'negative', 'Review_Total_Positive_Word_Counts': 0.0,
                            'Total_Number_of_Reviews_Reviewer_Has_Given': 0.0,
                            'Tags': "[' no trip ', ' no Couple  ', ' no Room ', ' Stayed 0 ', 'not Submitted']",
                            'days_since_review': '0 days', 'lat': X14, 'lng': X15, 'Reviewer_Score': X16})
    # value that will restore in file
    values = {'Average_Score': X3, 'Total_Number_of_Reviews': X8, 'lat': X14, 'lng': X15, 'Reviewer_Score': X16}

    return data
#########################################################################
# date handling
def split_date(data):
    # split Review_Date
    df = pd.DataFrame()
    df[["day", "month", "year"]] = data['Review_Date'].str.split(r'\D', expand=True)
    data['Day'] = df['day'].astype(str).astype(int)
    data['Month'] = df['month'].astype(str).astype(int)
    data['Year'] = df['year'].astype(str).astype(int)
    data = data.drop(["Review_Date"], axis=1)

    # split days_since_review
    days = list(data['days_since_review'])
    day = []
    for i in range(len(days)):
        days[i] = str(days[i]).replace("days", "")
        days[i] = str(days[i]).replace("day", "")
        day.append(days[i])
    res = [eval(i) for i in day]
    data['days_since_review'] = res
    return data


#######################################
# split tags and night
def split_Tags(data):
    # split Tags
    Trip = []
    Members = []
    Room_Kind = []
    Nights = []
    the_way_of_submission = []
    Tags = data['Tags'].tolist()

    for i in Tags:
        tag = []
        i = i.replace("[", "")
        i = i.replace("]", "")
        i = i.replace("'", "")
        tag.append(i.split(","))
        for j in tag:
            trip = False
            mem = False
            room = False
            night = False
            submission = False
            for k in j:
                if k.__contains__("trip"):
                    Trip.append(k)
                    trip = True
                elif k.__contains__('room') or k.__contains__('Room'):
                    Room_Kind.append(k)
                    room = True
                elif k.__contains__('Stayed'):
                    Nights.append(k)
                    night = True
                elif k.__contains__('Submitted'):
                    the_way_of_submission.append(k)
                    submission = True
                elif k.__contains__('Couple') or k.__contains__('Group') or k.__contains__(
                        'children') or k.__contains__('traveler'):
                    Members.append(k)
                    mem = True
            if not trip:
                Trip.append("no trip")
            if not room:
                Room_Kind.append("no Room")
            if not night:
                Nights.append("Stayed 0")
            if not submission:
                the_way_of_submission.append("not Submitted")
            if not mem:
                Members.append("no Couple")
    data['Trip'] = Trip
    data['Members'] = Members
    data['Room'] = Room_Kind
    data['Submission'] = the_way_of_submission

    # split Nights
    numbers = []
    for char in range(len(Nights)):
        Nights[char] = [x for x in Nights[char].split() if x.isdigit()]
        converted_num = Nights[char]
        convert = int(converted_num[0])
        numbers.append(convert)
    data['Nights'] = numbers
    data = data.drop(["Tags"], axis=1)
    return data


############################################################
# feature encoder
def Feature_Encoder_classi(data):
    # label encoder to convert data to numerical
    le = preprocessing.LabelEncoder()

    data['Hotel_Address'] = le.fit_transform(data['Hotel_Address'])
    data['Hotel_Name'] = le.fit_transform(data['Hotel_Name'])
    data['Reviewer_Nationality'] = le.fit_transform(data['Reviewer_Nationality'])
    data['Trip'] = le.fit_transform(data['Trip'])
    data['Members'] = le.fit_transform(data['Members'])
    data['Room'] = le.fit_transform(data['Room'])
    data['Submission'] = le.fit_transform(data['Submission'])
    data['Negative_Review'] = le.fit_transform(data['Negative_Review'])
    data['Positive_Review'] = le.fit_transform(data['Positive_Review'])
    data['Reviewer_Score'] = le.fit_transform(data['Reviewer_Score'])

    # data_num --> data after convert
    data_num = data.iloc[:, [0, 1, 15, 16, 17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 11, 12, 13, 14]]

    X = data_num.iloc[:, 0:22]
    Y = data_num.iloc[:, 22]

    return X, Y
#######################################################################################
# Feature Scaling
# 1- Normalize Data
def Normalization(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    Normalized_X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    return Normalized_X


# 2- standardization
def Standardization(X):
    # Standardization
    scaler = StandardScaler()
    scaler.fit(X)
    # calc mean and stadard devision
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    return X_scaled


################################################################
# BoxPlot
def plot_boxplot(df, ft):
    df.boxplot(column=[ft])
    plt.grid(False)
    plt.show()


##############################################
# outliers
# delete outliers data
def delete_outlier(data, name, upper_limit, lower_limit):
    update_data = data.loc[(data[name] <= upper_limit) & (data[name] >= lower_limit)]
    return update_data


#######################################################
# change values to upper or lower
def Norm_outlier(data, name, upper_limit, lower_limit):
    new_data = data.copy()
    new_data.loc[data[name] > upper_limit, name] = upper_limit
    new_data.loc[data[name] < lower_limit, name] = lower_limit
    return new_data


###########################################################
# IQR Method
def Outliers(data):
    length = len(data.columns)
    for i in range(length):
        name = data.columns[i]
        q1 = data[name].quantile(0.25)
        q3 = data[name].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr

        data = Norm_outlier(data, name, upper_limit, lower_limit)

    return data


############################################################
# method2 : z-score
def z_score(data):
    n = len(data.columns)
    for i in range(n):
        name = data.columns[i]
        upper_limit = data[name].mean() + 3 * data[name].std()
        lower_limit = data[name].mean() - 3 * data[name].std()

        data = Norm_outlier(data, name, upper_limit, lower_limit)
    return data
