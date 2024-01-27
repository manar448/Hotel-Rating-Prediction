import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

data = pd.read_csv('hotel-regression-dataset.csv')

x1 = data['Positive_Review']
x2 = data['Negative_Review']


def preprocess_text(text):
    text = text.lower()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    preprocessed_text = [token for token in lemmatized_tokens if token.isalpha()]
    return preprocessed_text


x1 = data['positive_reviews_preprocessed'] = data['Positive_Review'].apply(preprocess_text)
x2 = data['negative_reviews_preprocessed'] = data['Negative_Review'].apply(preprocess_text)
print(x1,x2)


# create a bag-of-words representation of the preprocessed reviews
vectorizer = CountVectorizer()
X1 = vectorizer.fit_transform(data['positive_reviews_preprocessed'].apply(lambda x: ' '.join(x)))
X2 = vectorizer.transform(data['negative_reviews_preprocessed'].apply(lambda x: ' '.join(x)))

# create the target variable
y = np.concatenate([np.ones(len(X1)), np.zeros(len(X2))], axis=0)

# split the data into training and test sets
X = np.concatenate((X1.toarray(), X2.toarray()), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# evaluate the classifier on the test set
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))