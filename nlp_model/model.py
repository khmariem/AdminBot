import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def read_data(path):
    data_path = path
    data = pd.read_csv(data_path)

    return data

def read_process_data(path):
    data = read_data(path)
    labels = data.iloc[:,0].values
    features = data.iloc[:,1].values

    processed_data = []

    for sentence in range(0, len(features)):
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
        processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
        processed_feature = re.sub(r'^b\s+', '', processed_feature)

        processed_data.append(processed_feature.lower())

    return processed_data, labels

def random_forest_classifier():

    return RandomForestClassifier(n_estimators=200, random_state=0)

def svm_classifier():
    pass

def logistic_regression_classifier():
    pass

def knn_classifier():
    pass

def eval(y_test, predictions):

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    print(accuracy_score(y_test, predictions))

def model(path,eval_crit=0):
    stop = ["manual","numbers","email","ip","dns","stustanet","stusta","room","laptop","pc","computer","internet","connection","minutes","online","wifi","surf"]
    data, labels = read_process_data(path)
    vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stop)
    processed_features = vectorizer.fit_transform(data).toarray()
    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
    text_classifier = random_forest_classifier()
    text_classifier.fit(X_train, y_train)

    if eval_crit==1:
        predictions = text_classifier.predict(X_test)
        print(predictions)
        eval(y_test, predictions)
    
    return text_classifier, vectorizer

if __name__ == "__main__":
    eval_crit = 1
    classifier, vec = model('../data/data.csv',eval_crit)

    processed_msg = vec.transform(["yay the problem is solved"]).toarray()
    prediction = classifier.predict(processed_msg)
    print(prediction)