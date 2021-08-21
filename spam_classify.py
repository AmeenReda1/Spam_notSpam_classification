import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import sklearn.feature_extraction.text as text
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
#Read the data
Email_Data = pd.read_csv("spam.csv",encoding ='latin1')
#Data undestanding
#print(Email_Data.columns)
Email_Data = Email_Data[['v1', 'v2']]
Email_Data = Email_Data.rename(columns={"v1":"Target","v2":"Email"})
#print(Email_Data.head())
#pre processing steps like lower case, stemming and lemmatization
Email_Data['Email'] = Email_Data['Email'].apply(lambda line:" ".join(word.lower() for word in line.split()))
stop = stopwords.words('english')
Email_Data['Email'] = Email_Data['Email'].apply(lambda line: " ".join(word for word in line.split() if word not in stop))
st = PorterStemmer()
Email_Data['Email'] = Email_Data['Email'].apply(lambda line: " ".join([st.stem(word) for word in line.split()]))
Email_Data['Email'] = Email_Data['Email'].apply(lambda line: " ".join([Word(word).lemmatize() for word in line.split()]))
#print(Email_Data.head())
#Splitting data into train and validation
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(Email_Data['Email'], Email_Data['Target'])
# TFIDF feature generation for a maximum of 5000 features
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
tfidf_vect = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}', max_features=5000)
print(tfidf_vect)
tfidf_vect.fit(Email_Data['Email'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)
print(xtrain_tfidf.data)
def train_model(classifier, feature_vector_train, label,feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, valid_y)
accuracy = train_model(naive_bayes.MultinomialNB(alpha=0.2),xtrain_tfidf, train_y, xvalid_tfidf)
print ("Accuracy: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(),xtrain_tfidf, train_y, xvalid_tfidf)
print ("Accuracy: ", accuracy)
    
