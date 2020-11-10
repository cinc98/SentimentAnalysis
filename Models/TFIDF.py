from preprocess import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
import joblib
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import words

from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression

def prepare_for_model(X_train_corpus, X_test_corpus, y_train, y_test):
    X_train = DataFrame (X_train_corpus,columns=['tweet'])
    y_train = DataFrame (y_train,columns=['sentiment'])
    X_test = DataFrame (X_test_corpus,columns=['tweet'])
    y_test = DataFrame (y_test,columns=['sentiment'])
    return  X_train, X_test, y_train, y_test


def main():
    df = read_data()
    X_train, X_test, y_train, y_test = train_test_split(df.tweet, df.sentiment,test_size=0.2)
    print("Data: ")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    y_test[y_test==4]=1
    y_train[y_train==4]=1

    X_train_corpus = clean_data(X_train) 
    X_test_corpus = clean_data(X_test)

    X_train, X_test, y_train, y_test  = prepare_for_model(X_train_corpus, X_test_corpus, y_train, y_test)

    tv = TfidfVectorizer(
                    ngram_range = (1,3),
                    sublinear_tf = True,
                    max_features = 2000000)
    
    train_tv = tv.fit_transform(X_train['tweet'])
    test_tv = tv.transform(X_test['tweet'])

    vocab = tv.get_feature_names()
    print("===> Training...")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(train_tv, y_train['sentiment'])

    print("===> Testing...")
    pred_logreg = logreg.predict(test_tv)
    print(classification_report(y_test[['sentiment']], pred_logreg))    
    

if __name__ == '__main__':
    main()