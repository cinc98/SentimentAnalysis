from preprocess import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
import joblib
import numpy as np


def prepare_for_training(data,test_data):
    vectorizer = CountVectorizer()
    train = vectorizer.fit_transform(data)
    test = vectorizer.transform(test_data)
    return train,test

def train(X_train,y_train):
    print("====> Training...")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    return logreg

def test(X_test,y_test,model):
    print("====> Testing...")
    pred_logreg = model.predict(X_test)
    print(classification_report(y_test, pred_logreg))

def save_model(model,filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def main():
    df = read_data()
    X_train, X_test, y_train, y_test = train_test_split(df.tweet, df.sentiment,test_size=0.2)
    print("Data: ")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    X_train_corpus = clean_data(X_train) 
    X_test_corpus = clean_data(X_test) 
    
    X_train = np.array(X_train_corpus)
    X_test = np.array(X_test_corpus)
    print("Data after cleaning: ")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    X_train,X_test = prepare_for_training(X_train,X_test)
    
    model = train(X_train,y_train)
    save_model(model,"LogisticRegression.sav")
    
    test(X_test,y_test,model)

if __name__ == '__main__':
    main()





