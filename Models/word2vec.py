from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import nltk.data
import nltk
from sklearn.metrics import classification_report
from gensim.models import word2vec
nltk.download('punkt')
import numpy as np

def review_sentences(review, tokenizer, remove_stopwords=False):
   
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(clean_doc(raw_sentence))

    return sentences
    
def featureVecMethod(words, model, num_features):
    
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs


def main():
    df = read_data()
    X_train, X_test, y_train, y_test = train_test_split(df.tweet, df.sentiment,test_size=0.2)

    print("Data: ")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    y_test[y_test==4]=1
    y_train[y_train==4]=1

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    print("Parsing sentences from training set")
    for review in X_train:
        sentences += review_sentences(review, tokenizer)
        
    num_features = 300  
    min_word_count = 40 
    num_workers = 4     
    context = 10        
    downsampling = 1e-3 

    print("Training model....")
    model = word2vec.Word2Vec(sentences,\
                            workers=num_workers,\
                            size=num_features,\
                            min_count=min_word_count,\
                            window=context,
                            sample=downsampling)

    model.init_sims(replace=True)

    model_name = "300features_40minwords_10context"
    model.save(model_name)

    clean_train_reviews = []
    for review in X_train:
        clean_train_reviews.append(clean_doc(review))
        
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
     
    clean_test_reviews = []
    for review in X_test:
        clean_test_reviews.append(clean_doc(review))
        
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

    indstest = pd.isnull(testDataVecs).any(1).nonzero()[0]
    testDataVecs = np.delete(testDataVecs, indstest, 0)
    indstrain = pd.isnull(trainDataVecs).any(1).nonzero()[0]
    trainDataVecs = np.delete(trainDataVecs, indstrain, 0)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    y_test=y_test.drop(indstest,axis=0)
    y_train=y_train.drop(indstrain,axis=0)

    print("TRAINING...")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(trainDataVecs, y_train)


    print("TESTING...")
    pred_logreg = logreg.predict(testDataVecs)
    print(classification_report(y_test, pred_logreg))

if __name__ == '__main__':
    main()