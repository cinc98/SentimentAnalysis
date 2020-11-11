import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
import pandas as pd
import string

def read_data():
    print("====> Reading data")
    df = pd.read_csv('new_data.csv',encoding = "ISO-8859-1")
    # df.columns =['tweet','sentiment']
    return df

def clean_doc(doc):
	tokens = doc.split() 
	tokens = [word for word in tokens if not word.startswith('@')]
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	tokens = [word for word in tokens if word.isalpha()]
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	tokens = [word for word in tokens if len(word) > 1]
	return tokens


def clean_data(data:list):
    print("====> Cleaning data")
    ret_list=[]
    for row in data:
        tokens = clean_doc(row)
        sentence= ' '.join(tokens)
        ret_list.append(sentence)
    return ret_list