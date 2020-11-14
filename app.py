from flask import Flask, request, render_template, url_for
import os
import random 
import keras 
import joblib
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from zipfile import ZipFile 

app = Flask(__name__)

  
# specifying the zip file name 
file_name = "Deep_learning/models.zip"
  
# opening the zip file in READ mode 
with ZipFile(file_name, 'r') as zip: 
    zip.extract('CNN_best_model.h5')

tokenizer = joblib.load("Deep_learning/tokenizer_full.joblib")
model = keras.models.load_model("CNN_best_model.h5")

@app.route('/predict', methods = ['POST'])
def modelPredict():
    sentence = request.form['sentence']

    sequences = tokenizer.texts_to_sequences([sentence])
    padded_sequences = pad_sequences(sequences, maxlen=46)
    with tf.device('/cpu'):
        predict = model.predict(padded_sequences)
        print(predict)
    return render_template('index.html', sentence = sentence, predict = predict[0][0])


@app.route('/')
def index():
    return render_template('index.html')


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == '__main__':
    app.run(debug=True)

