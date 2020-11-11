from flask import Flask, request, render_template, url_for
import os
import random 
app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def modelPredict():
    sentence = request.form['sentence']
    # Model predicts sentiment for text
    
    return render_template('index.html', sentence = sentence, predict = str(random.randint(0, 1)))


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

