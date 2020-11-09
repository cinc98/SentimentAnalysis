from flask import Flask, request

app = Flask(__name__)

@app.route('/model/<string:input_text>', methods = ['GET'])
def modelPredict(input_text):
    text_for_model = input_text
    # Model predicts sentiment for text
    
    sentiment = ""
    ret_value = { "sentiment" : text_for_model}
    return ret_value,200

if __name__ == '__main__':
    app.run()
