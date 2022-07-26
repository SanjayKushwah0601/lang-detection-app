from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        text = request.form['text']
        prediction=model.predict([text])
        output= prediction[0]
        return render_template('index.html',prediction_text="Language is: {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)