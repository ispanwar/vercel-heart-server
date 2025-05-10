from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load('heart_model.pkl')

@app.route('/')
def home():
    return "<h1>Welcome to Heart Disease Prediction</h1>"

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = pd.DataFrame(data, index=[0])
        prediction = model.predict(features)
        return jsonify({'prediction':int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')