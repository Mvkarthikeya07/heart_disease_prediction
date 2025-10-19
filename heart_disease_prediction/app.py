from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('models/heart_model.pkl')

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/', methods=['POST'])
def predict():
    # Get form data
    data = {
        'Age': int(request.form['Age']),
        'RestingBP': int(request.form['RestingBP']),
        'Cholesterol': int(request.form['Cholesterol']),
        'FastingBS': int(request.form['FastingBS']),
        'MaxHR': int(request.form['MaxHR']),
        'Oldpeak': float(request.form['Oldpeak']),
        'Sex': request.form['Sex'],
        'ChestPainType': request.form['ChestPainType'],
        'RestingECG': request.form['RestingECG'],
        'ExerciseAngina': request.form['ExerciseAngina'],
        'ST_Slope': request.form['ST_Slope']
    }

    # Convert form data to dataframe for model
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    
    # Ensure all model columns exist
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]
    
    # Prediction
    prediction = model.predict(df)[0]
    
    return render_template('result.html', result=prediction, data=data)

if __name__ == "__main__":
    app.run(debug=True)
