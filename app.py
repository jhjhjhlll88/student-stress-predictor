from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Load models and preprocessing objects
model = load_model('stress_model.h5')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        data = request.form.to_dict()
        
        # Convert to numpy array in correct order
        input_data = np.array([
            float(data['study_hours']),
            float(data['extracurricular_hours']),
            float(data['sleep_hours']),
            float(data['social_hours']),
            float(data['physical_activity_hours']),
            float(data['gpa'])
        ]).reshape(1, -1)
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        predicted_class = np.argmax(prediction, axis=-1)[0]
        
        # Get the stress level label
        stress_level = encoder.inverse_transform([predicted_class])[0]
        confidence = float(np.max(prediction))
        
        # Return results
        return jsonify({
            'status': 'success',
            'prediction': stress_level,
            'confidence': round(confidence * 100, 2)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)