from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler with robust path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'breast_cancer_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✓ Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# Feature names (must match training order)
FEATURE_NAMES = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness'
]

@app.route('/')
def home():
    """Render the home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded properly. Please check server logs.'
            }), 500

        # Get input data from the form with safer access
        radius_raw = request.form.get('radius_mean')
        texture_raw = request.form.get('texture_mean')
        perimeter_raw = request.form.get('perimeter_mean')
        area_raw = request.form.get('area_mean')
        smoothness_raw = request.form.get('smoothness_mean')

        # Check for missing fields
        if None in [radius_raw, texture_raw, perimeter_raw, area_raw, smoothness_raw]:
            return jsonify({'error': 'All fields are required'}), 400

        # Convert and validate inputs
        try:
            radius_mean = float(radius_raw)
            texture_mean = float(texture_raw)
            perimeter_mean = float(perimeter_raw)
            area_mean = float(area_raw)
            smoothness_mean = float(smoothness_raw)
        except ValueError:
            return jsonify({'error': 'Invalid numeric input'}), 400

        # Validate input ranges (based on dataset statistics)
        if not (5.0 <= radius_mean <= 35.0):
            return jsonify({'error': 'Mean radius must be between 5.0 and 35.0'}), 400
        
        if not (5.0 <= texture_mean <= 45.0):
            return jsonify({'error': 'Mean texture must be between 5.0 and 45.0'}), 400
        
        if not (30.0 <= perimeter_mean <= 200.0):
            return jsonify({'error': 'Mean perimeter must be between 30.0 and 200.0'}), 400
        
        if not (100.0 <= area_mean <= 2500.0):
            return jsonify({'error': 'Mean area must be between 100.0 and 2500.0'}), 400
        
        if not (0.05 <= smoothness_mean <= 0.20):
            return jsonify({'error': 'Mean smoothness must be between 0.05 and 0.20'}), 400

        # Prepare features in correct order
        features = pd.DataFrame({
            'mean radius': [radius_mean],
            'mean texture': [texture_mean],
            'mean perimeter': [perimeter_mean],
            'mean area': [area_mean],
            'mean smoothness': [smoothness_mean]
        })

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Benign' if prediction == 1 else 'Malignant',
            'benign_probability': float(probability[1] * 100),
            'malignant_probability': float(probability[0] * 100),
            'tumor_features': {
                'radius': radius_mean,
                'texture': texture_mean,
                'perimeter': perimeter_mean,
                'area': area_mean,
                'smoothness': smoothness_mean
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    # Use environment variable to control debug mode
    # In production (Render), this will be False by default
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)