from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and label encoder
MODEL_PATH = 'model/house_price_model.pkl'
ENCODER_PATH = 'model/label_encoder.pkl'

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Model and encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_encoder = None

# Get the list of neighborhoods from the encoder
neighborhoods = label_encoder.classes_.tolist() if label_encoder else []

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html', neighborhoods=neighborhoods)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or label_encoder is None:
            return jsonify({
                'error': 'Model not loaded. Please check if model files exist.'
            }), 500
        
        # Get form data
        overall_qual = int(request.form['overall_qual'])
        gr_liv_area = float(request.form['gr_liv_area'])
        total_bsmt_sf = float(request.form['total_bsmt_sf'])
        garage_cars = int(request.form['garage_cars'])
        year_built = int(request.form['year_built'])
        neighborhood = request.form['neighborhood']
        
        # Validate inputs
        if overall_qual < 1 or overall_qual > 10:
            return jsonify({'error': 'Overall Quality must be between 1 and 10'}), 400
        
        if gr_liv_area < 0 or total_bsmt_sf < 0:
            return jsonify({'error': 'Square footage values must be positive'}), 400
        
        if garage_cars < 0 or garage_cars > 5:
            return jsonify({'error': 'Garage Cars must be between 0 and 5'}), 400
        
        if year_built < 1800 or year_built > 2026:
            return jsonify({'error': 'Year Built must be between 1800 and 2026'}), 400
        
        # Encode neighborhood
        try:
            neighborhood_encoded = label_encoder.transform([neighborhood])[0]
        except ValueError:
            return jsonify({'error': f'Invalid neighborhood: {neighborhood}'}), 400
        
        # Prepare features in the correct order
        # Order: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, YearBuilt, Neighborhood_Encoded
        features = np.array([[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            year_built,
            neighborhood_encoded
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Format the prediction
        predicted_price = f"${prediction:,.2f}"
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'raw_price': float(prediction)
        })
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    encoder_status = "loaded" if label_encoder is not None else "not loaded"
    
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'encoder': encoder_status
    })

if __name__ == '__main__':
    # Use port from environment variable for deployment (Render uses PORT)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)