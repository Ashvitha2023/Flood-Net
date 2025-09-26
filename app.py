from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from datetime import datetime
import math
import os

app = Flask(__name__)

# Load ML model
model = None
try:
    model = pickle.load(open('flood_model.pkl', 'rb'))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Multi-dam data storage
dams_data = {}  # key: dam_id

# Risk thresholds
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8
}

# Haversine formula to compute distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        dam_id = data['dam_id']
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        water_level = float(data['water_level'])
        temperature = float(data['temperature'])
        rainfall = float(data['rainfall'])

        # Prediction using actual input values
        if model is not None:
            features = np.array([[water_level, temperature, rainfall]])
            prediction = int(model.predict(features)[0])
            prediction_proba = float(model.predict_proba(features)[0][1])

            if prediction_proba < RISK_THRESHOLDS["low"]:
                risk_level = "Low"
            elif prediction_proba < RISK_THRESHOLDS["medium"]:
                risk_level = "Medium"
            elif prediction_proba < RISK_THRESHOLDS["high"]:
                risk_level = "High"
            else:
                risk_level = "Severe"
        else:
            prediction, prediction_proba, risk_level = 0, 0.0, "Unknown"

        # Store/update dam data
        dams_data[dam_id] = {
            'dam_id': dam_id,
            'latitude': lat,
            'longitude': lon,
            'water_level': water_level,
            'temperature': temperature,
            'rainfall': rainfall,
            'prediction': prediction,
            'probability': prediction_proba,
            'risk_level': risk_level,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify({'status': 'success', 'dam_id': dam_id})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/data')
def get_data():
    # Reference dam for distance calculation (your main dam)
    my_dam_id = "DAM001"
    if my_dam_id in dams_data:
        ref_lat = dams_data[my_dam_id]['latitude']
        ref_lon = dams_data[my_dam_id]['longitude']
    else:
        ref_lat, ref_lon = None, None

    all_dams = []
    for dam_id, dam in dams_data.items():
        distance = haversine(ref_lat, ref_lon, dam['latitude'], dam['longitude']) if ref_lat else None
        dam_copy = dam.copy()
        dam_copy['distance_km'] = round(distance, 2) if distance is not None else None
        all_dams.append(dam_copy)

    return jsonify(all_dams)

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)  # Ensure templates folder exists
    app.run(host='0.0.0.0', port=5000, debug=True)
