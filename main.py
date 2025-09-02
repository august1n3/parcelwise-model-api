from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained machine learning model
# Make sure your trained GradientBoostingRegressor model is saved as a .joblib file
# You can do this in your notebook with: joblib.dump(your_model, 'gradient_boosting_model.joblib')
try:
    model = joblib.load('gradient_boosting_model.joblib')
except FileNotFoundError:
    model = None  # Or handle the error appropriately
    print("Error: The model file 'gradient_boosting_model.joblib' was not found.")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts travel time based on input data from a POST request.
    The input should be a JSON object with the required features.
    
    Example JSON input:
    {
        "receipt_lng": 35.7388,
        "receipt_lat": -6.1752,
        "sign_lng": 35.7723,
        "sign_lat": -6.1917,
        "hour": 14,
        "day_of_week": 1,
        "distance_km": 5.2,
        "city_encoded": 1,
        "typecode_encoded": 2
    }
    """
    if not model:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        # Get the JSON data from the request body
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided in the request body."}), 400

        # Convert the JSON data to a pandas DataFrame
        # The column order must match the order the model was trained on
        input_data = pd.DataFrame([data])
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_data)
        
        # Return the prediction as a JSON response
        return jsonify({"predicted_travel_time_minutes": prediction[0]})
        
    except Exception as e:
        # Handle potential errors during prediction
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the app. For development, use debug=True.
    # For production, use a production-ready WSGI server like Gunicorn or uWSGI.
    app.run(debug=True, host='0.0.0.0', port=8080)