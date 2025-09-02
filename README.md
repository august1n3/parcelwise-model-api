# Parcel Wise Travel Time Prediction API

This project provides a Flask API for predicting travel time for parcel deliveries. The API uses a pre-trained Gradient Boosting Regressor model to make predictions.

## How the API Server Works

The API server is a Flask application defined in `main.py`. It has a single endpoint:

### `/predict`

*   **Method:** POST
*   **Description:** Predicts the travel time for a list of parcel deliveries.
*   **Input:** A JSON array of delivery objects. Each object should have the following features:
    *   `receipt_lng`: Longitude of the pickup location.
    *   `receipt_lat`: Latitude of the pickup location.
    *   `sign_lng`: Longitude of the delivery location.
    *   `sign_lat`: Latitude of the delivery location.
    *   `hour`: The hour of the day when the delivery is scheduled.
    *   `day_of_week`: The day of the week (e.g., 0 for Monday, 1 for Tuesday).
    *   `distance_km`: The distance between the pickup and delivery locations in kilometers.
    *   `city_encoded`: An encoded representation of the city.
    *   `typecode_encoded`: An encoded representation of the delivery type.
*   **Output:** A JSON object with a single key, `predicted_travel_times`, which is a list of the predicted travel times in minutes for each delivery.

**Example Usage:**

```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @example.json
```

Where `example.json` contains:
```json
[
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
]
```

## How the Model Was Exported

The machine learning model used by the API is a `GradientBoostingRegressor` from the `scikit-learn` library. The model was trained and exported using the code in the `parcewisecode.ipynb` notebook.

The following steps were taken to create the model:

1.  **Data Loading:** The training data was loaded from a CSV file (`delivery_five_cities_tanzania.csv`).
2.  **Feature Engineering:**
    *   The travel time in minutes was calculated from the `receipt_time` and `sign_time` columns.
    *   Temporal features, such as the hour and day of the week, were extracted from the timestamp.
    *   The Haversine formula was used to calculate the distance in kilometers between the pickup and delivery locations.
    *   Categorical features, like the city name and delivery type, were encoded into numerical values.
3.  **Model Training:**
    *   The data was split into training and testing sets.
    *   A `GradientBoostingRegressor` model was trained on the training data.
4.  **Model Export:**
    *   The trained model was serialized and saved to the file `parcel_wise_model.joblib` using the `joblib.dump()` function.

This `.joblib` file is loaded by the Flask API to make predictions on new data.
