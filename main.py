# app.py
# This script loads a trained model and hosts it as a Flask API.
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Create the Flask application
app = Flask(__name__)

# --- Load the Trained Model ---
# Load the pipeline object you saved in train.py
try:
    model_pipeline = joblib.load('model.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'model.joblib' not found. Please run train.py first to create the model file.")
    exit()

# --- Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives client data in a POST request, uses the loaded model
    to make a prediction, and returns the prediction as JSON.
    """
    # Get the JSON data from the request
    json_data = request.get_json()
    if not json_data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Convert the JSON data into a pandas DataFrame
        # The model expects a DataFrame as input.
        input_df = pd.DataFrame([json_data])

        # Ensure the column order matches the training data
        # This is a good practice to avoid errors if the JSON keys are not in order.
        # Get the feature names from the model's preprocessor
        # Note: This is a robust way to get the original feature names before one-hot encoding
        original_features = list(model_pipeline.named_steps['preprocessor'].transformers_[0][2]) + \
                            list(model_pipeline.named_steps['preprocessor'].transformers_[1][2])

        input_df = input_df[original_features]

        # Use the pipeline to make a prediction
        # The pipeline handles both preprocessing and prediction.
        prediction_code = model_pipeline.predict(input_df)

        # Convert the prediction code (0 or 1) to a meaningful label
        prediction_label = 'Subscription' if prediction_code[0] == 1 else 'No Subscription'

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        # Handle potential errors, such as missing columns or invalid data
        return jsonify({'error': str(e)}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

