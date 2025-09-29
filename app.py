import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Load the saved preprocessor and model
# Make sure these files are in the same directory as your app.py or provide the correct path
try:
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: preprocessor.pkl or random_forest_model.pkl not found. Make sure they are in the correct directory.")
    preprocessor = None
    model = None

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if preprocessor is None or model is None:
        return jsonify({'error': 'Model or preprocessor not loaded.'}), 500

    try:
        # Get the data from the request
        data = request.get_json(force=True)
        # Assuming the incoming data is a dictionary matching the original feature structure
        # Convert the incoming data to a pandas DataFrame
        new_data = pd.DataFrame([data])

        # Preprocess the new data using the loaded preprocessor
        # Need to handle potential missing columns if the incoming data doesn't have all features
        # A more robust implementation would handle this, but for simplicity, we assume incoming data matches training features
        try:
             # Ensure columns are in the same order as during training if necessary, although ColumnTransformer handles this
             # based on feature names. However, missing columns in inference data can cause issues.
             # A safer approach might involve explicitly reindexing or checking for all expected columns.
             # For this example, assuming new_data has columns compatible with the preprocessor's expectations.
             processed_data = preprocessor.transform(new_data)
             # Convert processed_data back to DataFrame with correct column names if needed for consistency,
             # but model.predict typically works with the numpy array output of transform.

        except Exception as e:
             return jsonify({'error': f'Error during preprocessing: {e}'}), 400


        # Make predictions using the loaded model
        prediction = model.predict(processed_data)

        # The prediction is a numpy array, convert it to a list or string for the JSON response
        # Assuming the target variable 'Churn' is binary ('Yes'/'No' or 1/0)
        prediction_result = 'Yes' if prediction[0] == 1 else 'No' # Adjust based on how your target was encoded/handled

        return jsonify({'prediction': prediction_result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # This is for running locally. For production deployment, use a production-ready server like Gunicorn or uWSGI
    app.run(port=5000, debug=True)
