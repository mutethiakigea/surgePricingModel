from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Your data preprocessing pipeline function
def data_preprocessing_pipeline(df):
    numeric_features = df.select_dtypes(include=['float', 'int']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Handle missing values
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())

    # Handle outliers using IQR method
    for feature in numeric_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        df[feature] = np.where((df[feature] < lower_bound) | (df[feature] > upper_bound),
                               df[feature].mean(), df[feature])

    df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

    return df

# Your helper functions for vehicle and booking time
def get_vehicle_type_numeric(vehicle_type):
    vehicle_type_mapping = {
        "Premium": 1,
        "Economy": 0
    }
    return vehicle_type_mapping.get(vehicle_type, -1)

def get_time_of_booking_numeric(time_of_booking):
    time_of_booking_mapping = {
        "Afternoon": 0,
        "Evening": 1,
        "Morning": 2,
        "Night": 3
    }
    return time_of_booking_mapping.get(time_of_booking, -1)

# API route to receive input and return processed data
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert received JSON data into a pandas DataFrame
    df = pd.DataFrame([data])

    # Preprocess the data
    df_preprocessed = data_preprocessing_pipeline(df)

    # Here, you could call your machine learning model for prediction
    # For example, if you have a trained model, you would load it and use:
    # model = load_model('your_model.pkl')
    # prediction = model.predict(df_preprocessed)

    # In this example, we will just return the preprocessed data for simplicity
    return jsonify(df_preprocessed.to_dict(orient='records'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
