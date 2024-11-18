from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, scaler, and feature names
model = joblib.load('random_forest_regressor_model.pkl')
scaler = joblib.load('standard_scaler.pkl')
model_features = joblib.load('model_features.pkl')  # Load saved feature names

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        airline = request.form['Airline']
        source = request.form['Source']
        destination = request.form['Destination']
        total_stops = int(request.form['Total_Stops'])
        dep_hours = int(request.form['Dep_hours'])
        dep_min = int(request.form['Dep_min'])
        arrival_hours = int(request.form['Arrival_hours'])
        arrival_min = int(request.form['Arrival_min'])
        duration_hours = int(request.form['Duration_hours'])
        duration_min = int(request.form['Duration_min'])

        # Create a DataFrame to match the model's expected input
        input_data = pd.DataFrame({
            'Airline': [airline],
            'Source': [source],
            'Destination': [destination],
            'Total_Stops': [total_stops],
            'Dep_hours': [dep_hours],
            'Dep_min': [dep_min],
            'Arrival_hours': [arrival_hours],
            'Arrival_min': [arrival_min],
            'Duration_hours': [duration_hours],
            'Duration_min': [duration_min]
        })

        # Apply one-hot encoding for categorical features
        input_data = pd.get_dummies(input_data, columns=["Airline", "Source", "Destination"], drop_first=True)

        # Ensure the input data matches the saved model's features
        for col in model_features:
            if col not in input_data.columns:
                input_data[col] = 0  # Add missing columns with 0
        input_data = input_data[model_features]

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_scaled)

        # Display the result
        return render_template('index.html', prediction_text=f'Predicted Price: {prediction[0]:.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
