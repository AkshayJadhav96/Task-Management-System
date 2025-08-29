from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from ai_task_management_system import AITaskManagementSystem
import os
from datetime import datetime

app = Flask(__name__)

# Initialize the AI Task Management System
system = AITaskManagementSystem()

# Load artifacts
try:
    system.vectorizer = joblib.load('tfidf_vectorizer.pkl')
    system.label_encoders = joblib.load('label_encoders.pkl')
    system.scaler = joblib.load('feature_scaler.pkl')
    system.task_classifier = joblib.load('task_classifier_model.pkl')
    system.priority_predictor = joblib.load('priority_predictor_model.pkl')
    system.priority_encoder = joblib.load('priority_encoder.pkl')
    system.duration_predictor = joblib.load('duration_predictor_model.pkl')
    system.duration_scaler = joblib.load('duration_scaler.pkl')
    system.task_forecast_model = joblib.load('task_forecast_model.pkl')
    system.df = pd.read_csv('users.csv')  # For workload balancing
    print("✅ All artifacts loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: Missing artifact {e}. Please ensure all models and data are available.")
    raise

@app.route('/')
def index():
    # Get user roles for the dropdown
    user_roles = system.label_encoders['user_role'].classes_.tolist()
    return render_template('index.html', user_roles=user_roles)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        task_description = request.form['task_description']
        user_workload = float(request.form['user_workload'])
        user_behavior_score = float(request.form['user_behavior_score'])
        days_until_due = int(request.form['days_until_due'])
        user_role = request.form['user_role']

        # Validate inputs
        if not task_description:
            return jsonify({'error': 'Task description is required.'}), 400
        if user_workload < 0 or user_workload > 20:
            return jsonify({'error': 'User workload must be between 0 and 20.'}), 400
        if user_behavior_score < 0 or user_behavior_score > 1:
            return jsonify({'error': 'User behavior score must be between 0 and 1.'}), 400
        if days_until_due < 0:
            return jsonify({'error': 'Days until due must be non-negative.'}), 400
        if user_role not in system.label_encoders['user_role'].classes_:
            return jsonify({'error': f"Invalid user role. Choose from {system.label_encoders['user_role'].classes_}."}), 400

        # Make prediction
        category, priority, duration, user = system.predict_new_task(
            task_description=task_description,
            user_workload=user_workload,
            user_behavior_score=user_behavior_score,
            days_until_due=days_until_due,
            user_role=user_role
        )

        if category is None:
            return jsonify({'error': 'Prediction failed. Check artifacts and try again.'}), 500

        result = {
            'category': category,
            'priority': priority,
            'duration': f"{duration:.1f} minutes" if duration else "Not available",
            'assigned_user': user
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        periods = int(request.form['forecast_period'])
        if periods < 1 or periods > 90:
            return jsonify({'error': 'Forecast period must be between 1 and 90 days.'}), 400

        forecast = system.generate_task_forecast(periods)
        if forecast is None:
            return jsonify({'error': 'Forecasting failed. Check forecast model.'}), 500

        # Prepare data for Chart.js
        forecast_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in forecast['ds'].tail(periods)],
            'yhat': forecast['yhat'].tail(periods).tolist(),
            'yhat_lower': forecast['yhat_lower'].tail(periods).tolist(),
            'yhat_upper': forecast['yhat_upper'].tail(periods).tolist()
        }
        return jsonify(forecast_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
