import joblib
import os
import pandas as pd
import shap
import numpy as np  # ✅ for safe flattening

# Path to cardiovascular model
MODEL_PATH = os.path.join("models", "cardiovascular_risk_model.pkl")

# Load model once at startup
cardiovascular_model_package = joblib.load(MODEL_PATH)
cardiovascular_model = cardiovascular_model_package['model']
scaler = cardiovascular_model_package['scaler']

# Required columns in exact order used during training
COLUMNS = cardiovascular_model_package['features']

# Create SHAP explainer (LinearExplainer works well for LogisticRegression)
explainer = shap.LinearExplainer(cardiovascular_model, scaler.transform(np.zeros((1, len(COLUMNS)))))


def predict_cardiovascular(features: list, patient_data=None):
    """
    Predict cardiovascular health risk and return comprehensive analysis including trends and risk factors.

    Args:
        features (list): Input values [age, systolic_bp, diastolic_bp, heart_rate, cholesterol, 
                         glucose, medication_adherence, exercise_minutes, diet_score, stress_level,
                         weight_kg, oxygen_saturation, temperature_c, sleep_hours]
        patient_data (dict): Optional time series data for trend analysis

    Returns:
        dict: Comprehensive prediction result with trends and risk factors.
    """
    # Build input DataFrame
    input_df = pd.DataFrame([features], columns=COLUMNS)

    # Scale features using the same scaler from training
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = cardiovascular_model.predict(input_scaled)[0]
    probabilities = cardiovascular_model.predict_proba(input_scaled)[0]

    # SHAP values
    shap_values = explainer.shap_values(input_scaled)

    # Handle binary vs multiclass classifiers
    if isinstance(shap_values, list):  # Multiclass -> pick predicted class
        class_index = list(cardiovascular_model.classes_).index(prediction)
        shap_for_instance = shap_values[class_index][0]
    else:  # Binary
        shap_for_instance = shap_values[0]

    # Map feature names to SHAP values (flatten arrays safely)
    raw_shap_values = {
        col: float(np.ravel(val)[0]) for col, val in zip(COLUMNS, shap_for_instance)
    }

    # Normalize SHAP values to 0-1 scale
    shap_values_array = np.array(list(raw_shap_values.values()))
    min_shap = np.min(shap_values_array)
    max_shap = np.max(shap_values_array)
    
    # Avoid division by zero
    if max_shap - min_shap == 0:
        feature_importance = {col: 0.5 for col in COLUMNS}
    else:
        feature_importance = {
            col: (raw_shap_values[col] - min_shap) / (max_shap - min_shap)
            for col in COLUMNS
        }

    # Calculate risk level based on probability
    risk_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
    risk_level = "HIGH" if risk_prob > 0.7 else "MEDIUM" if risk_prob > 0.3 else "LOW"

    # Get top risk factors (sorted by absolute SHAP value)
    risk_factors = sorted(feature_importance.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
    top_risk_factors = risk_factors[:5]

    # Initialize trends (will be calculated if patient_data provided)
    trends = {}
    if patient_data and isinstance(patient_data, list) and len(patient_data) > 1:
        trends = calculate_trends(patient_data)

    # Create comprehensive response
    result = {
        "prediction": str(prediction),
        "risk_level": risk_level,
        "risk_score": float(risk_prob),
        "probabilities": {
            str(cls): float(prob) for cls, prob in zip(cardiovascular_model.classes_, probabilities)
        },
        "shap_values": feature_importance,
        "top_risk_factors": [
            {"feature": factor, "importance": importance} 
            for factor, importance in top_risk_factors
        ],
        "trends": trends,
        "recommendation": get_recommendation(risk_level, top_risk_factors)
    }

    return result


def calculate_trends(patient_data):
    """
    Calculate trends from time series patient data.
    
    Args:
        patient_data (list): List of patient measurements over time
        
    Returns:
        dict: Trend calculations for key vital signs
    """
    try:
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(patient_data)
        
        # Ensure we have datetime column
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
        elif 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
            df = df.sort_values('datetime')
        else:
            # If no datetime, assume sequential data
            df = df.reset_index()
        
        # Calculate trends using linear regression (polyfit)
        trends = {}
        
        # Key vital signs for trend analysis
        vital_signs = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'medication_adherence', 
                      'oxygen_saturation', 'temperature_c']
        
        for vital in vital_signs:
            if vital in df.columns and len(df) > 1:
                try:
                    # Use polyfit to get slope (trend)
                    x = np.arange(len(df))
                    y = df[vital].values
                    slope = np.polyfit(x, y, 1)[0]
                    trends[vital] = round(slope, 2)
                except:
                    trends[vital] = 0.0
        
        return trends
        
    except Exception as e:
        return {}


def get_recommendation(risk_level, top_risk_factors):
    """
    Generate recommendation based on risk level and top risk factors.
    
    Args:
        risk_level (str): Risk level (LOW, MEDIUM, HIGH)
        top_risk_factors (list): Top risk factors with importance scores
        
    Returns:
        str: Recommendation message
    """
    if risk_level == "HIGH":
        return "🚨 Immediate intervention needed - High cardiovascular risk detected"
    elif risk_level == "MEDIUM":
        return "⚠️ Enhanced monitoring recommended - Moderate risk factors present"
    else:
        return "✅ Continue current care - Low risk profile maintained"
