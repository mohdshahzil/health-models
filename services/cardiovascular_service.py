import joblib
import os
import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Path to cardiovascular model (we'll need to create this from the notebook)
MODEL_PATH = os.path.join("models", "cardiovascular_risk_model.pkl")

# Load model once at startup (create if doesn't exist)
try:
    cardiovascular_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(os.path.join("models", "cardiovascular_scaler.pkl"))
    feature_selector = joblib.load(os.path.join("models", "cardiovascular_feature_selector.pkl"))
    selected_features = joblib.load(os.path.join("models", "cardiovascular_selected_features.pkl"))
except FileNotFoundError:
    # If model files don't exist, create a simple model for demo
    print("Cardiovascular model files not found. Creating demo model...")
    cardiovascular_model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    feature_selector = SelectKBest(score_func=f_classif, k=20)
    selected_features = []

# Required columns in exact order used during training (original CSV columns)
COLUMNS = ["age", "gender", "diabetes", "hypertension", "systolic_bp", "diastolic_bp", 
           "heart_rate", "cholesterol", "glucose", "medication_adherence", 
           "exercise_minutes", "diet_score", "stress_level", "weight_kg", 
           "oxygen_saturation", "temperature_c", "sleep_hours"]

# All features that the scaler expects (43 features total) - must match training order exactly
ALL_FEATURES = ["age", "diabetes_encoded", "hypertension_encoded", "gender_encoded", 
                "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol", "glucose", 
                "medication_adherence", "exercise_minutes", "diet_score", "stress_level", 
                "weight_kg", "oxygen_saturation", "temperature_c", "sleep_hours",
                "systolic_bp_trend", "diastolic_bp_trend", "heart_rate_trend", 
                "cholesterol_trend", "glucose_trend", "medication_adherence_trend", 
                "exercise_minutes_trend", "diet_score_trend", "stress_level_trend", 
                "weight_kg_trend", "oxygen_saturation_trend", "temperature_c_trend", 
                "sleep_hours_trend", "systolic_bp_variability", "diastolic_bp_variability", 
                "heart_rate_variability", "cholesterol_variability", "glucose_variability", 
                "medication_adherence_variability", "exercise_minutes_variability", 
                "diet_score_variability", "stress_level_variability", "weight_kg_variability", 
                "oxygen_saturation_variability", "temperature_c_variability", "sleep_hours_variability"]

# Create SHAP explainer (TreeExplainer works well for RandomForest)
try:
    explainer = shap.TreeExplainer(cardiovascular_model)
except:
    explainer = None


def predict_cardiovascular(features: list):
    """
    Predict cardiovascular 90-day deterioration risk and return probabilities + SHAP values.

    Args:
        features (list): Input values [age, gender, diabetes, hypertension, systolic_bp, diastolic_bp, 
                         heart_rate, cholesterol, glucose, medication_adherence, exercise_minutes, 
                         diet_score, stress_level, weight_kg, oxygen_saturation, temperature_c, sleep_hours]

    Returns:
        dict: Prediction result with risk score, probabilities, and SHAP values.
    """
    # Build input DataFrame with basic features
    input_df = pd.DataFrame([features], columns=COLUMNS)
    
    # Create the full feature set that the model expects
    full_features = {}
    
    # Add basic features
    full_features['age'] = input_df['age'].iloc[0]
    
    # Encode categorical variables
    full_features['diabetes_encoded'] = 1 if input_df['diabetes'].iloc[0] == 'Yes' else 0
    full_features['hypertension_encoded'] = 1 if input_df['hypertension'].iloc[0] == 'Yes' else 0
    full_features['gender_encoded'] = 1 if input_df['gender'].iloc[0] == 'Male' else 0
    
    # Add health metrics
    health_metrics = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol', 'glucose',
                     'medication_adherence', 'exercise_minutes', 'diet_score', 'stress_level',
                     'weight_kg', 'oxygen_saturation', 'temperature_c', 'sleep_hours']
    
    for metric in health_metrics:
        full_features[metric] = input_df[metric].iloc[0]
    
    # Add trend features based on patient risk profile (realistic estimates)
    for metric in health_metrics:
        current_value = full_features[metric]
        
        # Create realistic trends based on clinical knowledge and patient risk factors
        if metric == 'systolic_bp':
            # High BP patients tend to have rising trends
            trend = 0.8 if current_value > 150 else 0.2 if current_value > 130 else -0.1
        elif metric == 'diastolic_bp':
            trend = 0.5 if current_value > 95 else 0.1 if current_value > 85 else -0.1
        elif metric == 'heart_rate':
            trend = 0.6 if current_value > 90 else 0.1 if current_value > 80 else -0.1
        elif metric == 'cholesterol':
            trend = 1.2 if current_value > 240 else 0.3 if current_value > 200 else -0.2
        elif metric == 'glucose':
            trend = 1.5 if current_value > 180 else 0.5 if current_value > 140 else -0.3
        elif metric == 'medication_adherence':
            # Poor adherence tends to get worse
            trend = -0.8 if current_value < 60 else -0.3 if current_value < 80 else 0.1
        elif metric == 'exercise_minutes':
            # Low exercise tends to stay low or decrease
            trend = -0.5 if current_value < 50 else -0.2 if current_value < 100 else 0.2
        elif metric == 'diet_score':
            # Poor diet tends to get worse
            trend = -0.6 if current_value < 5 else -0.2 if current_value < 7 else 0.1
        elif metric == 'stress_level':
            # High stress tends to increase
            trend = 0.4 if current_value > 7 else 0.1 if current_value > 5 else -0.1
        elif metric == 'weight_kg':
            # Overweight patients tend to gain more
            trend = 0.3 if current_value > 85 else 0.1 if current_value > 75 else -0.1
        elif metric == 'oxygen_saturation':
            # Low oxygen tends to decrease
            trend = -0.3 if current_value < 96 else -0.1 if current_value < 98 else 0.1
        elif metric == 'temperature_c':
            # High temp tends to increase
            trend = 0.2 if current_value > 37.2 else 0.05 if current_value > 37 else -0.05
        elif metric == 'sleep_hours':
            # Poor sleep tends to get worse
            trend = -0.4 if current_value < 6 else -0.1 if current_value < 7 else 0.1
        else:
            trend = 0.0
        
        full_features[f'{metric}_trend'] = trend
    
    # Add variability features based on patient risk profile (realistic estimates)
    for metric in health_metrics:
        current_value = full_features[metric]
        
        # Create realistic variability based on clinical knowledge
        if metric in ['systolic_bp', 'diastolic_bp']:
            # High BP patients have more variability
            variability = 20.0 if current_value > 150 else 12.0 if current_value > 130 else 8.0
        elif metric == 'heart_rate':
            variability = 15.0 if current_value > 90 else 8.0 if current_value > 80 else 5.0
        elif metric == 'cholesterol':
            variability = 35.0 if current_value > 240 else 20.0 if current_value > 200 else 15.0
        elif metric == 'glucose':
            variability = 40.0 if current_value > 180 else 25.0 if current_value > 140 else 15.0
        elif metric == 'medication_adherence':
            # Poor adherence has high variability
            variability = 25.0 if current_value < 60 else 15.0 if current_value < 80 else 8.0
        elif metric == 'exercise_minutes':
            variability = 20.0 if current_value < 50 else 12.0 if current_value < 100 else 8.0
        elif metric == 'diet_score':
            variability = 3.0 if current_value < 5 else 2.0 if current_value < 7 else 1.5
        elif metric == 'stress_level':
            variability = 2.5 if current_value > 7 else 1.5 if current_value > 5 else 1.0
        elif metric == 'weight_kg':
            variability = 4.0 if current_value > 85 else 2.5 if current_value > 75 else 2.0
        elif metric == 'oxygen_saturation':
            variability = 2.0 if current_value < 96 else 1.2 if current_value < 98 else 0.8
        elif metric == 'temperature_c':
            variability = 0.4 if current_value > 37.2 else 0.25 if current_value > 37 else 0.2
        elif metric == 'sleep_hours':
            variability = 2.0 if current_value < 6 else 1.2 if current_value < 7 else 1.0
        else:
            variability = 1.0
        
        full_features[f'{metric}_variability'] = variability
    
    # Create DataFrame with all features in the correct order
    full_input_df = pd.DataFrame([full_features], columns=ALL_FEATURES)
    
    # Scale features
    input_scaled = scaler.transform(full_input_df)
    
    # Apply feature selection if available
    if selected_features:
        input_selected = feature_selector.transform(input_scaled)
    else:
        input_selected = input_scaled

    # Prediction
    prediction = cardiovascular_model.predict(input_selected)[0]
    probabilities = cardiovascular_model.predict_proba(input_selected)[0]
    
    # Calculate risk score (probability of deterioration)
    risk_score = probabilities[1] if len(probabilities) > 1 else probabilities[0]
    
    # Determine risk level
    if risk_score > 0.7:
        risk_level = "HIGH"
    elif risk_score > 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # SHAP values (if explainer is available)
    shap_values = {}
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(input_selected)
            
            # Handle binary vs multiclass classifiers
            if isinstance(shap_vals, list):  # Multiclass -> pick predicted class
                class_index = list(cardiovascular_model.classes_).index(prediction)
                shap_for_instance = shap_vals[class_index][0]
            else:  # Binary
                shap_for_instance = shap_vals[0]
            
            # Map feature names to SHAP values
            feature_names = selected_features if selected_features else COLUMNS
            shap_values = {
                col: float(np.ravel(val)[0]) for col, val in zip(feature_names, shap_for_instance)
            }
        except Exception as e:
            print(f"SHAP calculation failed: {e}")
            shap_values = {}

    return {
        "prediction": str(prediction),
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "probabilities": {
            str(cls): float(prob) for cls, prob in zip(cardiovascular_model.classes_, probabilities)
        },
        "shap_values": shap_values,
        "model_info": {
            "model_type": "RandomForest",
            "features_used": len(selected_features) if selected_features else len(COLUMNS),
            "prediction_horizon": "90 days"
        }
    }
