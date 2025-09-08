import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Add the scripts directory to the path to import the glucose pipeline
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from glucose_prediction_pipeline import GlucosePredictionPipeline

# Path to the TFT model
MODEL_PATH = os.path.join("models", "glucose_model.pth")

# Initialize the pipeline once at startup
try:
    glucose_pipeline = GlucosePredictionPipeline(MODEL_PATH, device='cpu')
    print("✅ Glucose prediction pipeline initialized successfully")
except Exception as e:
    print(f"❌ Error initializing glucose pipeline: {e}")
    glucose_pipeline = None

def predict_glucose(patient_id: int, data: List[Dict], contexts: Optional[Dict] = None, risk_horizons: Optional[List[int]] = None) -> Dict:
    """
    Predict glucose levels and assess risk using TFT model and rule-based classifier.
    
    Args:
        patient_id (int): Patient identifier
        data (List[Dict]): List of daily patient data records
        contexts (Optional[Dict]): Context factors for risk assessment
        risk_horizons (Optional[List[int]]): List of risk assessment horizons in days
        
    Returns:
        Dict: Prediction results with glucose forecasts and risk assessment
    """
    if glucose_pipeline is None:
        raise Exception("Glucose prediction pipeline not initialized")
    
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        required_columns = [
            'patient_id', 'date', 'g_mean', 'g_std', 'pct_hypo', 'pct_hyper',
            'insulin_dose', 'insulin_adherence', 'sleep_quality', 'exercise_flag',
            'meal_variability', 'stress_index', 'illness_flag', 'missed_insulin',
            'hypo_past7d', 'hyper_past7d', 'g_mean_lag1', 'g_mean_7d_mean',
            'g_mean_14d_std', 'insulin_dose_lag1', 'insulin_dose_lag2',
            'insulin_dose_lag3', 'insulin_adherence_7d_mean', 'g_mean_14d_mean',
            'g_mean_30d_mean', 'weekday', 'is_weekend'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure patient_id matches
        if 'patient_id' in df.columns:
            df = df[df['patient_id'] == patient_id]
        
        # Add patient_id if not present
        if 'patient_id' not in df.columns:
            df['patient_id'] = patient_id
        
        # Ensure we have enough data (minimum 60 days)
        if len(df) < 60:
            raise ValueError(f"Insufficient data for patient {patient_id}. Need at least 60 days, got {len(df)}")
        
        # Run the full pipeline
        results = glucose_pipeline.run_full_pipeline(df, patient_id, contexts, risk_horizons)
        
        # Add model info
        results["model_info"] = {
            "model_type": "TemporalFusionTransformer",
            "prediction_horizon": "90 days",
            "features_used": 23,  # 2 known + 21 unknown features
            "risk_assessment": "Rule-based with explainability"
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results = convert_numpy(results)
        return results
        
    except Exception as e:
        raise Exception(f"Glucose prediction failed: {str(e)}")

def validate_glucose_data(data: List[Dict]) -> Dict:
    """
    Validate glucose prediction input data format.
    
    Args:
        data (List[Dict]): Input data to validate
        
    Returns:
        Dict: Validation results
    """
    if not isinstance(data, list):
        return {"valid": False, "error": "Data must be a list of records"}
    
    if len(data) == 0:
        return {"valid": False, "error": "Data cannot be empty"}
    
    # Check first record for required fields
    first_record = data[0]
    required_fields = [
        'date', 'g_mean', 'g_std', 'pct_hypo', 'pct_hyper',
        'insulin_dose', 'insulin_adherence', 'sleep_quality', 'exercise_flag',
        'meal_variability', 'stress_index', 'illness_flag', 'missed_insulin',
        'hypo_past7d', 'hyper_past7d', 'g_mean_lag1', 'g_mean_7d_mean',
        'g_mean_14d_std', 'insulin_dose_lag1', 'insulin_dose_lag2',
        'insulin_dose_lag3', 'insulin_adherence_7d_mean', 'g_mean_14d_mean',
        'g_mean_30d_mean', 'weekday', 'is_weekend'
    ]
    
    missing_fields = [field for field in required_fields if field not in first_record]
    if missing_fields:
        return {"valid": False, "error": f"Missing required fields: {missing_fields}"}
    
    # Check data types and ranges
    validation_errors = []
    
    for i, record in enumerate(data[:5]):  # Check first 5 records
        # Check date format
        try:
            pd.to_datetime(record['date'])
        except:
            validation_errors.append(f"Record {i}: Invalid date format")
        
        # Check numeric ranges
        numeric_checks = {
            'g_mean': (50, 500),  # Glucose range
            'g_std': (0, 100),    # Standard deviation
            'pct_hypo': (0, 100), # Percentage
            'pct_hyper': (0, 100), # Percentage
            'insulin_dose': (0, 200), # Insulin units
            'insulin_adherence': (0, 1), # Adherence rate
            'sleep_quality': (0, 1), # Sleep quality score
            'exercise_flag': (0, 1), # Binary flag
            'meal_variability': (0, 1), # Variability score
            'stress_index': (0, 1), # Stress level
            'illness_flag': (0, 1), # Binary flag
            'missed_insulin': (0, 1), # Binary flag
            'weekday': (0, 6), # Day of week
            'is_weekend': (0, 1) # Weekend flag
        }
        
        for field, (min_val, max_val) in numeric_checks.items():
            if field in record:
                try:
                    val = float(record[field])
                    if val < min_val or val > max_val:
                        validation_errors.append(f"Record {i}: {field} value {val} out of range [{min_val}, {max_val}]")
                except (ValueError, TypeError):
                    validation_errors.append(f"Record {i}: {field} must be numeric")
    
    if validation_errors:
        return {"valid": False, "error": f"Validation errors: {'; '.join(validation_errors)}"}
    
    return {"valid": True, "message": "Data format is valid"}

def get_glucose_model_info() -> Dict:
    """
    Get information about the glucose prediction model.
    
    Returns:
        Dict: Model information
    """
    return {
        "model_type": "TemporalFusionTransformer (TFT)",
        "architecture": "Encoder-decoder with attention mechanism",
        "input_features": {
            "time_varying_known": ["weekday", "is_weekend"],
            "time_varying_unknown": [
                "g_mean", "g_std", "pct_hypo", "pct_hyper", "insulin_dose",
                "insulin_adherence", "sleep_quality", "exercise_flag",
                "meal_variability", "stress_index", "illness_flag",
                "missed_insulin", "g_mean_lag1", "g_mean_7d_mean",
                "g_mean_14d_std", "insulin_dose_lag1", "insulin_dose_lag2",
                "insulin_dose_lag3", "insulin_adherence_7d_mean",
                "g_mean_14d_mean", "g_mean_30d_mean", "hypo_past7d", "hyper_past7d"
            ],
            "static": ["patient_id"]
        },
        "prediction_horizon": "90 days",
        "output_quantiles": [0.1, 0.5, 0.9],
        "risk_assessment": "Rule-based classifier with explainability",
        "minimum_data_required": "60 consecutive days",
        "model_file": "models/glucose_model.pth"
    }
