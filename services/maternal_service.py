# import joblib
# import os
# import pandas as pd  # ✅ Needed for DataFrame

# # Path to maternal model
# MODEL_PATH = os.path.join("models", "maternal_risk_model.pkl")

# # Load model once at startup
# maternal_model = joblib.load(MODEL_PATH)

# # Required columns in exact order used during training
# COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

# def predict_maternal(features: list):
#     """
#     Predict maternal health risk.

#     Args:
#         features (list): Input values [Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]

#     Returns:
#         dict: Prediction result.
#     """
#     input_df = pd.DataFrame([features], columns=COLUMNS)

#     prediction = maternal_model.predict(input_df)[0]  # already "low risk", "mid risk", etc.
#     return {"prediction": str(prediction)}

# File: /services/maternal_service.py

import joblib
import os
import pandas as pd  # ✅ Needed for DataFrame

# Path to maternal model
MODEL_PATH = os.path.join("models", "maternal_risk_model.pkl")

# Load model once at startup
maternal_model = joblib.load(MODEL_PATH)

# Required columns in exact order used during training
COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

def predict_maternal(features: list):
    """
    Predict maternal health risk.

    Args:
        features (list): Input values [Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]

    Returns:
        dict: Prediction result including label + probabilities.
    """
    # Convert input into DataFrame with proper column names
    input_df = pd.DataFrame([features], columns=COLUMNS)

    # Prediction (already returns "low risk", "mid risk", "high risk")
    prediction = maternal_model.predict(input_df)[0]

    # Probabilities (if the model supports predict_proba)
    try:
        probabilities = maternal_model.predict_proba(input_df)[0]
        class_labels = maternal_model.classes_
        prob_dict = {str(label): float(prob) for label, prob in zip(class_labels, probabilities)}
    except Exception:
        prob_dict = None  # Model doesn’t support probability output

    return {
        "prediction": str(prediction),
        "probabilities": prob_dict
    }
