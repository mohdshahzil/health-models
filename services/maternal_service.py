# # File: /services/maternal_service.py

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
#         dict: Prediction result including label + probabilities.
#     """
#     # Convert input into DataFrame with proper column names
#     input_df = pd.DataFrame([features], columns=COLUMNS)

#     # Prediction (already returns "low risk", "mid risk", "high risk")
#     prediction = maternal_model.predict(input_df)[0]

#     # Probabilities (if the model supports predict_proba)
#     try:
#         probabilities = maternal_model.predict_proba(input_df)[0]
#         class_labels = maternal_model.classes_
#         prob_dict = {str(label): float(prob) for label, prob in zip(class_labels, probabilities)}
#     except Exception:
#         prob_dict = None  # Model doesn’t support probability output

#     return {
#         "prediction": str(prediction),
#         "probabilities": prob_dict
#     }

# File: /services/maternal_service.py
# File: /services/maternal_service.py

import joblib
import os
import pandas as pd
import shap
import numpy as np  # ✅ for safe flattening

# Path to maternal model
MODEL_PATH = os.path.join("models", "maternal_risk_model.pkl")

# Load model once at startup
maternal_model = joblib.load(MODEL_PATH)

# Required columns in exact order used during training
COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

# Extract classifier from pipeline
classifier = maternal_model.named_steps["classifier"]

# Create SHAP explainer (TreeExplainer works well for RandomForest)
explainer = shap.TreeExplainer(classifier)


def predict_maternal(features: list):
    """
    Predict maternal health risk and return probabilities + SHAP values.

    Args:
        features (list): Input values [Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]

    Returns:
        dict: Prediction result.
    """
    # Build input DataFrame
    input_df = pd.DataFrame([features], columns=COLUMNS)

    # Prediction
    prediction = maternal_model.predict(input_df)[0]
    probabilities = maternal_model.predict_proba(input_df)[0]

    # SHAP values
    shap_values = explainer.shap_values(input_df)

    # Handle binary vs multiclass classifiers
    if isinstance(shap_values, list):  # Multiclass -> pick predicted class
        class_index = list(classifier.classes_).index(prediction)
        shap_for_instance = shap_values[class_index][0]
    else:  # Binary
        shap_for_instance = shap_values[0]

    # Map feature names to SHAP values (flatten arrays safely)
    feature_importance = {
        col: float(np.ravel(val)[0]) for col, val in zip(COLUMNS, shap_for_instance)
    }

    return {
        "prediction": str(prediction),
        "probabilities": {
            str(cls): float(prob) for cls, prob in zip(classifier.classes_, probabilities)
        },
        "shap_values": feature_importance,
    }
