from flask import Blueprint, request, jsonify
from services.cardiovascular_service import predict_cardiovascular

cardiovascular_bp = Blueprint("cardiovascular", __name__)

@cardiovascular_bp.route("", methods=["POST"])
def cardiovascular_predict():
    """
    POST /api/cardiovascular
    """
    try:
        data = request.get_json()

        required_fields = ["age", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol", 
                          "glucose", "medication_adherence", "exercise_minutes", "diet_score", 
                          "stress_level", "weight_kg", "oxygen_saturation", "temperature_c", "sleep_hours"]

        if not data:
            return jsonify({"error": "Request body is empty"}), 400

        # Check missing fields
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Convert to feature list in correct order
        features = [data[f] for f in required_fields]

        # Optional: Extract patient_data for trend analysis
        patient_data = data.get("patient_data", None)

        result = predict_cardiovascular(features, patient_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
