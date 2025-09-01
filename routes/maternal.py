from flask import Blueprint, request, jsonify
from services.maternal_service import predict_maternal

maternal_bp = Blueprint("maternal", __name__)

@maternal_bp.route("", methods=["POST"])
def maternal_predict():
    """
    POST /api/maternal
    """
    try:
        data = request.get_json()

        required_fields = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

        if not data:
            return jsonify({"error": "Request body is empty"}), 400

        # Check missing fields
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Convert to feature list in correct order
        features = [data[f] for f in required_fields]

        result = predict_maternal(features)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
