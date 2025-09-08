from flask import Blueprint, request, jsonify
from services.glucose_service import predict_glucose

glucose_bp = Blueprint("glucose", __name__)

@glucose_bp.route("", methods=["POST"])
def glucose_predict():
    """
    POST /api/glucose
    Handles glucose prediction with TFT model and risk assessment
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is empty"}), 400

        # Check if this is multiple patients (has "patients" array)
        if "patients" in data:
            # Multiple patients format
            patients = data["patients"]
            if not isinstance(patients, list):
                return jsonify({"error": "'patients' must be an array"}), 400
            
            results = []
            for i, patient in enumerate(patients):
                if not isinstance(patient, dict):
                    return jsonify({"error": f"Patient {i+1} must be an object"}), 400
                
                # Check required fields for glucose prediction
                if "patient_id" not in patient:
                    return jsonify({"error": f"Patient {i+1} missing 'patient_id'"}), 400
                
                if "data" not in patient:
                    return jsonify({"error": f"Patient {i+1} missing 'data' array"}), 400
                
                # Get prediction
                try:
                    result = predict_glucose(
                        patient["patient_id"], 
                        patient["data"], 
                        patient.get("contexts"),
                        patient.get("risk_horizons")
                    )
                    result["patient_id"] = patient["patient_id"]
                    results.append(result)
                except Exception as e:
                    results.append({
                        "patient_id": patient["patient_id"],
                        "error": str(e)
                    })
            
            # Calculate cohort statistics
            successful_predictions = [r for r in results if "error" not in r]
            if successful_predictions:
                risk_scores = [r["risk_assessment"]["overall_risk_score"] for r in successful_predictions]
                risk_levels = [r["risk_assessment"]["overall_risk_level"] for r in successful_predictions]
                
                cohort_stats = {
                    "total_patients": len(results),
                    "successful_predictions": len(successful_predictions),
                    "failed_predictions": len(results) - len(successful_predictions),
                    "average_risk_score": sum(risk_scores) / len(risk_scores),
                    "high_risk_count": risk_levels.count("high"),
                    "moderate_risk_count": risk_levels.count("moderate"),
                    "low_risk_count": risk_levels.count("low"),
                    "high_risk_percentage": (risk_levels.count("high") / len(risk_levels)) * 100
                }
            else:
                cohort_stats = {
                    "total_patients": len(results),
                    "successful_predictions": 0,
                    "failed_predictions": len(results),
                    "error": "No successful predictions"
                }
            
            return jsonify({
                "cohort_statistics": cohort_stats,
                "patient_predictions": results
            })
        
        else:
            # Single patient format
            if "patient_id" not in data:
                return jsonify({"error": "Missing 'patient_id'"}), 400
            
            if "data" not in data:
                return jsonify({"error": "Missing 'data' array"}), 400
            
            # Get prediction
            result = predict_glucose(
                data["patient_id"], 
                data["data"], 
                data.get("contexts"),
                data.get("risk_horizons")
            )
            return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@glucose_bp.route("/cohort", methods=["POST"])
def glucose_cohort_predict():
    """
    POST /api/glucose/cohort
    Batch prediction for multiple patients (cohort analysis)
    """
    try:
        data = request.get_json()

        if not data or "patients" not in data:
            return jsonify({"error": "Request body must contain 'patients' array"}), 400

        patients = data["patients"]
        if not isinstance(patients, list) or len(patients) == 0:
            return jsonify({"error": "Patients must be a non-empty array"}), 400

        results = []
        for i, patient in enumerate(patients):
            try:
                # Check required fields
                if "patient_id" not in patient:
                    results.append({
                        "patient_id": f"patient_{i}",
                        "error": "Missing 'patient_id'"
                    })
                    continue
                
                if "data" not in patient:
                    results.append({
                        "patient_id": patient["patient_id"],
                        "error": "Missing 'data' array"
                    })
                    continue

                # Get prediction
                prediction_result = predict_glucose(
                    patient["patient_id"], 
                    patient["data"], 
                    patient.get("contexts"),
                    patient.get("risk_horizons")
                )
                
                # Add patient_id to result
                prediction_result["patient_id"] = patient["patient_id"]
                results.append(prediction_result)
                
            except Exception as e:
                results.append({
                    "patient_id": patient.get("patient_id", f"patient_{i}"),
                    "error": str(e)
                })

        # Calculate cohort statistics
        successful_predictions = [r for r in results if "error" not in r]
        if successful_predictions:
            risk_scores = [r["risk_assessment"]["overall_risk_score"] for r in successful_predictions]
            risk_levels = [r["risk_assessment"]["overall_risk_level"] for r in successful_predictions]
            
            cohort_stats = {
                "total_patients": len(patients),
                "successful_predictions": len(successful_predictions),
                "failed_predictions": len(results) - len(successful_predictions),
                "average_risk_score": sum(risk_scores) / len(risk_scores),
                "high_risk_count": risk_levels.count("high"),
                "moderate_risk_count": risk_levels.count("moderate"),
                "low_risk_count": risk_levels.count("low"),
                "high_risk_percentage": (risk_levels.count("high") / len(risk_levels)) * 100
            }
        else:
            cohort_stats = {
                "total_patients": len(patients),
                "successful_predictions": 0,
                "failed_predictions": len(results),
                "error": "No successful predictions"
            }

        return jsonify({
            "cohort_statistics": cohort_stats,
            "patient_predictions": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
