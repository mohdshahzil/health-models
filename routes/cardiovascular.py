from flask import Blueprint, request, jsonify
from services.cardiovascular_service import predict_cardiovascular

cardiovascular_bp = Blueprint("cardiovascular", __name__)

@cardiovascular_bp.route("", methods=["POST"])
def cardiovascular_predict():
    """
    POST /api/cardiovascular
    Handles both single patient and multiple patients prediction
    """
    try:
        data = request.get_json()

        required_fields = ["age", "gender", "diabetes", "hypertension", "systolic_bp", "diastolic_bp", 
                          "heart_rate", "cholesterol", "glucose", "medication_adherence", 
                          "exercise_minutes", "diet_score", "stress_level", "weight_kg", 
                          "oxygen_saturation", "temperature_c", "sleep_hours"]

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
                
                # Check missing fields and provide defaults for each patient
                missing = [f for f in required_fields if f not in patient]
                if missing:
                    defaults = {
                        "age": 50,
                        "gender": "Male",
                        "diabetes": "No",
                        "hypertension": "No",
                        "systolic_bp": 120,
                        "diastolic_bp": 80,
                        "heart_rate": 70,
                        "cholesterol": 200,
                        "glucose": 100,
                        "medication_adherence": 80,
                        "exercise_minutes": 150,
                        "diet_score": 6,
                        "stress_level": 5,
                        "weight_kg": 70,
                        "oxygen_saturation": 98,
                        "temperature_c": 36.8,
                        "sleep_hours": 7
                    }
                    
                    for field in missing:
                        patient[field] = defaults[field]

                # Convert to feature list in correct order
                features = [patient[f] for f in required_fields]
                result = predict_cardiovascular(features)
                
                # Add patient_id if provided
                if "patient_id" in patient:
                    result["patient_id"] = patient["patient_id"]
                else:
                    result["patient_id"] = f"P{i+1:03d}"
                
                results.append(result)
            
            # Calculate cohort statistics
            risk_scores = [r["risk_score"] for r in results]
            high_risk_count = sum(1 for score in risk_scores if score > 0.7)
            medium_risk_count = sum(1 for score in risk_scores if 0.3 <= score <= 0.7)
            low_risk_count = sum(1 for score in risk_scores if score < 0.3)
            
            cohort_stats = {
                "total_patients": len(results),
                "average_risk_score": sum(risk_scores) / len(risk_scores),
                "high_risk_patients": high_risk_count,
                "medium_risk_patients": medium_risk_count,
                "low_risk_patients": low_risk_count,
                "high_risk_percentage": (high_risk_count / len(results)) * 100
            }
            
            return jsonify({
                "cohort_statistics": cohort_stats,
                "patient_predictions": results
            })
        
        else:
            # Single patient format
            # Check missing fields and provide defaults
            missing = [f for f in required_fields if f not in data]
            if missing:
                defaults = {
                    "age": 50,
                    "gender": "Male",
                    "diabetes": "No",
                    "hypertension": "No",
                    "systolic_bp": 120,
                    "diastolic_bp": 80,
                    "heart_rate": 70,
                    "cholesterol": 200,
                    "glucose": 100,
                    "medication_adherence": 80,
                    "exercise_minutes": 150,
                    "diet_score": 6,
                    "stress_level": 5,
                    "weight_kg": 70,
                    "oxygen_saturation": 98,
                    "temperature_c": 36.8,
                    "sleep_hours": 7
                }
                
                for field in missing:
                    data[field] = defaults[field]

            # Convert to feature list in correct order
            features = [data[f] for f in required_fields]
            result = predict_cardiovascular(features)
            return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@cardiovascular_bp.route("/cohort", methods=["POST"])
def cardiovascular_cohort_predict():
    """
    POST /api/cardiovascular/cohort
    Batch prediction for multiple patients (cohort analysis)
    """
    try:
        data = request.get_json()

        if not data or "patients" not in data:
            return jsonify({"error": "Request body must contain 'patients' array"}), 400

        patients = data["patients"]
        if not isinstance(patients, list) or len(patients) == 0:
            return jsonify({"error": "Patients must be a non-empty array"}), 400

        required_fields = ["patient_id", "age", "gender", "diabetes", "hypertension", "systolic_bp", "diastolic_bp", 
                          "heart_rate", "cholesterol", "glucose", "medication_adherence", 
                          "exercise_minutes", "diet_score", "stress_level", "weight_kg", 
                          "oxygen_saturation", "temperature_c", "sleep_hours"]

        results = []
        for i, patient in enumerate(patients):
            try:
                # Check missing fields for this patient
                missing = [f for f in required_fields if f not in patient]
                if missing:
                    results.append({
                        "patient_id": patient.get("patient_id", f"patient_{i}"),
                        "error": f"Missing fields: {missing}"
                    })
                    continue

                # Convert to feature list in correct order (excluding patient_id)
                features = [patient[f] for f in required_fields[1:]]  # Skip patient_id
                
                # Get prediction
                prediction_result = predict_cardiovascular(features)
                
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
            risk_scores = [r["risk_score"] for r in successful_predictions]
            risk_levels = [r["risk_level"] for r in successful_predictions]
            
            cohort_stats = {
                "total_patients": len(patients),
                "successful_predictions": len(successful_predictions),
                "failed_predictions": len(results) - len(successful_predictions),
                "average_risk_score": sum(risk_scores) / len(risk_scores),
                "high_risk_count": risk_levels.count("HIGH"),
                "medium_risk_count": risk_levels.count("MEDIUM"),
                "low_risk_count": risk_levels.count("LOW"),
                "high_risk_percentage": (risk_levels.count("HIGH") / len(risk_levels)) * 100
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
