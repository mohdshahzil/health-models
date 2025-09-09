## Health Models API

A Flask-based API providing machine learning-powered predictions and risk assessments for:
- Maternal health
- Cardiovascular risk
- Glucose forecasting and diabetes risk

The service exposes REST endpoints with interactive Swagger docs and includes example scripts and pre-trained models.

### Clinician-friendly overview
This project provides decision-support signals from routinely collected vitals, labs, and daily patterns. It does not replace clinical judgment; it summarizes risk in plain language and surfaces the most influential factors for each patient.

- Maternal health: estimates likelihood of maternal risk based on age, blood pressure, blood sugar, temperature, and heart rate; also shows which measurements are most influencing the result.
- Cardiovascular risk (90-day deterioration): combines vitals, cholesterol/glucose and lifestyle signals (adherence, activity, sleep, stress) to classify near-term risk as LOW/MEDIUM/HIGH, with a probability score and driving factors.
- Glucose forecasting (90 days) and diabetes risk: uses 60+ days of daily glucose summaries, insulin patterns, lifestyle/context signals to forecast glucose trajectories with uncertainty and assess near-term risk; provides horizon-specific risk and actionable recommendations.

Key benefits for clinical use:
- Plain-language risk levels with probabilities
- Per-patient contributing factors to support explanation and shared decision making
- Batch processing to triage multiple patients at once (for glucose)

---

## Project structure

```
health-models/
  app.py                      # Flask app factory and entrypoint
  swagger_ui.py               # Swagger documentation and schemas
  routes/                     # API route handlers (Flask blueprints)
    maternal.py
    cardiovascular.py
    glucose.py
  services/                   # Model loading and prediction logic
    maternal_service.py
    cardiovascular_service.py
    glucose_service.py
  scripts/                    # Utilities and data/pipeline helpers
    csv_to_glucose_json.py
    generate_test_data.py
    glucose_prediction_pipeline.py
  models/                     # Pre-trained model artifacts and sample data
    maternal_risk_model.pkl
    cardiovascular_risk_model.pkl
    cardiovascular_scaler.pkl
    cardiovascular_feature_selector.pkl
    cardiovascular_selected_features.pkl
    glucose_model.pth
    high_risk_glucose_60_days.json
    patient_data.csv
    Maternal_explainableai.ipynb
    cardiovascular_risk_demo.ipynb
    synthetic_glucose_risk.ipynb
  requirements.txt
  PIPELINE_DOCUMENTATION.md
  README.md
```

---

## How it works (clinical terms)

### Maternal health
- What it uses: age, blood pressure (systolic/diastolic), blood sugar, body temperature, heart rate.
- What it outputs: a predicted class (e.g., higher vs lower risk), class probabilities, and a list of top contributing measurements.
- How to interpret: HIGH probability or a rising contribution from blood pressure/blood sugar warrants review; contributing features help contextualize why.

Under the hood: a trained model (Random Forest within a pipeline) is loaded from `models/maternal_risk_model.pkl`. Feature contributions are computed with SHAP values and returned per request.

Related notebook: `models/Maternal_explainableai.ipynb` demonstrates model behavior and explains how each feature influences risk, using example cases and plots.

### Cardiovascular risk (90-day deterioration)
- What it uses: age, sex, diabetes/hypertension status, vitals (BP, HR), cholesterol, glucose, medication adherence, exercise minutes, diet score, stress, weight, oxygen saturation, temperature, sleep hours.
- What it outputs: LOW/MEDIUM/HIGH risk classification with a probability score and feature-attribution summary.
- How to interpret: prioritize patients with HIGH risk; use contributing factors (e.g., elevated BP trends, suboptimal adherence) to guide targeted interventions.

Under the hood: a model and preprocessing artifacts (scaler, feature selector) are loaded from `models/*.pkl`. If artifacts are missing, a reasonable demo model is created at runtime. The service augments current measurements with trend/variability features to reflect recent trajectory, then scores risk. SHAP values are provided when available.

Related notebook: `models/cardiovascular_risk_demo.ipynb` (derivative of `cardiovascular_risk_model.ipynb`) shows training/evaluation and the effect of features and thresholds on near-term risk classification.

### Glucose forecasting and risk (90 days)
- What it uses: at least 60 consecutive days of daily glucose summaries, insulin dose/adherence, sleep/exercise, meal timing variability, stress/illness, plus engineered lag and rolling-window features.
- What it outputs: 90-day glucose forecasts with uncertainty bands (p10, p50, p90), overall and horizon-specific risk levels (e.g., 7/14/30/60/90 days), and concise recommendations (e.g., review insulin dosing, increase monitoring).
- How to interpret: use the median forecast to understand trend; a HIGH risk classification at short horizons suggests earlier clinical action. Recommendations summarize common next steps and are not prescriptive.

Under the hood: the service loads a Temporal Fusion Transformer (TFT) model from `models/glucose_model.pth` and runs a feature pipeline defined in `scripts/glucose_prediction_pipeline.py`. Risk assessment applies clinically-inspired rules to forecasts and recent context; outputs are fully JSON-serializable.

Related notebook: `models/synthetic_glucose_risk.ipynb` demonstrates synthetic patient trajectories, model forecasts, and how context (e.g., lower adherence, poor sleep) shifts risk.

---

## Getting started

### Prerequisites
- Python 3.10+
- pip

Note: If installing `torch` on Windows, you may need a specific wheel (CPU/GPU) from the official PyTorch site.

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the API (development)

```bash
$env:PORT=5000
python app.py
```

- App runs on `http://localhost:5000/`
- Health check: `GET /health`
- Swagger docs: `GET /docs/`

### Run with Gunicorn (production-like)

Render uses port 10000 by default; locally you can run:

```bash
# On Windows, use "waitress" or run via WSL; Gunicorn is Linux/macOS focused.
# Example (WSL/Linux):
PORT=10000 gunicorn app:app --bind 0.0.0.0:$PORT --workers 2
```

---

## API overview

Base URL: `/api`

- `POST /api/maternal` — Maternal health risk prediction
- `POST /api/cardiovascular` — Cardiovascular risk prediction (single or multiple)
- `POST /api/glucose` — TFT glucose forecast and risk (single or multiple)
- `POST /api/glucose/cohort` — Strict batch with cohort stats

Interactive documentation: `GET /docs/`

Root endpoint:
- `GET /` → basic info and links
- `GET /health` → service status

---

## Endpoints and examples

### Maternal — POST `/api/maternal`
Required JSON fields: `Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`

Example:
```bash
curl -X POST http://localhost:5000/api/maternal \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 28,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 90,
    "BodyTemp": 36.6,
    "HeartRate": 72
  }'
```
Response (abridged):
```json
{
  "prediction": "0",
  "probabilities": {"0": 0.85, "1": 0.15},
  "shap_values": {"Age": -0.01, "SystolicBP": 0.02, ...}
}
```

### Cardiovascular — POST `/api/cardiovascular`
Accepts either a single patient object or `{ "patients": [ ... ] }`.

Required fields per patient (single request):
`age, gender, diabetes, hypertension, systolic_bp, diastolic_bp, heart_rate, cholesterol, glucose, medication_adherence, exercise_minutes, diet_score, stress_level, weight_kg, oxygen_saturation, temperature_c, sleep_hours`

- For missing fields, defaults are applied for convenience in this endpoint.

Example (single):
```bash
curl -X POST http://localhost:5000/api/cardiovascular \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": "Male",
    "diabetes": "No",
    "hypertension": "Yes",
    "systolic_bp": 145,
    "diastolic_bp": 92,
    "heart_rate": 78,
    "cholesterol": 220,
    "glucose": 110,
    "medication_adherence": 75,
    "exercise_minutes": 60,
    "diet_score": 6,
    "stress_level": 6,
    "weight_kg": 85,
    "oxygen_saturation": 97,
    "temperature_c": 36.9,
    "sleep_hours": 6
  }'
```
Response (abridged):
```json
{
  "prediction": "0",
  "risk_score": 0.32,
  "risk_level": "MEDIUM",
  "probabilities": {"0": 0.68, "1": 0.32},
  "shap_values": {"systolic_bp": 0.04, ...},
  "model_info": {"model_type": "RandomForest", "prediction_horizon": "90 days"}
}
```

Example (multiple in one request):
```bash
curl -X POST http://localhost:5000/api/cardiovascular \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {"patient_id": "P001", "age": 60, "gender": "Male", "diabetes": "No", "hypertension": "Yes", "systolic_bp": 150, "diastolic_bp": 95, "heart_rate": 80, "cholesterol": 230, "glucose": 115, "medication_adherence": 70, "exercise_minutes": 45, "diet_score": 5, "stress_level": 7, "weight_kg": 90, "oxygen_saturation": 96, "temperature_c": 37.0, "sleep_hours": 6},
      {"patient_id": "P002", "age": 45, "gender": "Female", "diabetes": "No", "hypertension": "No", "systolic_bp": 120, "diastolic_bp": 80, "heart_rate": 70, "cholesterol": 190, "glucose": 95, "medication_adherence": 90, "exercise_minutes": 150, "diet_score": 7, "stress_level": 4, "weight_kg": 68, "oxygen_saturation": 99, "temperature_c": 36.7, "sleep_hours": 7}
    ]
  }'
```

### Glucose — POST `/api/glucose`
Accepts either a single patient object or `{ "patients": [ ... ] }`.

Input requires at least 60 consecutive days of data with engineered features. Minimum fields per daily record include:
- Core: `g_mean, g_std, pct_hypo, pct_hyper`
- Insulin: `insulin_dose, insulin_adherence, missed_insulin`
- Lifestyle: `sleep_quality, exercise_flag, meal_variability, stress_index, illness_flag`
- History: `hypo_past7d, hyper_past7d`
- Glucose lags: `g_mean_lag1, g_mean_7d_mean, g_mean_14d_std, g_mean_14d_mean, g_mean_30d_mean`
- Insulin lags: `insulin_dose_lag1, insulin_dose_lag2, insulin_dose_lag3, insulin_adherence_7d_mean`
- Temporal: `weekday, is_weekend`

Example (single):
```bash
curl -X POST http://localhost:5000/api/glucose \
  -H "Content-Type: application/json" \
  -d @models/high_risk_glucose_60_days.json
```
Response (abridged):
```json
{
  "prediction_metadata": {"prediction_horizon_days": 90, "quantiles": [0.1,0.5,0.9]},
  "glucose_predictions": {"horizons_days": [...], "p10_quantile": [...], "p50_quantile": [...], "p90_quantile": [...]},
  "risk_assessment": {"overall_risk_score": 0.45, "overall_risk_level": "moderate", "recommendations": ["..."]},
  "model_info": {"model_type": "TemporalFusionTransformer", "features_used": 23}
}
```

Batch in one request: provide `{ "patients": [ {"patient_id", "data", "contexts?", "risk_horizons?"}, ... ] }` to `/api/glucose`.

Strict batch variant with cohort stats: `POST /api/glucose/cohort`.

---

## Swagger documentation
- Visit `http://localhost:5000/docs/`
- Namespaces: `maternal`, `cardiovascular`, `glucose`
- Schemas include full input/output models and rich endpoint descriptions

---

## Scripts
- `scripts/generate_test_data.py`: Generate synthetic data for testing
- `scripts/csv_to_glucose_json.py`: Convert CSV to the expected glucose JSON schema
- `scripts/glucose_prediction_pipeline.py`: TFT pipeline used by the glucose service

Example (convert CSV to JSON):
```bash
python scripts/csv_to_glucose_json.py \
  --input models/patient_data.csv \
  --output models/high_risk_glucose_60_days.json
```

---

## Models
Model artifacts are stored in `models/`.
- Maternal: `maternal_risk_model.pkl`
- Cardiovascular: `cardiovascular_risk_model.pkl` plus `cardiovascular_scaler.pkl`, `cardiovascular_feature_selector.pkl`, `cardiovascular_selected_features.pkl`
- Glucose: `glucose_model.pth`

If cardiovascular model files are missing, a demo model is created at runtime with sensible defaults.

---

## Environment variables
- `PORT` — server port (default: 10000 in app, set to 5000 locally)

---

## Development notes
- SHAP is used for explainability in maternal and cardiovascular services
- Glucose service initializes a TFT pipeline; ensure `glucose_model.pth` exists
- Input validation for glucose requires minimum 60 days and strict field presence

---

## Health check
- `GET /health` returns status, timestamp, and service availability

---
