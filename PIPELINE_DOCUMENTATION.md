# Glucose Prediction Pipeline Documentation

## Overview

This pipeline combines a trained Temporal Fusion Transformer (TFT) model with a rule-based risk classifier to provide glucose predictions and comprehensive risk assessment with full explainability.

## Components

### 1. TFT Model (TemporalFusionTransformer)
- **Purpose**: Generates 90-day glucose forecasts with uncertainty quantification
- **Input**: 60 days of historical data
- **Output**: Quantile predictions (p10, p50, p90) for 90 days
- **Architecture**: Encoder-decoder with attention mechanism

### 2. Rule-Based Risk Classifier
- **Purpose**: Assesses risk of hypo/hyperglycemia events
- **Features**: Continuous risk scoring with component-level explainability
- **Output**: Risk scores, explanations, and actionable recommendations

## Data Requirements

### Input Data Format
The pipeline expects a CSV file with ALL engineered features from the notebook:

```csv
patient_id,date,g_mean,g_std,pct_hypo,pct_hyper,insulin_dose,insulin_adherence,sleep_quality,exercise_flag,meal_variability,stress_index,illness_flag,missed_insulin,hypo_past7d,hyper_past7d,g_mean_lag1,g_mean_7d_mean,g_mean_14d_std,insulin_dose_lag1,insulin_dose_lag2,insulin_dose_lag3,insulin_adherence_7d_mean,g_mean_14d_mean,g_mean_30d_mean,weekday,is_weekend
```

### Required Features (EXACTLY like notebook)

#### Time-Varying Known Features (2 features)
- `weekday`: Day of week (0-6, Monday=0)
- `is_weekend`: Weekend flag (0/1, Saturday/Sunday=1)

#### Time-Varying Unknown Features (21 features)
- `g_mean`: Mean daily glucose (mg/dL)
- `g_std`: Glucose standard deviation (mg/dL)
- `insulin_dose`: Daily insulin dose in units
- `insulin_adherence`: Adherence rate (0-1)
- `sleep_quality`: Sleep quality score (0-1)
- `exercise_flag`: Exercise performed (0/1)
- `meal_variability`: Meal timing variability (0-1)
- `stress_index`: Stress level (0-1)
- `illness_flag`: Illness present (0/1)
- `missed_insulin`: Insulin dose missed (0/1)
- `g_mean_lag1`: Previous day's glucose mean
- `g_mean_7d_mean`: 7-day rolling mean of glucose
- `g_mean_14d_std`: 14-day rolling std of glucose
- `insulin_dose_lag1`: Previous day's insulin dose
- `insulin_dose_lag2`: 2 days ago insulin dose
- `insulin_dose_lag3`: 3 days ago insulin dose
- `insulin_adherence_7d_mean`: 7-day rolling mean adherence
- `g_mean_14d_mean`: 14-day rolling mean of glucose
- `g_mean_30d_mean`: 30-day rolling mean of glucose
- `hypo_past7d`: Hypoglycemic events in past 7 days
- `hyper_past7d`: Hyperglycemic events in past 7 days

#### Static Features
- `patient_id`: Unique patient identifier

### Data Requirements
- **Minimum**: 60 days of consecutive data per patient
- **Format**: Daily observations (one row per day)
- **Date column**: Must be parseable by pandas
- **Missing values**: Should be handled before input

## Usage

### Command Line Interface

```bash
python glucose_prediction_pipeline.py \
    --data_path data.csv \
    --patient_id 123 \
    --model_path tft_epoch50.pth \
    --output_dir results/ \
    --contexts '{"insulin_adherence": 0.8, "sleep_quality": 0.6, "insulin_dose": 8.0}'
```

### Python API

```python
from glucose_prediction_pipeline import GlucosePredictionPipeline

# Initialize pipeline
pipeline = GlucosePredictionPipeline(
    model_path='tft_epoch50.pth',
    device='cpu'
)

# Load data
df = pd.read_csv('data.csv')

# Define context factors
contexts = {
    'insulin_adherence': 0.8,
    'sleep_quality': 0.6,
    'insulin_dose': 8.0
}

# Run pipeline
results = pipeline.run_full_pipeline(df, patient_id=123, contexts=contexts)
```

## Output Structure

### JSON Output Format

```json
{
  "patient_id": 123,
  "prediction_metadata": {
    "patient_id": 123,
    "prediction_horizon_days": 90,
    "quantiles": [0.1, 0.5, 0.9],
    "model_used": "TFT_epoch50",
    "prediction_timestamp": "2024-01-15T10:30:00"
  },
  "glucose_predictions": {
    "horizons_days": [1, 2, 3, ..., 90],
    "p10_quantile": [85.2, 87.1, 89.3, ...],
    "p50_quantile": [120.5, 122.1, 124.8, ...],
    "p90_quantile": [165.3, 168.2, 171.5, ...]
  },
  "risk_assessment": {
    "overall_risk_score": 0.45,
    "overall_risk_level": "moderate",
    "horizon_risks": {
      "horizon_1d": {
        "risk_score": 0.42,
        "risk_level": "moderate",
        "hypo_risk": 0.15,
        "hyper_risk": 0.28,
        "volatility_risk": 0.12,
        "trend_low_risk": 0.05,
        "trend_high_risk": 0.08
      }
    },
    "context_factors": {
      "insulin_adherence": {
        "value": 0.8,
        "impact": "low_risk",
        "multiplier": 1.0
      }
    },
    "recommendations": [
      "⚠️ MODERATE RISK: Increase monitoring frequency",
      "📝 Review insulin dosing with healthcare provider"
    ]
  }
}
```

## Test Dataset Generation

### Synthetic Data Generator

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_test_dataset(n_patients=10, n_days=90, seed=42):
    """Generate synthetic test dataset matching pipeline requirements"""
    np.random.seed(seed)
    
    records = []
    base_date = datetime(2024, 1, 1)
    
    for pid in range(n_patients):
        # Patient-specific parameters
        g_base = np.random.normal(120, 15)
        ins_base = np.random.normal(40, 10)
        adherence = np.random.uniform(0.7, 1.0)
        
        for day in range(n_days):
            date = base_date + timedelta(days=day)
            
            # Generate realistic daily values
            g_mean = g_base + np.random.normal(0, 5)
            g_std = np.random.uniform(15, 35)
            
            # Calculate percentages
            pct_hypo = max(0, np.random.beta(2, 20) * 100)
            pct_hyper = max(0, np.random.beta(2, 10) * 100)
            
            # Lifestyle factors
            sleep_quality = np.random.uniform(0.5, 1.0)
            exercise_flag = np.random.binomial(1, 0.3)
            meal_variability = np.random.uniform(0, 1)
            stress_index = np.random.uniform(0, 1)
            illness_flag = np.random.binomial(1, 0.05)
            missed_insulin = np.random.binomial(1, 0.05)
            
            # Insulin dose with adherence
            if missed_insulin:
                insulin_dose = 0
                adherence = max(0.5, adherence - 0.1)
            else:
                insulin_dose = ins_base * adherence
                adherence = min(1.0, adherence + 0.02)
            
            # Past 7-day events (simplified)
            hypo_past7d = np.random.poisson(0.5)
            hyper_past7d = np.random.poisson(1.0)
            
            records.append({
                'patient_id': pid,
                'date': date.strftime('%Y-%m-%d'),
                'g_mean': round(g_mean, 1),
                'g_std': round(g_std, 1),
                'pct_hypo': round(pct_hypo, 1),
                'pct_hyper': round(pct_hyper, 1),
                'insulin_dose': round(insulin_dose, 1),
                'insulin_adherence': round(adherence, 2),
                'sleep_quality': round(sleep_quality, 2),
                'exercise_flag': exercise_flag,
                'meal_variability': round(meal_variability, 2),
                'stress_index': round(stress_index, 2),
                'illness_flag': illness_flag,
                'missed_insulin': missed_insulin,
                'hypo_past7d': hypo_past7d,
                'hyper_past7d': hyper_past7d
            })
    
    return pd.DataFrame(records)

# Generate and save test dataset
test_df = generate_test_dataset(n_patients=5, n_days=90)
test_df.to_csv('test_dataset.csv', index=False)
print(f"Generated test dataset with {len(test_df)} records")
```

### Example Test Data

```python
# Quick test with minimal data
import pandas as pd
from datetime import datetime, timedelta

def create_minimal_test_data():
    """Create minimal test data for one patient"""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
    
    data = []
    for i, date in enumerate(dates):
        data.append({
            'patient_id': 0,
            'date': date.strftime('%Y-%m-%d'),
            'g_mean': 120 + 10 * np.sin(i * 0.1) + np.random.normal(0, 5),
            'g_std': 20 + np.random.uniform(-5, 5),
            'pct_hypo': max(0, np.random.beta(2, 20) * 100),
            'pct_hyper': max(0, np.random.beta(2, 10) * 100),
            'insulin_dose': 40 + np.random.normal(0, 5),
            'insulin_adherence': 0.8 + np.random.uniform(-0.1, 0.1),
            'sleep_quality': 0.7 + np.random.uniform(-0.2, 0.2),
            'exercise_flag': np.random.binomial(1, 0.3),
            'meal_variability': np.random.uniform(0, 1),
            'stress_index': np.random.uniform(0, 1),
            'illness_flag': np.random.binomial(1, 0.05),
            'missed_insulin': np.random.binomial(1, 0.05),
            'hypo_past7d': np.random.poisson(0.5),
            'hyper_past7d': np.random.poisson(1.0)
        })
    
    return pd.DataFrame(data)

# Create and save minimal test data
test_data = create_minimal_test_data()
test_data.to_csv('minimal_test_data.csv', index=False)
```

## Model Files Required

### 1. TFT Model Checkpoint (`tft_epoch50.pth`)
- Trained TFT model weights and scalers
- Contains model state dict, scalers, and training history
- Generated from notebook training (scalers are saved WITH the model)
- No separate scalers file needed

## Risk Assessment Details

### Risk Components

1. **Hypoglycemia Risk**
   - Based on distance below 80 mg/dL threshold
   - Considers both median and 10th percentile predictions
   - Weight: 0.4

2. **Hyperglycemia Risk**
   - Based on distance above 160 mg/dL threshold
   - Considers both median and 90th percentile predictions
   - Weight: 0.4

3. **Volatility Risk**
   - Based on prediction interval width (p90 - p10)
   - Higher uncertainty increases risk
   - Weight: 0.2

4. **Context Factors**
   - Insulin adherence: <0.7 (high risk), <0.85 (moderate)
   - Sleep quality: <0.5 (high risk), <0.7 (moderate)
   - Insulin dose: <5 units (high risk)

### Risk Levels
- **Low**: < 0.3
- **Moderate**: 0.3 - 0.6
- **High**: > 0.6

## Troubleshooting

### Common Issues

1. **Insufficient Data**
   - Error: "Insufficient data for patient X"
   - Solution: Ensure at least 60 days of data per patient

2. **Model Loading Error**
   - Error: "Error loading model"
   - Solution: Check model file path and format

3. **Scaler Loading Error**
   - Error: "Error loading scalers"
   - Solution: Ensure scalers file exists and is valid

4. **Memory Issues**
   - Error: CUDA out of memory
   - Solution: Use `--device cpu` for CPU inference

### Performance Tips

1. **Batch Processing**: Process multiple patients in sequence
2. **Memory Management**: Use CPU for large datasets
3. **Data Validation**: Check data format before processing

## Dependencies

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

## License

This pipeline is part of the Medical Prediction System project.
