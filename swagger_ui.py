"""
Swagger UI endpoint for Health Models API
This module provides a comprehensive Swagger documentation interface
"""

from flask import Flask, render_template_string
from flask_restx import Api, Resource, fields, Namespace
import json

def create_swagger_ui(app):
    """
    Add Swagger UI documentation to the Flask app
    """
    
    # Create API instance
    api = Api(
        app,
        version='1.0',
        title='Health Models API',
        description='Comprehensive API for chronic disease prediction and risk assessment',
        doc='/docs/',  # Swagger UI will be available at /docs/
        prefix='/api'
    )
    
    # Create namespaces for each disease model
    maternal_ns = Namespace('maternal', description='Maternal health prediction endpoints')
    cardiovascular_ns = Namespace('cardiovascular', description='Cardiovascular risk prediction endpoints')
    glucose_ns = Namespace('glucose', description='Glucose prediction and diabetes risk assessment endpoints')
    
    api.add_namespace(maternal_ns, path='/maternal')
    api.add_namespace(cardiovascular_ns, path='/cardiovascular')
    api.add_namespace(glucose_ns, path='/glucose')
    
    # Maternal Health Models
    maternal_input_model = api.model('MaternalInput', {
        'patient_id': fields.Integer(required=True, description='Patient identifier', example=123),
        'age': fields.Float(required=True, description='Patient age in years', example=28.0, min=15, max=50),
        'bmi': fields.Float(required=True, description='Body Mass Index', example=24.5, min=15, max=50),
        'systolic_bp': fields.Float(required=True, description='Systolic blood pressure (mmHg)', example=120.0, min=80, max=200),
        'diastolic_bp': fields.Float(required=True, description='Diastolic blood pressure (mmHg)', example=80.0, min=50, max=120),
        'glucose': fields.Float(required=True, description='Blood glucose level (mg/dL)', example=95.0, min=50, max=300),
        'insulin': fields.Float(required=True, description='Insulin level (μU/mL)', example=8.5, min=0, max=50),
        'triceps_skinfold': fields.Float(required=True, description='Triceps skinfold thickness (mm)', example=20.0, min=5, max=50),
        'diabetes_pedigree': fields.Float(required=True, description='Diabetes pedigree function', example=0.672, min=0, max=2.5),
        'pregnancies': fields.Integer(required=True, description='Number of pregnancies', example=1, min=0, max=20)
    })
    
    maternal_output_model = api.model('MaternalOutput', {
        'patient_id': fields.Integer(description='Patient identifier', example=123),
        'prediction': fields.Integer(description='Predicted outcome (0=no diabetes, 1=diabetes)', example=0, enum=[0, 1]),
        'probability': fields.Float(description='Probability of diabetes (0-1)', example=0.15, min=0, max=1),
        'risk_level': fields.String(description='Risk level classification', example='low', enum=['low', 'moderate', 'high']),
        'model_info': fields.Raw(description='Model information and configuration')
    })
    
    # Cardiovascular Health Models
    cardiovascular_input_model = api.model('CardiovascularInput', {
        'patient_id': fields.Integer(required=True, description='Patient identifier', example=123),
        'age': fields.Float(required=True, description='Patient age in years', example=45.0, min=18, max=100),
        'gender': fields.Integer(required=True, description='Gender (1=male, 2=female)', example=1, enum=[1, 2]),
        'height': fields.Float(required=True, description='Height in cm', example=175.0, min=100, max=250),
        'weight': fields.Float(required=True, description='Weight in kg', example=70.0, min=30, max=200),
        'systolic_bp': fields.Float(required=True, description='Systolic blood pressure (mmHg)', example=130.0, min=80, max=250),
        'diastolic_bp': fields.Float(required=True, description='Diastolic blood pressure (mmHg)', example=85.0, min=50, max=150),
        'cholesterol': fields.Float(required=True, description='Total cholesterol (mg/dL)', example=200.0, min=100, max=400),
        'glucose': fields.Float(required=True, description='Blood glucose level (mg/dL)', example=100.0, min=50, max=300),
        'smoking': fields.Integer(required=True, description='Smoking status (0=no, 1=yes)', example=0, enum=[0, 1]),
        'alcohol': fields.Integer(required=True, description='Alcohol consumption (0=no, 1=yes)', example=0, enum=[0, 1]),
        'physical_activity': fields.Integer(required=True, description='Physical activity level (0=low, 1=high)', example=1, enum=[0, 1])
    })
    
    cardiovascular_output_model = api.model('CardiovascularOutput', {
        'patient_id': fields.Integer(description='Patient identifier', example=123),
        'prediction': fields.Integer(description='Predicted cardiovascular risk (0=low, 1=high)', example=0, enum=[0, 1]),
        'probability': fields.Float(description='Probability of cardiovascular disease (0-1)', example=0.25, min=0, max=1),
        'risk_level': fields.String(description='Risk level classification', example='moderate', enum=['low', 'moderate', 'high']),
        'probabilities': fields.Raw(description='Risk probabilities for each class'),
        'shap_values': fields.Raw(description='Feature importance scores'),
        'model_info': fields.Raw(description='Model information')
    })
    
    # Glucose Prediction Models - Complete Documentation
    glucose_data_point_model = api.model('GlucoseDataPoint', {
        'date': fields.String(required=True, description='Date in YYYY-MM-DD format', example='2024-01-01'),
        
        # Core Glucose Metrics
        'g_mean': fields.Float(required=True, description='Mean daily glucose level in mg/dL', example=120.5, min=50, max=500),
        'g_std': fields.Float(required=True, description='Standard deviation of daily glucose readings in mg/dL', example=25.0, min=0, max=100),
        'pct_hypo': fields.Float(required=True, description='Percentage of time spent in hypoglycemia (<70 mg/dL)', example=5.0, min=0, max=100),
        'pct_hyper': fields.Float(required=True, description='Percentage of time spent in hyperglycemia (>180 mg/dL)', example=15.0, min=0, max=100),
        
        # Insulin Management
        'insulin_dose': fields.Float(required=True, description='Daily insulin dose in units', example=40.0, min=0, max=200),
        'insulin_adherence': fields.Float(required=True, description='Insulin adherence rate (0-1, where 1=perfect adherence)', example=0.85, min=0, max=1),
        'missed_insulin': fields.Integer(required=True, description='Whether insulin dose was missed (0=no, 1=yes)', example=0, enum=[0, 1]),
        
        # Lifestyle Factors
        'sleep_quality': fields.Float(required=True, description='Sleep quality score (0-1, where 1=excellent sleep)', example=0.7, min=0, max=1),
        'exercise_flag': fields.Integer(required=True, description='Whether exercise was performed (0=no, 1=yes)', example=1, enum=[0, 1]),
        'meal_variability': fields.Float(required=True, description='Meal timing variability (0-1, where 1=high variability)', example=0.3, min=0, max=1),
        'stress_index': fields.Float(required=True, description='Stress level index (0-1, where 1=high stress)', example=0.4, min=0, max=1),
        'illness_flag': fields.Integer(required=True, description='Whether illness is present (0=no, 1=yes)', example=0, enum=[0, 1]),
        
        # Historical Events
        'hypo_past7d': fields.Float(required=True, description='Number of hypoglycemic events in past 7 days', example=1.0, min=0),
        'hyper_past7d': fields.Float(required=True, description='Number of hyperglycemic events in past 7 days', example=2.0, min=0),
        
        # Engineered Features - Glucose Lags
        'g_mean_lag1': fields.Float(required=True, description='Previous day mean glucose level in mg/dL', example=118.0, min=50, max=500),
        'g_mean_7d_mean': fields.Float(required=True, description='7-day rolling average of mean glucose in mg/dL', example=122.0, min=50, max=500),
        'g_mean_14d_std': fields.Float(required=True, description='14-day rolling standard deviation of glucose in mg/dL', example=22.0, min=0, max=100),
        'g_mean_14d_mean': fields.Float(required=True, description='14-day rolling average of mean glucose in mg/dL', example=120.0, min=50, max=500),
        'g_mean_30d_mean': fields.Float(required=True, description='30-day rolling average of mean glucose in mg/dL', example=125.0, min=50, max=500),
        
        # Engineered Features - Insulin Lags
        'insulin_dose_lag1': fields.Float(required=True, description='Previous day insulin dose in units', example=38.0, min=0, max=200),
        'insulin_dose_lag2': fields.Float(required=True, description='Insulin dose from 2 days ago in units', example=42.0, min=0, max=200),
        'insulin_dose_lag3': fields.Float(required=True, description='Insulin dose from 3 days ago in units', example=40.0, min=0, max=200),
        'insulin_adherence_7d_mean': fields.Float(required=True, description='7-day rolling average of insulin adherence (0-1)', example=0.87, min=0, max=1),
        
        # Temporal Features
        'weekday': fields.Integer(required=True, description='Day of week (0=Monday, 1=Tuesday, ..., 6=Sunday)', example=0, min=0, max=6),
        'is_weekend': fields.Integer(required=True, description='Weekend indicator (0=weekday, 1=weekend)', example=0, enum=[0, 1])
    })
    
    glucose_context_model = api.model('GlucoseContext', {
        'insulin_adherence': fields.Float(description='Current insulin adherence rate (0-1, where 1=perfect adherence)', example=0.85, min=0, max=1),
        'sleep_quality': fields.Float(description='Current sleep quality score (0-1, where 1=excellent sleep)', example=0.7, min=0, max=1),
        'insulin_dose': fields.Float(description='Current insulin dose in units', example=30.0, min=0, max=200)
    })
    
    glucose_input_model = api.model('GlucoseInput', {
        'patient_id': fields.Integer(required=True, description='Patient identifier', example=123),
        'data': fields.List(fields.Nested(glucose_data_point_model), required=True, description='Array of daily patient data (minimum 60 days)'),
        'contexts': fields.Nested(glucose_context_model, description='Context factors for risk assessment'),
        'risk_horizons': fields.List(fields.Integer, description='Risk assessment horizons in days', example=[7, 14, 30, 60, 90])
    })
    
    # Glucose Prediction Output Models
    glucose_prediction_model = api.model('GlucosePrediction', {
        'horizons_days': fields.List(fields.Integer, description='Prediction horizons from 1 to 90 days', example=[1, 2, 3, 4, 5]),
        'p10_quantile': fields.List(fields.Float, description='10th percentile glucose predictions (conservative estimate) in mg/dL', example=[85.2, 87.1, 89.3]),
        'p50_quantile': fields.List(fields.Float, description='50th percentile (median) glucose predictions in mg/dL', example=[120.5, 122.1, 124.8]),
        'p90_quantile': fields.List(fields.Float, description='90th percentile glucose predictions (optimistic estimate) in mg/dL', example=[165.3, 168.2, 171.5])
    })
    
    glucose_risk_assessment_model = api.model('GlucoseRiskAssessment', {
        'overall_risk_score': fields.Float(description='Overall risk score across all horizons (0-1)', example=0.45, min=0, max=1),
        'overall_risk_level': fields.String(description='Overall risk level classification', example='moderate', enum=['low', 'moderate', 'high']),
        'horizon_risks': fields.Raw(description='Risk assessment for each specified horizon (e.g., horizon_7d, horizon_30d)'),
        'context_factors': fields.Raw(description='Impact of context factors on risk assessment'),
        'context_multiplier': fields.Float(description='Overall context multiplier applied', example=1.0),
        'detailed_explanations': fields.Raw(description='Detailed risk component explanations and calculations'),
        'recommendations': fields.List(fields.String, description='Actionable recommendations based on risk assessment', example=['⚠️ MODERATE RISK: Increase monitoring frequency', '📝 Review insulin dosing with healthcare provider'])
    })
    
    glucose_prediction_metadata_model = api.model('GlucosePredictionMetadata', {
        'patient_id': fields.Integer(description='Patient identifier', example=123),
        'prediction_horizon_days': fields.Integer(description='Total prediction horizon in days', example=90),
        'quantiles': fields.List(fields.Float, description='Prediction quantiles used', example=[0.1, 0.5, 0.9]),
        'model_used': fields.String(description='Model identifier', example='TFT_epoch50'),
        'prediction_timestamp': fields.String(description='Timestamp when prediction was made', example='2024-01-15T10:30:00'),
        'denormalization_applied': fields.Boolean(description='Whether predictions were denormalized', example=True)
    })
    
    glucose_model_info_model = api.model('GlucoseModelInfo', {
        'model_type': fields.String(description='Type of model used', example='TemporalFusionTransformer'),
        'prediction_horizon': fields.String(description='Prediction horizon description', example='90 days'),
        'features_used': fields.Integer(description='Number of features used in the model', example=23),
        'risk_assessment': fields.String(description='Risk assessment method', example='Rule-based with explainability')
    })
    
    glucose_output_model = api.model('GlucoseOutput', {
        'patient_id': fields.Integer(description='Patient identifier', example=123),
        'prediction_metadata': fields.Nested(glucose_prediction_metadata_model, description='Prediction metadata including model info and timestamps'),
        'glucose_predictions': fields.Nested(glucose_prediction_model, description='90-day glucose predictions with uncertainty quantiles (p10, p50, p90)'),
        'risk_assessment': fields.Nested(glucose_risk_assessment_model, description='Comprehensive risk assessment with horizon-specific analysis and recommendations'),
        'model_info': fields.Nested(glucose_model_info_model, description='Model information and configuration'),
        'pipeline_timestamp': fields.String(description='Pipeline execution timestamp', example='2024-01-15T10:30:00.305343')
    })
    
    # Error models
    error_model = api.model('Error', {
        'error': fields.String(description='Error message', example='Invalid input data')
    })
    
    # Maternal Health Endpoints
    @maternal_ns.route('')
    class MaternalPrediction(Resource):
        @maternal_ns.expect(maternal_input_model)
        @maternal_ns.marshal_with(maternal_output_model, code=200)
        @maternal_ns.marshal_with(error_model, code=400)
        @maternal_ns.marshal_with(error_model, code=500)
        @maternal_ns.doc('maternal_prediction', description='Predict maternal health and diabetes risk')
        def post(self):
            """
            🤱 Maternal Health & Diabetes Risk Prediction
            
            This endpoint predicts the risk of gestational diabetes in pregnant women using
            comprehensive health metrics and machine learning models.
            
            ## 🎯 **What This Endpoint Does:**
            - **Diabetes risk prediction** for pregnant women
            - **Risk level classification** (low/moderate/high)
            - **Probability scoring** for clinical decision support
            - **Feature importance** analysis for explainability
            
            ## 📋 **Required Input:**
            - `patient_id`: Unique patient identifier
            - `age`: Patient age in years (15-50)
            - `bmi`: Body Mass Index (15-50)
            - `systolic_bp`: Systolic blood pressure in mmHg (80-200)
            - `diastolic_bp`: Diastolic blood pressure in mmHg (50-120)
            - `glucose`: Blood glucose level in mg/dL (50-300)
            - `insulin`: Insulin level in μU/mL (0-50)
            - `triceps_skinfold`: Triceps skinfold thickness in mm (5-50)
            - `diabetes_pedigree`: Diabetes pedigree function (0-2.5)
            - `pregnancies`: Number of pregnancies (0-20)
            
            ## 📊 **Output:**
            - `prediction`: Binary outcome (0=no diabetes, 1=diabetes)
            - `probability`: Risk probability (0-1)
            - `risk_level`: Risk classification (low/moderate/high)
            - `model_info`: Model details and configuration
            
            ## 🎯 **Use Cases:**
            - **Prenatal care** planning and monitoring
            - **Early intervention** for high-risk pregnancies
            - **Clinical decision support** for healthcare providers
            - **Risk stratification** in maternal health programs
            """
            pass
    
    # Cardiovascular Health Endpoints
    @cardiovascular_ns.route('')
    class CardiovascularPrediction(Resource):
        @cardiovascular_ns.expect(cardiovascular_input_model)
        @cardiovascular_ns.marshal_with(cardiovascular_output_model, code=200)
        @cardiovascular_ns.marshal_with(error_model, code=400)
        @cardiovascular_ns.marshal_with(error_model, code=500)
        @cardiovascular_ns.doc('cardiovascular_prediction', description='Predict cardiovascular disease risk')
        def post(self):
            """
            ❤️ Cardiovascular Disease Risk Prediction
            
            This endpoint predicts the risk of cardiovascular disease using comprehensive
            health metrics, lifestyle factors, and advanced machine learning models.
            
            ## 🎯 **What This Endpoint Does:**
            - **Cardiovascular risk prediction** with high accuracy
            - **Risk level classification** (low/moderate/high)
            - **Feature importance** analysis using SHAP values
            - **Probability scoring** for clinical decision support
            
            ## 📋 **Required Input:**
            - `patient_id`: Unique patient identifier
            - `age`: Patient age in years (18-100)
            - `gender`: Gender (1=male, 2=female)
            - `height`: Height in cm (100-250)
            - `weight`: Weight in kg (30-200)
            - `systolic_bp`: Systolic blood pressure in mmHg (80-250)
            - `diastolic_bp`: Diastolic blood pressure in mmHg (50-150)
            - `cholesterol`: Total cholesterol in mg/dL (100-400)
            - `glucose`: Blood glucose level in mg/dL (50-300)
            - `smoking`: Smoking status (0=no, 1=yes)
            - `alcohol`: Alcohol consumption (0=no, 1=yes)
            - `physical_activity`: Physical activity level (0=low, 1=high)
            
            ## 📊 **Output:**
            - `prediction`: Binary outcome (0=low risk, 1=high risk)
            - `probability`: Risk probability (0-1)
            - `risk_level`: Risk classification (low/moderate/high)
            - `probabilities`: Class probabilities for explainability
            - `shap_values`: Feature importance scores
            - `model_info`: Model details and configuration
            
            ## 🎯 **Use Cases:**
            - **Preventive care** and early intervention
            - **Risk stratification** for patient monitoring
            - **Clinical decision support** for healthcare providers
            - **Population health** management and screening
            """
            pass
    
    # Glucose Prediction Endpoints
    @glucose_ns.route('')
    class GlucosePrediction(Resource):
        @glucose_ns.expect(glucose_input_model)
        @glucose_ns.marshal_with(glucose_output_model, code=200)
        @glucose_ns.marshal_with(error_model, code=400)
        @glucose_ns.marshal_with(error_model, code=500)
        @glucose_ns.doc('glucose_prediction', description='Predict glucose levels and assess diabetes risk using TFT model')
        def post(self):
            """
            🩸 Glucose Prediction & Diabetes Risk Assessment
            
            This endpoint provides comprehensive glucose prediction and risk assessment using a
            **Temporal Fusion Transformer (TFT)** model for 90-day glucose forecasting combined with
            rule-based risk classification and full explainability.
            
            ## 🎯 **What This Endpoint Does:**
            - **90-day glucose forecasting** with uncertainty quantification (p10, p50, p90)
            - **Risk assessment** across multiple time horizons (7, 14, 30, 60, 90 days)
            - **Context-aware risk adjustment** based on lifestyle factors
            - **Actionable recommendations** based on risk levels and context
            - **Full explainability** with detailed risk component breakdown
            
            ## 📋 **Required Input:**
            
            ### **Patient Data (Minimum 60 days)**
            Each daily record must include **ALL** the following engineered features:
            
            **Core Glucose Metrics:**
            - `g_mean`: Mean daily glucose level (mg/dL) - Range: 50-500
            - `g_std`: Standard deviation of glucose readings (mg/dL) - Range: 0-100
            - `pct_hypo`: % time in hypoglycemia (<70 mg/dL) - Range: 0-100
            - `pct_hyper`: % time in hyperglycemia (>180 mg/dL) - Range: 0-100
            
            **Insulin Management:**
            - `insulin_dose`: Daily insulin dose (units) - Range: 0-200
            - `insulin_adherence`: Adherence rate (0-1, 1=perfect) - Range: 0-1
            - `missed_insulin`: Dose missed (0/1) - Values: 0 or 1
            
            **Lifestyle Factors:**
            - `sleep_quality`: Sleep quality score (0-1, 1=excellent) - Range: 0-1
            - `exercise_flag`: Exercise performed (0/1) - Values: 0 or 1
            - `meal_variability`: Meal timing variability (0-1, 1=high) - Range: 0-1
            - `stress_index`: Stress level (0-1, 1=high) - Range: 0-1
            - `illness_flag`: Illness present (0/1) - Values: 0 or 1
            
            **Historical Events:**
            - `hypo_past7d`: Hypoglycemic events in past 7 days - Min: 0
            - `hyper_past7d`: Hyperglycemic events in past 7 days - Min: 0
            
            **Engineered Features - Glucose Lags:**
            - `g_mean_lag1`: Previous day mean glucose (mg/dL) - Range: 50-500
            - `g_mean_7d_mean`: 7-day rolling average glucose (mg/dL) - Range: 50-500
            - `g_mean_14d_std`: 14-day rolling std glucose (mg/dL) - Range: 0-100
            - `g_mean_14d_mean`: 14-day rolling average glucose (mg/dL) - Range: 50-500
            - `g_mean_30d_mean`: 30-day rolling average glucose (mg/dL) - Range: 50-500
            
            **Engineered Features - Insulin Lags:**
            - `insulin_dose_lag1`: Previous day insulin dose (units) - Range: 0-200
            - `insulin_dose_lag2`: 2 days ago insulin dose (units) - Range: 0-200
            - `insulin_dose_lag3`: 3 days ago insulin dose (units) - Range: 0-200
            - `insulin_adherence_7d_mean`: 7-day rolling average adherence (0-1) - Range: 0-1
            
            **Temporal Features:**
            - `weekday`: Day of week (0=Monday, 6=Sunday) - Range: 0-6
            - `is_weekend`: Weekend indicator (0/1) - Values: 0 or 1
            
            ## 🔧 **Optional Input:**
            
            **Context Factors (for risk adjustment):**
            - `insulin_adherence`: Current adherence rate (0-1) - Range: 0-1
            - `sleep_quality`: Current sleep quality (0-1) - Range: 0-1
            - `insulin_dose`: Current insulin dose (units) - Range: 0-200
            
            **Risk Horizons (customizable):**
            - `risk_horizons`: Array of days for risk assessment - Default: [7, 14, 30, 60, 90]
            
            ## 📊 **Output Structure:**
            
            **Glucose Predictions:**
            - `horizons_days`: Array of prediction days (1-90)
            - `p10_quantile`: Conservative glucose estimates (10th percentile)
            - `p50_quantile`: Median glucose predictions (50th percentile)
            - `p90_quantile`: Optimistic glucose estimates (90th percentile)
            
            **Risk Assessment:**
            - `overall_risk_score`: Overall risk score (0-1)
            - `overall_risk_level`: Risk classification (low/moderate/high)
            - `horizon_risks`: Risk breakdown for each specified horizon
            - `context_factors`: Impact of context factors on risk
            - `recommendations`: Actionable recommendations
            
            **Model Information:**
            - `model_type`: TemporalFusionTransformer
            - `prediction_horizon`: 90 days
            - `features_used`: 23 features
            - `risk_assessment`: Rule-based with explainability
            
            ## ⚠️ **Important Notes:**
            - **Minimum 60 days** of consecutive data required
            - **All engineered features** must be present in each data point
            - **Date format**: YYYY-MM-DD
            - **Risk horizons** can be customized (e.g., [1, 7, 30, 90])
            - **Context factors** adjust risk scores with multipliers
            
            ## 🎯 **Use Cases:**
            - **Clinical decision support** for diabetes management
            - **Long-term glucose trend** analysis and planning
            - **Risk stratification** for patient monitoring
            - **Personalized recommendations** based on individual risk factors
            """
            pass
    
    @glucose_ns.route('/cohort')
    class GlucoseCohort(Resource):
        @glucose_ns.doc('glucose_cohort', description='Batch glucose prediction for multiple patients with cohort analysis')
        def post(self):
            """
            🩸 Batch Glucose Prediction & Cohort Analysis
            
            This endpoint provides batch glucose prediction and risk assessment for multiple patients
            in a single request, along with comprehensive cohort statistics and analysis.
            
            ## 🎯 **What This Endpoint Does:**
            - **Batch processing** of multiple patients simultaneously
            - **Individual predictions** for each patient with full detail
            - **Cohort statistics** including risk distribution and averages
            - **Efficient processing** for large patient populations
            - **Error handling** for individual patient failures
            
            ## 📋 **Input Format:**
            ```json
            {
              "patients": [
                {
                  "patient_id": 123,
                  "data": [...], // Same format as single patient endpoint
                  "contexts": {...}, // Optional context factors
                  "risk_horizons": [7, 14, 30, 60, 90] // Optional custom horizons
                },
                {
                  "patient_id": 124,
                  "data": [...],
                  "contexts": {...}
                }
              ]
            }
            ```
            
            ## 📊 **Output Structure:**
            
            **Cohort Statistics:**
            - `total_patients`: Total number of patients processed
            - `successful_predictions`: Number of successful predictions
            - `failed_predictions`: Number of failed predictions
            - `average_risk_score`: Average risk score across all patients
            - `high_risk_count`: Number of high-risk patients
            - `moderate_risk_count`: Number of moderate-risk patients
            - `low_risk_count`: Number of low-risk patients
            - `high_risk_percentage`: Percentage of high-risk patients
            
            **Individual Results:**
            - `patient_predictions`: Array of individual patient results
            - Each result includes full glucose predictions and risk assessment
            - Failed predictions include error messages
            
            ## ⚠️ **Important Notes:**
            - **Same data requirements** as single patient endpoint
            - **Individual failures** don't stop batch processing
            - **Cohort statistics** calculated only from successful predictions
            - **Memory efficient** for large patient cohorts
            - **Error details** provided for failed predictions
            
            ## 🎯 **Use Cases:**
            - **Population health management** for diabetes patients
            - **Risk stratification** across patient cohorts
            - **Clinical research** and population studies
            - **Healthcare system** monitoring and planning
            """
            pass
    
    return api
