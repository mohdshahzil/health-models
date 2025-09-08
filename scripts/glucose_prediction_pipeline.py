#!/usr/bin/env python3
"""
Glucose Prediction Pipeline with TFT Model and Rule-Based Risk Assessment
=======================================================================

This pipeline loads a trained TFT model (epoch 50) and performs:
1. Data preprocessing for 60-day input
2. TFT forecasting (90-day horizon)
3. Rule-based risk classification
4. Comprehensive explainability analysis

Usage:
    python glucose_prediction_pipeline.py --data_path data.csv --patient_id 123 --output_dir results/

Author: Medical Prediction System
"""

import os
import sys
import argparse
import warnings
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants from notebook
HYPO_THRESHOLD = 80.0
HYPER_THRESHOLD = 160.0
BORDERLINE_LOW = 110.0
BORDERLINE_HIGH = 140.0
MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 90
QUANTILES = [0.1, 0.5, 0.9]

class TemporalFusionTransformer(nn.Module):
    """TFT model architecture from notebook"""
    
    def __init__(self,
                 encoder_length=60,
                 prediction_length=90,
                 n_encoder_known=2,
                 n_encoder_unknown=20,
                 n_decoder_known=2,
                 n_static=100,
                 hidden_size=64,
                 n_heads=4,
                 n_layers=2,
                 dropout=0.1,
                 n_quantiles=3):
        super().__init__()
        
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.hidden_size = hidden_size
        self.n_quantiles = n_quantiles
        
        # Input projections
        self.encoder_known_proj = nn.Linear(n_encoder_known, hidden_size)
        self.encoder_unknown_proj = nn.Linear(n_encoder_unknown, hidden_size)
        self.decoder_known_proj = nn.Linear(n_decoder_known, hidden_size)
        
        # Static embedding
        self.static_embedding = nn.Embedding(n_static, hidden_size)
        
        # Variable selection
        self.var_select_encoder = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1)
        )
        
        # LSTMs
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, n_layers,
                                   batch_first=True, dropout=dropout)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, n_layers,
                                   batch_first=True, dropout=dropout)
        
        # Attention
        self.attention = nn.MultiheadAttention(hidden_size, n_heads,
                                               dropout=dropout, batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, n_quantiles)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoder_known, encoder_unknown, decoder_known, static):
        batch_size = encoder_known.size(0)
        
        # Project inputs
        enc_known_proj = self.encoder_known_proj(encoder_known)
        enc_unknown_proj = self.encoder_unknown_proj(encoder_unknown)
        dec_known_proj = self.decoder_known_proj(decoder_known)
        
        # Static embedding
        static_proj = self.static_embedding(static.view(-1)).unsqueeze(1)
        static_exp_enc = static_proj.expand(-1, self.encoder_length, -1)
        
        # Variable selection
        enc_combined = torch.cat([enc_known_proj, enc_unknown_proj, static_exp_enc], dim=-1)
        var_weights = self.var_select_encoder(enc_combined)
        
        w_known = var_weights[..., 0:1]
        w_unknown = var_weights[..., 1:2]
        enc_selected = w_known * enc_known_proj + w_unknown * enc_unknown_proj
        
        # Encoder LSTM
        enc_output, (enc_hidden, enc_cell) = self.encoder_lstm(enc_selected)
        
        # Decoder LSTM
        dec_output, _ = self.decoder_lstm(dec_known_proj, (enc_hidden, enc_cell))
        
        # Attention
        attn_output, _ = self.attention(dec_output, enc_output, enc_output)
        
        # Combine with static
        static_exp_dec = static_proj.expand(-1, self.prediction_length, -1)
        combined = attn_output + static_exp_dec
        combined = self.dropout(combined)
        
        # Output
        output = self.output_layer(combined)
        
        return output

class RuleBasedGlucoseRiskClassifier:
    """Rule-based classifier with explainability from notebook"""
    
    def __init__(self, 
                 hypo_weight=0.4, 
                 hyper_weight=0.4, 
                 volatility_weight=0.2,
                 context_weight=0.3):
        self.hypo_weight = hypo_weight
        self.hyper_weight = hyper_weight  
        self.volatility_weight = volatility_weight
        self.context_weight = context_weight
        
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def _smoothstep(self, x, a, b):
        t = np.clip((x - a) / (b - a + 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    def calculate_base_risk(self, predictions):
        """Calculate base glucose risk with explainability tracking"""
        N, H, Q = predictions.shape
        p10, p50, p90 = predictions[:, :, 0], predictions[:, :, 1], predictions[:, :, 2]
        
        # Track contributions for explainability
        explanations = {
            'hypo_risk': np.zeros((N, H)),
            'hyper_risk': np.zeros((N, H)),
            'volatility_risk': np.zeros((N, H)),
            'trend_low_risk': np.zeros((N, H)),
            'trend_high_risk': np.zeros((N, H))
        }
        
        # Hypoglycemia risk (continuous)
        d_low_med = (HYPO_THRESHOLD - p50) / 20.0
        d_low_tail = (HYPO_THRESHOLD - p10) / 15.0
        hypo = 0.6 * self._sigmoid(d_low_med) + 0.4 * self._sigmoid(d_low_tail)
        explanations['hypo_risk'] = hypo
        
        # Hyperglycemia risk (continuous)
        d_high_med = (p50 - HYPER_THRESHOLD) / 20.0
        d_high_tail = (p90 - HYPER_THRESHOLD) / 15.0
        hyper = 0.6 * self._sigmoid(d_high_med) + 0.4 * self._sigmoid(d_high_tail)
        explanations['hyper_risk'] = hyper
        
        # Volatility risk
        spread = p90 - p10
        vol = np.clip((spread - 20.0) / 30.0, 0.0, 1.0)
        explanations['volatility_risk'] = vol
        
        # Trending risks
        trend_low = 0.2 * self._smoothstep(BORDERLINE_LOW - p50, 0.0, 15.0) * np.clip((spread - 25.0) / 15.0, 0.0, 1.0)
        trend_high = 0.2 * self._smoothstep(p50 - BORDERLINE_HIGH, 0.0, 15.0) * np.clip((spread - 25.0) / 15.0, 0.0, 1.0)
        explanations['trend_low_risk'] = trend_low
        explanations['trend_high_risk'] = trend_high
        
        # Combine risks
        base = (
            self.hypo_weight * hypo +
            self.hyper_weight * hyper +
            self.volatility_weight * vol +
            trend_low + trend_high
        )
        
        # Monotone calibration
        base = 1.0 - np.exp(-2.0 * np.clip(base, 0.0, 1.0))
        
        return np.clip(base, 0.0, 1.0), explanations
    
    def calculate_context_risk(self, insulin_adherence=None, sleep_quality=None, insulin_dose=None):
        """Calculate contextual risk adjustments with explanations"""
        multiplier = 1.0
        context_explanations = {}
        
        if insulin_adherence is not None:
            if insulin_adherence < 0.7:
                multiplier *= 1.5
                context_explanations['insulin_adherence'] = {'value': insulin_adherence, 'impact': 'high_risk', 'multiplier': 1.5}
            elif insulin_adherence < 0.85:
                multiplier *= 1.2
                context_explanations['insulin_adherence'] = {'value': insulin_adherence, 'impact': 'moderate_risk', 'multiplier': 1.2}
            else:
                context_explanations['insulin_adherence'] = {'value': insulin_adherence, 'impact': 'low_risk', 'multiplier': 1.0}
                
        if sleep_quality is not None:
            if sleep_quality < 0.5:
                multiplier *= 1.3
                context_explanations['sleep_quality'] = {'value': sleep_quality, 'impact': 'high_risk', 'multiplier': 1.3}
            elif sleep_quality < 0.7:
                multiplier *= 1.15
                context_explanations['sleep_quality'] = {'value': sleep_quality, 'impact': 'moderate_risk', 'multiplier': 1.15}
            else:
                context_explanations['sleep_quality'] = {'value': sleep_quality, 'impact': 'low_risk', 'multiplier': 1.0}
                
        if insulin_dose is not None:
            if insulin_dose < 5:
                multiplier *= 1.4
                context_explanations['insulin_dose'] = {'value': insulin_dose, 'impact': 'high_risk', 'multiplier': 1.4}
            else:
                context_explanations['insulin_dose'] = {'value': insulin_dose, 'impact': 'low_risk', 'multiplier': 1.0}
                
        return multiplier, context_explanations
    
    def predict_risk_with_explanations(self, predictions, contexts=None):
        """Main prediction method with full explainability"""
        # Base glucose risk
        base_risk, base_explanations = self.calculate_base_risk(predictions)
        
        # Context adjustments
        context_mult = 1.0
        context_explanations = {}
        if contexts is not None:
            context_mult, context_explanations = self.calculate_context_risk(
                contexts.get('insulin_adherence'),
                contexts.get('sleep_quality'), 
                contexts.get('insulin_dose')
            )
            base_risk = base_risk * context_mult
            
        # Summary risk (weighted average across horizons)
        H = predictions.shape[1]
        horizon_weights = np.exp(-np.arange(H) * 0.05)
        summary_risk = np.average(base_risk, axis=1, weights=horizon_weights)
        
        # Compile full explanations
        full_explanations = {
            'base_risk_components': base_explanations,
            'context_factors': context_explanations,
            'context_multiplier': context_mult,
            'horizon_weights': horizon_weights.tolist(),
            'final_risk_scores': base_risk.tolist(),
            'summary_risk': summary_risk.tolist()
        }
        
        return base_risk, summary_risk, full_explanations

class GlucosePredictionPipeline:
    """Main pipeline class"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self.scalers = None
        self.rule_classifier = RuleBasedGlucoseRiskClassifier()
        
        # Load model and scalers (scalers are saved with the model checkpoint)
        self._load_model_and_scalers(model_path)
        
    def _load_model_and_scalers(self, model_path: str):
        """Load trained TFT model and scalers from checkpoint (matches notebook approach)"""
        try:
            # Load checkpoint (contains model state, scalers, and training history)
            # Handle PyTorch 2.6+ weights_only security change
            try:
                # Try with weights_only=True first (secure)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception:
                # Fallback: allow sklearn scalers (trusted source)
                import torch.serialization
                torch.serialization.add_safe_globals([
                    'sklearn.preprocessing._data.StandardScaler',
                    'sklearn.preprocessing.StandardScaler'
                ])
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Initialize model with EXACT architecture from notebook
            self.model = TemporalFusionTransformer(
                encoder_length=MAX_ENCODER_LENGTH,
                prediction_length=MAX_PREDICTION_LENGTH,
                n_encoder_known=2,  # ['weekday', 'is_weekend']
                n_encoder_unknown=21,  # All 21 unknown features from notebook
                n_decoder_known=2,  # ['weekday', 'is_weekend'] for future
                n_static=1000,  # Large enough for patient IDs
                hidden_size=64,
                n_heads=4,
                n_layers=2,
                dropout=0.1,
                n_quantiles=len(QUANTILES)
            )
            
            # Load model state dict from checkpoint
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Fallback: assume checkpoint is just the state dict
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            
            # Load scalers from checkpoint (as done in notebook)
            if 'scalers' in checkpoint:
                self.scalers = checkpoint['scalers']
                print(f"✅ TFT model and scalers loaded from {model_path}")
            else:
                raise ValueError("Checkpoint does not contain scalers. Please use a checkpoint saved with the training script.")
            
        except Exception as e:
            print(f"❌ Error loading model and scalers: {e}")
            raise
    
    def prepare_tft_data(self, df: pd.DataFrame, patient_id: int) -> Dict[str, torch.Tensor]:
        """Prepare data for TFT inference (EXACTLY like notebook)"""
        # Filter patient data
        patient_data = df[df['patient_id'] == patient_id].copy()
        
        if len(patient_data) < MAX_ENCODER_LENGTH:
            raise ValueError(f"Insufficient data for patient {patient_id}. Need at least {MAX_ENCODER_LENGTH} days.")
        
        # Sort by date
        patient_data = patient_data.sort_values('date').reset_index(drop=True)
        
        # EXACT feature definitions from notebook (21 unknown features)
        time_varying_known_features = ['weekday', 'is_weekend']
        time_varying_unknown_features = [
            'g_mean', 'g_std', 'insulin_dose', 'insulin_adherence',
            'sleep_quality', 'exercise_flag', 'meal_variability',
            'stress_index', 'illness_flag', 'missed_insulin',
            'g_mean_lag1', 'g_mean_7d_mean', 'g_mean_14d_std',
            'insulin_dose_lag1', 'insulin_dose_lag2', 'insulin_dose_lag3',
            'insulin_adherence_7d_mean', 'g_mean_14d_mean', 'g_mean_30d_mean',
            'hypo_past7d', 'hyper_past7d'
        ]
        
        # Check if all required features exist
        missing_features = []
        for feature in time_varying_known_features + time_varying_unknown_features:
            if feature not in patient_data.columns:
                missing_features.append(feature)
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}. "
                           f"Please ensure your data has all engineered features from the notebook.")
        
        # Get the last MAX_ENCODER_LENGTH days for encoding
        encoder_data = patient_data.tail(MAX_ENCODER_LENGTH)
        
        # Prepare encoder inputs (EXACTLY like notebook)
        encoder_known = encoder_data[time_varying_known_features].astype(float).values
        encoder_unknown = encoder_data[time_varying_unknown_features].astype(float).values
        
        # For decoder, create future known features (weekday/is_weekend for future dates)
        # This is a simplified approach - in practice you'd have actual future calendar info
        last_date = pd.to_datetime(encoder_data['date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i+1) for i in range(MAX_PREDICTION_LENGTH)]
        future_weekdays = [d.weekday() for d in future_dates]
        future_is_weekend = [d.weekday() >= 5 for d in future_dates]
        
        decoder_known = np.column_stack([future_weekdays, future_is_weekend]).astype(float)
        
        # Static features (patient ID) - convert to categorical index like notebook
        # The notebook converts patient_id to categorical codes
        static_vals = np.array([patient_id])
        
        # Normalize data using scalers (EXACTLY like notebook)
        encoder_unknown_norm = self.scalers['encoder_unknown'].transform(encoder_unknown)
        
        # Convert to tensors (EXACTLY like notebook)
        batch = {
            'encoder_known': torch.tensor(encoder_known, dtype=torch.float32).unsqueeze(0).to(self.device),
            'encoder_unknown': torch.tensor(encoder_unknown_norm, dtype=torch.float32).unsqueeze(0).to(self.device),
            'decoder_known': torch.tensor(decoder_known, dtype=torch.float32).unsqueeze(0).to(self.device),
            'static': torch.tensor(static_vals, dtype=torch.long).unsqueeze(0).to(self.device)
        }
        
        return batch
    
    def predict_glucose(self, df: pd.DataFrame, patient_id: int) -> Tuple[np.ndarray, Dict]:
        """Generate glucose predictions using TFT model (matches notebook denormalization)"""
        # Prepare data
        batch = self.prepare_tft_data(df, patient_id)
        
        # Generate predictions
        with torch.no_grad():
            predictions = self.model(
                batch['encoder_known'],
                batch['encoder_unknown'], 
                batch['decoder_known'],
                batch['static']
            )
        
        # Denormalize predictions exactly like notebook
        # Shape: (1, H, 3) -> (H, 3) after squeeze
        predictions_np = predictions.cpu().numpy().squeeze(0)  # Remove batch dimension
        
        # Reshape to (H*3, 1) for inverse_transform, then reshape back to (H, 3)
        predictions_denorm = self.scalers['target'].inverse_transform(
            predictions_np.reshape(-1, 1)
        ).reshape(predictions_np.shape)
        
        # Prepare prediction metadata
        prediction_metadata = {
            'patient_id': patient_id,
            'prediction_horizon_days': MAX_PREDICTION_LENGTH,
            'quantiles': QUANTILES,
            'model_used': 'TFT_epoch50',
            'prediction_timestamp': datetime.now().isoformat(),
            'denormalization_applied': True
        }
        
        return predictions_denorm, prediction_metadata
    
    def assess_risk_with_explanations(self, predictions: np.ndarray, contexts: Dict = None, risk_horizons: List[int] = None) -> Dict:
        """Assess risk using rule-based classifier with full explainability"""
        # Ensure predictions have correct shape (1, H, 3)
        if predictions.ndim == 2:
            predictions = predictions.reshape(1, -1, 3)
        
        # Get risk scores and explanations
        risk_scores, summary_risk, explanations = self.rule_classifier.predict_risk_with_explanations(
            predictions, contexts
        )
        
        # Default risk horizons if not specified
        if risk_horizons is None:
            # Default: key clinical horizons for 90-day prediction period
            risk_horizons = [7, 14, 30, 60, 90]
        
        # Calculate risk metrics for specified horizons
        risk_metrics = {}
        
        for h in risk_horizons:
            if h <= risk_scores.shape[1]:
                risk_h = risk_scores[0, h-1]  # First (and only) sample
                
                risk_metrics[f'horizon_{h}d'] = {
                    'risk_score': float(risk_h),
                    'risk_level': self._classify_risk_level(risk_h),
                    'hypo_risk': float(explanations['base_risk_components']['hypo_risk'][0, h-1]),
                    'hyper_risk': float(explanations['base_risk_components']['hyper_risk'][0, h-1]),
                    'volatility_risk': float(explanations['base_risk_components']['volatility_risk'][0, h-1]),
                    'trend_low_risk': float(explanations['base_risk_components']['trend_low_risk'][0, h-1]),
                    'trend_high_risk': float(explanations['base_risk_components']['trend_high_risk'][0, h-1])
                }
        
        # Overall risk assessment
        overall_risk = float(summary_risk[0])
        
        # Compile comprehensive results
        results = {
            'overall_risk_score': overall_risk,
            'overall_risk_level': self._classify_risk_level(overall_risk),
            'horizon_risks': risk_metrics,
            'context_factors': explanations['context_factors'],
            'context_multiplier': explanations['context_multiplier'],
            'detailed_explanations': explanations,
            'recommendations': self._generate_recommendations(overall_risk, explanations['context_factors'])
        }
        
        return results
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score"""
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'moderate'
        else:
            return 'high'
    
    def _generate_recommendations(self, risk_score: float, context_factors: Dict) -> List[str]:
        """Generate actionable recommendations based on risk and context"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_score > 0.6:
            recommendations.append("🚨 HIGH RISK: Consider immediate medical consultation")
            recommendations.append("📊 Monitor glucose more frequently (every 2-4 hours)")
        elif risk_score > 0.3:
            recommendations.append("⚠️ MODERATE RISK: Increase monitoring frequency")
            recommendations.append("📝 Review insulin dosing with healthcare provider")
        else:
            recommendations.append("✅ LOW RISK: Continue current management plan")
        
        # Context-based recommendations
        for factor, details in context_factors.items():
            if details['impact'] == 'high_risk':
                if factor == 'insulin_adherence':
                    recommendations.append("💉 Improve insulin adherence - consider reminders or support")
                elif factor == 'sleep_quality':
                    recommendations.append("😴 Focus on sleep hygiene - poor sleep affects glucose control")
                elif factor == 'insulin_dose':
                    recommendations.append("💊 Review insulin dosing - very low doses may indicate issues")
        
        return recommendations
    
    def run_full_pipeline(self, df: pd.DataFrame, patient_id: int, contexts: Dict = None, risk_horizons: List[int] = None) -> Dict:
        """Run complete pipeline: TFT prediction + risk assessment + explainability"""
        print(f"🔍 Running pipeline for patient {patient_id}...")
        
        # Step 1: TFT glucose prediction
        print("📈 Generating glucose predictions...")
        predictions, pred_metadata = self.predict_glucose(df, patient_id)
        
        # Step 2: Risk assessment with explainability
        print("🎯 Assessing risk with explainability...")
        risk_results = self.assess_risk_with_explanations(predictions, contexts, risk_horizons)
        
        # Step 3: Compile final results
        final_results = {
            'patient_id': patient_id,
            'prediction_metadata': pred_metadata,
            'glucose_predictions': {
                'horizons_days': list(range(1, MAX_PREDICTION_LENGTH + 1)),
                'p10_quantile': predictions[:, 0].tolist(),
                'p50_quantile': predictions[:, 1].tolist(),
                'p90_quantile': predictions[:, 2].tolist()
            },
            'risk_assessment': risk_results,
            'pipeline_timestamp': datetime.now().isoformat()
        }
        
        print("✅ Pipeline completed successfully!")
        return final_results

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Glucose Prediction Pipeline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input CSV data')
    parser.add_argument('--patient_id', type=int, required=True, help='Patient ID to analyze')
    parser.add_argument('--model_path', type=str, default='tft_epoch50.pth', help='Path to TFT model checkpoint (contains scalers)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--contexts', type=str, help='JSON string with context factors')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse contexts if provided
    contexts = None
    if args.contexts:
        try:
            contexts = json.loads(args.contexts)
        except json.JSONDecodeError:
            print("❌ Invalid JSON format for contexts")
            return
    
    try:
        # Load data
        print(f"📂 Loading data from {args.data_path}...")
        df = pd.read_csv(args.data_path)
        
        # Initialize pipeline
        print("🚀 Initializing pipeline...")
        pipeline = GlucosePredictionPipeline(args.model_path, args.device)
        
        # Run pipeline
        results = pipeline.run_full_pipeline(df, args.patient_id, contexts)
        
        # Save results
        output_file = os.path.join(args.output_dir, f'patient_{args.patient_id}_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Patient ID: {args.patient_id}")
        print(f"Overall Risk: {results['risk_assessment']['overall_risk_level'].upper()}")
        print(f"Risk Score: {results['risk_assessment']['overall_risk_score']:.3f}")
        print("\nRecommendations:")
        for rec in results['risk_assessment']['recommendations']:
            print(f"  {rec}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
