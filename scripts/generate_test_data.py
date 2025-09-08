#!/usr/bin/env python3
"""
Test Dataset Generator for Glucose Prediction Pipeline
=====================================================

This script generates synthetic test datasets that match the exact format
required by the glucose prediction pipeline.

Usage:
    python generate_test_data.py --output test_data.csv --patients 5 --days 90

Author: Medical Prediction System
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def generate_realistic_glucose_pattern(patient_id, n_days, seed=None):
    """Generate realistic glucose patterns for a patient"""
    if seed is not None:
        np.random.seed(seed + patient_id)
    
    # Patient-specific baseline parameters
    g_base = np.random.normal(120, 15)  # Base glucose level
    g_volatility = np.random.uniform(0.7, 1.3)  # Patient-specific volatility
    circadian_strength = np.random.uniform(0.3, 0.8)  # How much glucose varies by time of day
    
    # Generate base pattern with circadian rhythm
    days = np.arange(n_days)
    circadian = circadian_strength * 20 * np.sin(2 * np.pi * days / 7)  # Weekly pattern
    trend = np.cumsum(np.random.normal(0, 0.5, n_days))  # Slow trend
    
    # Add some realistic variability
    daily_noise = np.random.normal(0, 8, n_days)
    
    # Combine all components
    g_mean = g_base + circadian + trend + daily_noise
    
    # Ensure realistic bounds
    g_mean = np.clip(g_mean, 60, 300)
    
    return g_mean

def generate_lifestyle_factors(patient_id, n_days, seed=None):
    """Generate realistic lifestyle factors"""
    if seed is not None:
        np.random.seed(seed + patient_id + 1000)
    
    # Patient-specific lifestyle patterns
    sleep_quality_base = np.random.uniform(0.6, 0.9)
    exercise_prob = np.random.uniform(0.2, 0.4)
    stress_base = np.random.uniform(0.3, 0.7)
    
    factors = {}
    
    # Sleep quality with some weekly pattern
    sleep_pattern = 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    factors['sleep_quality'] = np.clip(sleep_quality_base + sleep_pattern + 
                                      np.random.normal(0, 0.1, n_days), 0.3, 1.0)
    
    # Exercise (binary with some clustering)
    exercise = np.random.binomial(1, exercise_prob, n_days)
    # Add some clustering (if exercised yesterday, more likely today)
    for i in range(1, n_days):
        if exercise[i-1] == 1:
            exercise[i] = np.random.binomial(1, min(0.7, exercise_prob * 1.5))
    factors['exercise_flag'] = exercise
    
    # Meal variability (higher on weekends)
    weekend_mask = np.arange(n_days) % 7 >= 5
    meal_var = np.where(weekend_mask, 
                       np.random.uniform(0.4, 0.9, n_days),
                       np.random.uniform(0.2, 0.6, n_days))
    factors['meal_variability'] = meal_var
    
    # Stress index with some correlation to sleep
    stress = stress_base + np.random.normal(0, 0.2, n_days)
    stress = np.clip(stress - 0.3 * (factors['sleep_quality'] - 0.7), 0.0, 1.0)
    factors['stress_index'] = stress
    
    return factors

def generate_insulin_pattern(patient_id, n_days, glucose_pattern, seed=None):
    """Generate insulin dosing pattern based on glucose levels"""
    if seed is not None:
        np.random.seed(seed + patient_id + 2000)
    
    # Patient-specific insulin sensitivity
    ins_base = np.random.normal(40, 10)
    adherence_base = np.random.uniform(0.7, 0.95)
    
    insulin_doses = []
    adherence_history = []
    current_adherence = adherence_base
    
    for i, glucose in enumerate(glucose_pattern):
        # Miss insulin occasionally
        missed = np.random.binomial(1, 0.05)
        
        if missed:
            dose = 0
            current_adherence = max(0.5, current_adherence - 0.1)
        else:
            # Base dose with some glucose-responsive adjustment
            glucose_factor = max(0.5, min(1.5, (glucose - 100) / 50))
            dose = ins_base * current_adherence * glucose_factor
            current_adherence = min(1.0, current_adherence + 0.02)
        
        insulin_doses.append(dose)
        adherence_history.append(current_adherence)
    
    return np.array(insulin_doses), np.array(adherence_history)

def calculate_glucose_metrics(glucose_mean, glucose_std):
    """Calculate derived glucose metrics"""
    # Hypoglycemia percentage (simplified calculation)
    hypo_prob = np.maximum(0, (70 - glucose_mean) / 30)
    pct_hypo = np.clip(hypo_prob * 100, 0, 50)
    
    # Hyperglycemia percentage
    hyper_prob = np.maximum(0, (glucose_mean - 180) / 50)
    pct_hyper = np.clip(hyper_prob * 100, 0, 80)
    
    return pct_hypo, pct_hyper

def generate_patient_data(patient_id, n_days, seed=None):
    """Generate complete data for one patient with ALL engineered features from notebook"""
    if seed is not None:
        np.random.seed(seed + patient_id * 10000)
    
    # Generate base glucose pattern
    glucose_mean = generate_realistic_glucose_pattern(patient_id, n_days, seed)
    glucose_std = np.random.uniform(15, 35, n_days)
    
    # Calculate derived metrics
    pct_hypo, pct_hyper = calculate_glucose_metrics(glucose_mean, glucose_std)
    
    # Generate lifestyle factors
    lifestyle = generate_lifestyle_factors(patient_id, n_days, seed)
    
    # Generate insulin pattern
    insulin_doses, adherence = generate_insulin_pattern(patient_id, n_days, glucose_mean, seed)
    
    # Generate other factors
    illness_flag = np.random.binomial(1, 0.05, n_days)
    missed_insulin = np.random.binomial(1, 0.05, n_days)
    
    # Create DataFrame first
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    data = {
        'patient_id': [patient_id] * n_days,
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'g_mean': np.round(glucose_mean, 1),
        'g_std': np.round(glucose_std, 1),
        'pct_hypo': np.round(pct_hypo, 1),
        'pct_hyper': np.round(pct_hyper, 1),
        'insulin_dose': np.round(insulin_doses, 1),
        'insulin_adherence': np.round(adherence, 2),
        'sleep_quality': np.round(lifestyle['sleep_quality'], 2),
        'exercise_flag': lifestyle['exercise_flag'],
        'meal_variability': np.round(lifestyle['meal_variability'], 2),
        'stress_index': np.round(lifestyle['stress_index'], 2),
        'illness_flag': illness_flag,
        'missed_insulin': missed_insulin,
        'hypo_past7d': [0] * n_days,  # Will be calculated below
        'hyper_past7d': [0] * n_days  # Will be calculated below
    }
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # ENGINEER ALL FEATURES EXACTLY LIKE NOTEBOOK
    # Glucose lag features
    df['g_mean_lag1'] = df.groupby('patient_id')['g_mean'].shift(1)
    df['g_mean_7d_mean'] = df.groupby('patient_id')['g_mean'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['g_mean_14d_std'] = df.groupby('patient_id')['g_mean'].transform(lambda x: x.rolling(window=14, min_periods=1).std())
    df['g_mean_14d_mean'] = df.groupby('patient_id')['g_mean'].transform(lambda x: x.rolling(window=14, min_periods=1).mean())
    df['g_mean_30d_mean'] = df.groupby('patient_id')['g_mean'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    
    # Insulin lag features
    df['insulin_dose_lag1'] = df.groupby('patient_id')['insulin_dose'].shift(1)
    df['insulin_dose_lag2'] = df.groupby('patient_id')['insulin_dose'].shift(2)
    df['insulin_dose_lag3'] = df.groupby('patient_id')['insulin_dose'].shift(3)
    df['insulin_adherence_7d_mean'] = df.groupby('patient_id')['insulin_adherence'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # Regime/context features
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'] >= 5
    
    # Past 7-day events (rolling calculation)
    df['hypo_past7d'] = df.groupby('patient_id')['pct_hypo'].transform(lambda x: x.rolling(window=7, min_periods=1).apply(lambda y: np.sum(y > 10)))
    df['hyper_past7d'] = df.groupby('patient_id')['pct_hyper'].transform(lambda x: x.rolling(window=7, min_periods=1).apply(lambda y: np.sum(y > 20)))
    
    # Handle missing values (EXACTLY like notebook)
    # 1. Create missingness indicator columns for key lag features
    lag_cols = ["g_mean_lag1", "insulin_dose_lag1", "insulin_dose_lag2", "insulin_dose_lag3", "g_mean_14d_std"]
    for col in lag_cols:
        df[f"{col}_missing"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(df.groupby('patient_id')[col].transform('mean'))
    
    # 2. Rolling means/std can have NaN for the first few days → backfill or use patient-level mean
    rolling_cols = ["g_mean_7d_mean", "g_mean_14d_mean", "g_mean_30d_mean", "insulin_adherence_7d_mean"]
    for col in rolling_cols:
        df[col] = df[col].fillna(df.groupby('patient_id')[col].transform('mean'))
    
    # 3. Past 7-day events
    df['hypo_past7d'] = df['hypo_past7d'].fillna(0)
    df['hyper_past7d'] = df['hyper_past7d'].fillna(0)
    
    # Convert date back to string for consistency
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    return df

def generate_test_dataset(n_patients=5, n_days=90, seed=42, output_path=None):
    """Generate complete test dataset"""
    print(f"Generating test dataset with {n_patients} patients and {n_days} days each...")
    
    all_data = []
    
    for patient_id in range(n_patients):
        print(f"  Generating data for patient {patient_id}...")
        patient_df = generate_patient_data(patient_id, n_days, seed)
        all_data.append(patient_df)
    
    # Combine all patient data
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by patient_id and date
    full_df = full_df.sort_values(['patient_id', 'date']).reset_index(drop=True)
    
    # Save if output path provided
    if output_path:
        full_df.to_csv(output_path, index=False)
        print(f"✅ Dataset saved to {output_path}")
        print(f"   Total records: {len(full_df)}")
        print(f"   Date range: {full_df['date'].min()} to {full_df['date'].max()}")
        print(f"   Patients: {full_df['patient_id'].nunique()}")
    
    return full_df

def create_minimal_test_data(output_path=None):
    """Create minimal test data for quick testing"""
    print("Creating minimal test data (1 patient, 60 days)...")
    
    # Generate data for one patient with exactly 60 days
    df = generate_patient_data(patient_id=0, n_days=60, seed=42)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✅ Minimal test data saved to {output_path}")
    
    return df

def validate_dataset(df):
    """Validate that dataset meets pipeline requirements"""
    print("Validating dataset...")
    
    required_columns = [
        'patient_id', 'date', 'g_mean', 'g_std', 'pct_hypo', 'pct_hyper',
        'insulin_dose', 'insulin_adherence', 'sleep_quality', 'exercise_flag',
        'meal_variability', 'stress_index', 'illness_flag', 'missed_insulin',
        'hypo_past7d', 'hyper_past7d'
    ]
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    
    # Check data types
    numeric_cols = [col for col in required_columns if col not in ['patient_id', 'date']]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"❌ Column {col} is not numeric")
            return False
    
    # Check date format
    try:
        pd.to_datetime(df['date'])
    except:
        print("❌ Date column format is invalid")
        return False
    
    # Check minimum data per patient
    min_days = df.groupby('patient_id').size().min()
    if min_days < 60:
        print(f"❌ Some patients have less than 60 days of data (minimum: {min_days})")
        return False
    
    # Check for missing values
    missing_count = df[numeric_cols].isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️  Warning: {missing_count} missing values found")
    
    print("✅ Dataset validation passed!")
    return True

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Generate test datasets for glucose prediction pipeline')
    parser.add_argument('--output', type=str, default='test_dataset.csv', help='Output CSV file path')
    parser.add_argument('--patients', type=int, default=5, help='Number of patients to generate')
    parser.add_argument('--days', type=int, default=90, help='Number of days per patient')
    parser.add_argument('--minimal', action='store_true', help='Generate minimal test data (1 patient, 60 days)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--validate', action='store_true', help='Validate generated dataset')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.minimal:
            df = create_minimal_test_data(args.output)
        else:
            df = generate_test_dataset(
                n_patients=args.patients,
                n_days=args.days,
                seed=args.seed,
                output_path=args.output
            )
        
        if args.validate:
            validate_dataset(df)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Total records: {len(df)}")
        print(f"Patients: {df['patient_id'].nunique()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Days per patient: {len(df) // df['patient_id'].nunique()}")
        
        print("\nGlucose statistics:")
        print(f"  Mean glucose: {df['g_mean'].mean():.1f} ± {df['g_std'].mean():.1f} mg/dL")
        print(f"  Hypo events: {df['pct_hypo'].mean():.1f}% of time")
        print(f"  Hyper events: {df['pct_hyper'].mean():.1f}% of time")
        
        print("\nLifestyle factors:")
        print(f"  Sleep quality: {df['sleep_quality'].mean():.2f}")
        print(f"  Exercise rate: {df['exercise_flag'].mean():.1%}")
        print(f"  Insulin adherence: {df['insulin_adherence'].mean():.2f}")
        
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
