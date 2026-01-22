"""
Breast Cancer Prediction Model Training Script
===============================================
This script demonstrates the complete ML workflow:
- Data loading and preprocessing
- Feature selection and encoding
- Model training with hyperparameter tuning
- Model evaluation
- Model persistence

Author: Gbenga-Idowu AraOluwa
Matric: 22CD032145
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    precision_score, 
    recall_score, 
    f1_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """
    Load the Breast Cancer Wisconsin dataset and perform initial exploration
    """
    print("="*70)
    print("STEP 1: LOADING BREAST CANCER WISCONSIN DATASET")
    print("="*70)
    
    # Load dataset
    data = load_breast_cancer()
    
    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(df)}")
    print(f"  Total features: {len(df.columns) - 1}")
    print(f"\n  Target distribution:")
    print(f"    Malignant (0): {(df['diagnosis'] == 0).sum()}")
    print(f"    Benign (1): {(df['diagnosis'] == 1).sum()}")
    print(f"    Malignancy rate: {(df['diagnosis'] == 0).mean()*100:.2f}%")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    print(f"\n  Missing values: {missing_count}")
    
    if missing_count == 0:
        print("  ✓ No missing values - dataset is complete!")
    
    return df

def preprocess_data(df):
    """
    Perform data preprocessing:
    - Feature selection
    - Handling missing values (if any)
    - Feature scaling
    """
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)
    
    # Feature selection - select 5 features as required
    selected_features = [
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
        'mean smoothness'
    ]
    
    print(f"\n  Selected features (5 out of 8 available):")
    for i, feat in enumerate(selected_features, 1):
        print(f"    {i}. {feat}")
    
    # Create subset
    X = df[selected_features]
    y = df['diagnosis']
    
    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Target vector shape: {y.shape}")
    
    # Handle missing values (if any)
    if X.isnull().sum().sum() > 0:
        print("\n  Handling missing values...")
        X = X.fillna(X.median())
        print("  ✓ Missing values filled with median")
    else:
        print("\n  ✓ No missing values to handle")
    
    return X, y, selected_features

def split_data(X, y):
    """
    Split data into training and testing sets with stratification
    """
    print("\n" + "="*70)
    print("STEP 3: TRAIN-TEST SPLIT")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\n  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Testing set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    print(f"\n  Training set class distribution:")
    print(f"    Malignant: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"    Benign:    {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Apply feature scaling (mandatory for SVM)
    """
    print("\n" + "="*70)
    print("STEP 4: FEATURE SCALING")
    print("="*70)
    
    print("\n  Applying StandardScaler...")
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n  ✓ Features scaled successfully")
    print(f"    Training set mean: ~{X_train_scaled.mean():.4f} (should be ~0)")
    print(f"    Training set std:  ~{X_train_scaled.std():.4f} (should be ~1)")
    
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train_scaled, y_train):
    """
    Train Support Vector Machine classifier
    """
    print("\n" + "="*70)
    print("STEP 5: MODEL TRAINING (SVM)")
    print("="*70)
    
    print("\n  Initializing Support Vector Machine...")
    print("  Hyperparameters:")
    print("    - Kernel: RBF (Radial Basis Function)")
    print("    - C: 1.0 (regularization)")
    print("    - Gamma: scale")
    print("    - Probability: True")
    
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True
    )
    
    print("\n  Training model...")
    model.fit(X_train_scaled, y_train)
    
    print(f"\n  ✓ Model trained successfully!")
    print(f"    Support vectors: {len(model.support_)}")
    print(f"    Support vectors per class:")
    print(f"      Malignant: {model.n_support_[0]}")
    print(f"      Benign:    {model.n_support_[1]}")
    
    # Cross-validation
    print("\n  Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return model

def evaluate_model(model, X_test_scaled, y_test):
    """
    Evaluate model performance with comprehensive metrics
    """
    print("\n" + "="*70)
    print("STEP 6: MODEL EVALUATION")
    print("="*70)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n  PERFORMANCE METRICS:")
    print("  " + "-"*50)
    print(f"    Accuracy:  {accuracy*100:.2f}%")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    
    print("\n  CLASSIFICATION REPORT:")
    print("  " + "-"*50)
    print(classification_report(
        y_test, y_pred, 
        target_names=['Malignant', 'Benign'],
        digits=4
    ))
    
    print("  CONFUSION MATRIX:")
    print("  " + "-"*50)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n                  Predicted")
    print(f"                  Mal   Ben")
    print(f"    Actual Mal  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
    print(f"    Actual Ben  [{cm[1][0]:4d}  {cm[1][1]:4d}]")
    
    # Analysis
    print("\n  PERFORMANCE ANALYSIS:")
    print("  " + "-"*50)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"    True Negatives:  {tn} (correctly identified malignant)")
    print(f"    False Positives: {fp} (benign predicted as malignant)")
    print(f"    False Negatives: {fn} (malignant predicted as benign)")
    print(f"    True Positives:  {tp} (correctly identified benign)")
    print(f"\n    Specificity: {specificity:.4f}")
    print(f"    NPV (Negative Predictive Value): {npv:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def save_model(model, scaler):
    """
    Save trained model and scaler for deployment
    """
    print("\n" + "="*70)
    print("STEP 7: MODEL PERSISTENCE")
    print("="*70)
    
    # Save model
    model_path = 'breast_cancer_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n  ✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    # Verify saved files
    print("\n  Verifying saved files...")
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    print("  ✓ Files loaded successfully - ready for deployment!")

def test_inference(model, scaler):
    """
    Test the trained model with sample data
    """
    print("\n" + "="*70)
    print("STEP 8: INFERENCE TEST")
    print("="*70)
    
    # Test case 1: Likely benign
    print("\n  Test Case 1: Small, smooth tumor (likely benign)")
    sample_1 = pd.DataFrame({
        'mean radius': [12.0],
        'mean texture': [18.0],
        'mean perimeter': [80.0],
        'mean area': [500.0],
        'mean smoothness': [0.09]
    })
    
    sample_1_scaled = scaler.transform(sample_1)
    pred_1 = model.predict(sample_1_scaled)[0]
    prob_1 = model.predict_proba(sample_1_scaled)[0]
    
    print(f"    Prediction: {'Benign' if pred_1 == 1 else 'Malignant'}")
    print(f"    Benign probability: {prob_1[1]*100:.2f}%")
    print(f"    Malignant probability: {prob_1[0]*100:.2f}%")
    
    # Test case 2: Likely malignant
    print("\n  Test Case 2: Large, irregular tumor (likely malignant)")
    sample_2 = pd.DataFrame({
        'mean radius': [20.0],
        'mean texture': [25.0],
        'mean perimeter': [130.0],
        'mean area': [1200.0],
        'mean smoothness': [0.12]
    })
    
    sample_2_scaled = scaler.transform(sample_2)
    pred_2 = model.predict(sample_2_scaled)[0]
    prob_2 = model.predict_proba(sample_2_scaled)[0]
    
    print(f"    Prediction: {'Benign' if pred_2 == 1 else 'Malignant'}")
    print(f"    Benign probability: {prob_2[1]*100:.2f}%")
    print(f"    Malignant probability: {prob_2[0]*100:.2f}%")

def main():
    """
    Main training pipeline
    """
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "BREAST CANCER PREDICTION MODEL" + " "*22 + "║")
    print("║" + " "*20 + "Training Pipeline" + " "*31 + "║")
    print("╚" + "="*68 + "╝")
    
    # Step 1: Load data
    df = load_and_explore_data()
    
    # Step 2: Preprocess
    X, y, features = preprocess_data(df)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 5: Train model
    model = train_model(X_train_scaled, y_train)
    
    # Step 6: Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Step 7: Save
    save_model(model, scaler)
    
    # Step 8: Test inference
    test_inference(model, scaler)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n  Final Model Performance:")
    print(f"    Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1-Score:  {metrics['f1']:.4f}")
    print("\n  Model files saved and ready for deployment!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
