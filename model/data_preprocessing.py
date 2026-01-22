"""
Data Preprocessing Pipeline for Breast Cancer Dataset
=====================================================
Demonstrates actual preprocessing implementation with pandas

Author: Gbenga-Idowu AraOluwa
Matric: 22CD032145
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_dataset():
    """Load and convert dataset to pandas DataFrame"""
    print("Loading Breast Cancer Wisconsin Dataset...")
    
    data = load_breast_cancer()
    
    # Convert to DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)-1} features")
    return df

def handle_missing_values(df):
    """
    Handle missing values using pandas operations
    Demonstrates: fillna, dropna, median, mean strategies
    """
    print("\n--- Handling Missing Values ---")
    
    # Check for missing values
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()
    
    print(f"Total missing values: {total_missing}")
    
    if total_missing > 0:
        print("\nMissing values per column:")
        print(missing_summary[missing_summary > 0])
        
        # Strategy 1: Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"  Filled '{col}' with median: {median_value:.4f}")
        
        # Strategy 2: Drop rows with remaining missing values
        df.dropna(inplace=True)
        print(f"\n✓ Missing values handled. Remaining samples: {len(df)}")
    else:
        print("✓ No missing values found in dataset")
    
    return df

def encode_target_variable(df):
    """
    Encode target variable (diagnosis)
    Demonstrates: LabelEncoder usage
    """
    print("\n--- Encoding Target Variable ---")
    
    print(f"Original diagnosis distribution:")
    print(df['diagnosis'].value_counts())
    
    # In this dataset, diagnosis is already encoded (0 and 1)
    # But let's demonstrate the encoding process
    print(f"\nEncoding scheme:")
    print(f"  0 = Malignant (cancerous)")
    print(f"  1 = Benign (non-cancerous)")
    
    return df

def select_features(df):
    """
    Select specific features for modeling
    Demonstrates: DataFrame column selection, loc, iloc
    """
    print("\n--- Feature Selection ---")
    
    # List of required features
    selected_features = [
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
        'mean smoothness'
    ]
    
    print(f"Selecting {len(selected_features)} features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")
    
    # Select features using pandas
    X = df[selected_features].copy()
    y = df['diagnosis'].copy()
    
    print(f"\n✓ Feature matrix shape: {X.shape}")
    print(f"✓ Target vector shape: {y.shape}")
    
    # Display basic statistics
    print("\nFeature statistics:")
    print(X.describe())
    
    return X, y, selected_features

def apply_feature_scaling(X_train, X_test, feature_names):
    """
    Apply StandardScaler to features
    Demonstrates: StandardScaler with pandas DataFrames
    """
    print("\n--- Feature Scaling ---")
    
    print("Before scaling:")
    print(f"  Mean range: [{X_train.mean().min():.2f}, {X_train.mean().max():.2f}]")
    print(f"  Std range:  [{X_train.std().min():.2f}, {X_train.std().max():.2f}]")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames for better visualization
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, 
        columns=feature_names,
        index=X_train.index
    )
    
    print("\nAfter scaling:")
    print(f"  Mean range: [{X_train_scaled_df.mean().min():.2f}, {X_train_scaled_df.mean().max():.2f}]")
    print(f"  Std range:  [{X_train_scaled_df.std().min():.2f}, {X_train_scaled_df.std().max():.2f}]")
    
    print("\n✓ Features scaled successfully")
    
    return X_train_scaled, X_test_scaled, scaler

def analyze_feature_correlation(X, feature_names):
    """
    Analyze feature correlations using pandas
    Demonstrates: corr(), heatmap visualization (conceptually)
    """
    print("\n--- Feature Correlation Analysis ---")
    
    correlation_matrix = X.corr()
    
    print("\nHighly correlated feature pairs (|r| > 0.8):")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:
                print(f"  {feature_names[i]} <-> {feature_names[j]}: {corr_value:.3f}")
    
    return correlation_matrix

def demonstrate_pandas_operations(df):
    """
    Demonstrate various pandas operations used in preprocessing
    """
    print("\n--- Pandas Operations Demonstration ---")
    
    # 1. Filtering rows
    print("\n1. Filtering: Malignant tumors with large radius")
    malignant_large = df[(df['diagnosis'] == 0) & (df['mean radius'] > 20)]
    print(f"   Found {len(malignant_large)} samples")
    
    # 2. Groupby operations
    print("\n2. GroupBy: Average features by diagnosis")
    grouped = df.groupby('diagnosis')['mean radius', 'mean area'].mean()
    print(grouped)
    
    # 3. Apply function
    print("\n3. Apply: Categorize tumor size")
    df['size_category'] = df['mean radius'].apply(
        lambda x: 'Small' if x < 12 else ('Medium' if x < 17 else 'Large')
    )
    print(df['size_category'].value_counts())
    
    # 4. Value counts
    print("\n4. Value Counts: Diagnosis distribution")
    print(df['diagnosis'].value_counts(normalize=True))
    
    return df

def main():
    """Main preprocessing pipeline"""
    print("="*70)
    print("BREAST CANCER DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    # Load dataset
    df = load_dataset()
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode target
    df = encode_target_variable(df)
    
    # Demonstrate pandas operations
    df = demonstrate_pandas_operations(df)
    
    # Select features
    X, y, features = select_features(df)
    
    # Analyze correlations
    corr_matrix = analyze_feature_correlation(X, features)
    
    # Split data (for scaling demonstration)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply scaling
    X_train_scaled, X_test_scaled, scaler = apply_feature_scaling(
        X_train, X_test, features
    )
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nFinal dataset shapes:")
    print(f"  Training features: {X_train_scaled.shape}")
    print(f"  Testing features:  {X_test_scaled.shape}")
    print(f"  Training target:   {y_train.shape}")
    print(f"  Testing target:    {y_test.shape}")
    print("="*70)

if __name__ == "__main__":
    main()


