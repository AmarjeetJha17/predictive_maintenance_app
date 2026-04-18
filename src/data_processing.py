import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import re

def load_and_preprocess_data(filepath: str = "data/raw/ai4i2020.csv"):
    """
    Loads the AI4I dataset, cleans it, prevents data leakage, 
    and prepares scaled features and target arrays.
    """
    print(f"Loading dataset from: {filepath}...")
    df = pd.read_csv(filepath)
    
    # 0. Clean column names for XGBoost
    # This removes [, ], and < from all column names so XGBoost doesn't crash
    df.columns = [re.sub(r'[\[\]<]', '', col).strip() for col in df.columns]
    
    # 1. Prevent Data Leakage & Drop Identifiers
    # UDI and Product ID are arbitrary labels, not physical properties.
    # We drop TWF, HDF, PWF, OSF, and RNV because these are exact failure modes. 
    # If we leave them in, the model will "cheat" and perfectly predict 'Machine failure' 
    # just by looking at these flags, making the model useless for actual predictive forecasting.
    cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 2. Categorical Encoding
    # The 'Type' column represents machine quality variants (Low, Medium, High).
    # We map these to an ordinal numeric scale.
    quality_mapping = {'L': 0, 'M': 1, 'H': 2}
    df['Type'] = df['Type'].map(quality_mapping)
    
    # 3. Define Continuous Features (Updated to match the newly cleaned column names)
    
    continuous_features = [
        'Air temperature K', 'Process temperature K', 
        'Rotational speed rpm', 'Torque Nm', 'Tool wear min'
    ]
    
    # Explicitly cast integer columns to float64 to prevent LossySetitemError
    df[continuous_features] = df[continuous_features].astype(float)
    
    # 4. Separate Features (X) and Target (y)
    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']
    
    # 5. Train/Test Split with Stratification
    # We use stratify=y to ensure that the percentage of failures is the same in both the training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Feature Scaling
    # We scale the continuous features to have a mean of 0 and a standard deviation of 1.
    # This helps the model converge faster and perform better.
    scaler = StandardScaler()
    X_train.loc[:, continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test.loc[:, continuous_features] = scaler.transform(X_test[continuous_features])
    
    # Ensure the models directory exists for saving artifacts
    os.makedirs("models", exist_ok=True)
    
    # Save the fitted scaler
    joblib.dump(scaler, "models/feature_scaler.pkl")
    print("Fitted StandardScaler saved to models/feature_scaler.pkl")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test execution block
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("\n--- Pipeline Success ---")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Total failures in training set: {y_train.sum()}")