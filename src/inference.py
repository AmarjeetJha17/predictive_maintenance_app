import pandas as pd
import joblib
import shap
import streamlit as st

# We use Streamlit's caching decorator to load these large .pkl files only ONCE.
# Because you are running a robust Intel Core Ultra 7 processor, this caching 
# will make the slider interactions and SHAP rendering practically instantaneous.
@st.cache_resource
def load_artifacts():
    """Loads the trained model, scaler, and SHAP explainer into memory."""
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    return model, scaler, explainer

def predict_failure(input_df: pd.DataFrame):
    """
    Takes raw sensor data from the UI, scales it, and returns probabilities + SHAP values.
    """
    model, scaler, explainer = load_artifacts()
    
    # Must perfectly match the cleaned columns from data_processing.py
    continuous_features = [
        'Air temperature K', 'Process temperature K', 
        'Rotational speed rpm', 'Torque Nm', 'Tool wear min'
    ]
    
    # 1. Scale the incoming data
    # We scale the continuous features to have a mean of 0 and a standard deviation of 1.
    # This helps the model converge faster and perform better.   
    df_processed = input_df.copy()
    df_processed[continuous_features] = scaler.transform(df_processed[continuous_features])
    
    # 2. Get the probability of Class 1 (Machine Failure)
    probabilities = model.predict_proba(df_processed)[:, 1]
    
    # 3. Generate SHAP values for the explainability waterfall charts
    shap_values = explainer(df_processed)
    
    return probabilities, shap_values

if __name__ == "__main__":
    print("Inference module ready. Import 'predict_failure' in your Streamlit app.")