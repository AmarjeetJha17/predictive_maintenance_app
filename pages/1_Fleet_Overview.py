import streamlit as st
import pandas as pd
import re
import sys
import os

# Ensure the app can find the src directory no matter where Streamlit runs it from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import predict_failure

# 1. Page Config
st.set_page_config(page_title="Fleet Overview", page_icon="📊", layout="wide")
st.title("📊 Fleet Status Overview")
st.markdown("Real-time monitoring of all active shop-floor machinery. Sorted by highest failure risk.")

# 2. Simulate Live Factory Data
@st.cache_data
def get_simulated_fleet_data():
    """
    Loads a batch of 100 machines to simulate live factory data.
    We grab a mix of failures and normal operations to see our R/Y/G logic working.
    """
    df = pd.read_csv("data/raw/ai4i2020.csv")
    
    # Clean columns to match our training setup
    df.columns = [re.sub(r'[\[\]<]', '', col).strip() for col in df.columns]
    
    # Force 10 failures and 90 normal machines into our "live" view
    failures = df[df['Machine failure'] == 1].head(10)
    normals = df[df['Machine failure'] == 0].head(90)
    
    # Combine and shuffle
    fleet_df = pd.concat([failures, normals]).sample(frac=1, random_state=42).reset_index(drop=True)
    return fleet_df

fleet_df = get_simulated_fleet_data()

# 3. Prepare Data for Inference
# We must format the data exactly as the inference engine expects
X_input = fleet_df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'UDI', 'Product ID'])

quality_mapping = {'L': 0, 'M': 1, 'H': 2}
X_input['Type'] = X_input['Type'].map(quality_mapping)

# 4. Run the Machine Learning Model
probabilities, _ = predict_failure(X_input)

# 5. Apply the Red/Yellow/Green Logic
display_df = fleet_df[['UDI', 'Type', 'Air temperature K', 'Process temperature K', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min']].copy()
display_df['Failure Risk %'] = (probabilities * 100).round(1)

def assign_status(prob):
    if prob > 75:
        return "🔴 Critical"
    elif prob > 30:
        return "🟡 Warning"
    else:
        return "🟢 Normal"

display_df['Status'] = display_df['Failure Risk %'].apply(assign_status)

# 6. Render KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Machines Active", len(display_df))
col2.metric("🟢 Normal", len(display_df[display_df['Status'] == "🟢 Normal"]))
col3.metric("🟡 Warnings", len(display_df[display_df['Status'] == "🟡 Warning"]))
col4.metric("🔴 Critical Alerts", len(display_df[display_df['Status'] == "🔴 Critical"]))

st.markdown("---")

# 7. Render Color-Coded Data Table
# Sort so the highest-risk machines are immediately visible at the top
display_df = display_df.sort_values(by='Failure Risk %', ascending=False)

# Pandas styling for factory manager readability
def highlight_status(val):
    if val == "🔴 Critical":
        return 'background-color: #ffcccc; color: #990000; font-weight: bold'
    elif val == "🟡 Warning":
        return 'background-color: #fff4cc; color: #997a00; font-weight: bold'
    return ''

# Use pandas styling to apply the colors to the Status column
styled_df = display_df.style.map(highlight_status, subset=['Status'])

st.dataframe(styled_df, use_container_width=True, hide_index=True)