import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import re
import sys
import os

# Ensure import path works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import predict_failure

st.set_page_config(page_title="Machine Diagnostics", page_icon="🔍", layout="wide")
st.title("🔍 Deep-Dive Machine Diagnostics")
st.markdown("Select a specific machine to view real-time telemetry and AI failure justifications.")

# 1. Load Data for Selection
@st.cache_data
def load_diagnostic_data():
    df = pd.read_csv("data/raw/ai4i2020.csv")
    df.columns = [re.sub(r'[\[\]<]', '', col).strip() for col in df.columns]
    return df

df = load_diagnostic_data()

# 2. Machine Selection UI
st.sidebar.header("Diagnostic Controls")
# Let's filter to just machines that actually failed to see the most interesting SHAP plots
failed_udis = df[df['Machine failure'] == 1]['UDI'].tolist()
selected_udi = st.sidebar.selectbox("Select Machine UDI (Pre-filtered to known failures for testing):", failed_udis)

# 3. Extract Target Machine Data
machine_data = df[df['UDI'] == selected_udi].iloc[[0]]
machine_type = machine_data['Type'].values[0]

# Display basic header info
st.subheader(f"Status Report: Machine UDI {selected_udi} | Quality Type: {machine_type}")

# 4. Prepare inference payload
X_input = machine_data.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'UDI', 'Product ID'])
quality_mapping = {'L': 0, 'M': 1, 'H': 2}
X_input['Type'] = X_input['Type'].map(quality_mapping)

# Run Inference
probabilities, shap_values = predict_failure(X_input)
risk_score = (probabilities[0] * 100).round(1)

# Determine Color
if risk_score > 75:
    color = "red"
elif risk_score > 30:
    color = "orange"
else:
    color = "green"

st.markdown(f"### Current Failure Risk: <span style='color:{color}'>{risk_score}%</span>", unsafe_allow_html=True)
st.markdown("---")

# 5. Plotly Gauge Charts for Telemetry
st.markdown("#### Real-Time Sensor Telemetry")
cols = st.columns(5)

sensors = [
    ('Air temperature K', 290, 310, cols[0]),
    ('Process temperature K', 300, 320, cols[1]),
    ('Rotational speed rpm', 1000, 3000, cols[2]),
    ('Torque Nm', 0, 80, cols[3]),
    ('Tool wear min', 0, 250, cols[4])
]

for name, min_val, max_val, col in sensors:
    val = machine_data[name].values[0]
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        title = {'text': name, 'font': {'size': 14}},
        gauge = {'axis': {'range': [min_val, max_val]},
                 'bar': {'color': "darkblue"}}
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    col.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 6. Explainable AI (SHAP Waterfall Plot)
st.markdown("#### Root Cause Analysis (XAI)")
st.markdown("The chart below explains *why* the AI assigned this risk score. Red bars push the risk higher; blue bars push it lower.")

# Streamlit uses matplotlib under the hood for SHAP plots
fig, ax = plt.subplots(figsize=(10, 5))
# shap_values[0] because we only passed one row (one machine) into the model
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)
plt.clf() # Clear the figure from memory