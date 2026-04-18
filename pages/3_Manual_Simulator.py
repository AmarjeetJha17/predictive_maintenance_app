import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import sys
import os

# Ensure the app can find the src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import predict_failure

st.set_page_config(page_title="Manual Simulator", page_icon="🎛️", layout="wide")
st.title("🎛️ Manual Simulator (What-If Analysis)")
st.markdown("Adjust the operational parameters below to stress-test a virtual machine and observe how the AI evaluates the changing risk of failure in real-time.")

# 1. UI: Input Controls (Placed in Sidebar for clean layout)
st.sidebar.header("Stress-Test Parameters")

# We use Streamlit sliders bounded by the general min/max of our dataset
machine_type = st.sidebar.radio("Machine Quality Type", ["L", "M", "H"])
air_temp = st.sidebar.slider("Air temperature [K]", 290.0, 310.0, 298.0, step=0.1)
process_temp = st.sidebar.slider("Process temperature [K]", 300.0, 320.0, 308.0, step=0.1)
rpm = st.sidebar.slider("Rotational speed [rpm]", 1000, 3000, 1500, step=10)
torque = st.sidebar.slider("Torque [Nm]", 0.0, 80.0, 40.0, step=0.5)
tool_wear = st.sidebar.slider("Tool wear [min]", 0, 250, 50, step=1)

# 2. Format Data for the Inference Engine
# The column names MUST match exactly what the scaler and XGBoost expect
quality_mapping = {'L': 0, 'M': 1, 'H': 2}
input_data = pd.DataFrame([{
    'Type': quality_mapping[machine_type],
    'Air temperature K': air_temp,
    'Process temperature K': process_temp,
    'Rotational speed rpm': rpm,
    'Torque Nm': torque,
    'Tool wear min': tool_wear
}])

# 3. Run Inference dynamically
# Because of our @st.cache_resource, this executes instantly every time a slider moves
probabilities, shap_values = predict_failure(input_data)
risk_score = (probabilities[0] * 100).round(1)

# 4. Display Results with R/Y/G Logic
if risk_score > 75:
    color = "#ffcccc" # Light Red
    text_color = "#990000"
    status_text = "🔴 CRITICAL FAILURE IMMINENT"
elif risk_score > 30:
    color = "#fff4cc" # Light Yellow
    text_color = "#997a00"
    status_text = "🟡 WARNING STATE"
else:
    color = "#ccffcc" # Light Green
    text_color = "#006600"
    status_text = "🟢 NORMAL OPERATION"

# Layout: 1/3 for the big score, 2/3 for the SHAP explanation
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Virtual Machine Status")
    # Using a stylized container for the alert status
    st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; border: 2px solid {text_color}; text-align: center;">
            <h2 style="color: {text_color}; margin: 0;">{status_text}</h2>
            <h1 style="color: {text_color}; margin: 10px 0 0 0; font-size: 3rem;">{risk_score}%</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("**Recruiter / Manager Testing Guide:**\n\nTry dragging **Torque** to maximum while dropping **Rotational speed** to simulate a physical jam. Watch how the SHAP chart reacts!")

with col2:
    st.markdown("#### Real-Time Explainer (SHAP)")
    st.markdown("Watch the bars shift as you move the sliders. This explains *why* the AI is changing its mind.")
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    plt.clf()