import streamlit as st

# 1. Page Configuration MUST be the first command
st.set_page_config(
    page_title="Factory Operations",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS to widen the usable space for our future data tables
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Main Landing Page UI
st.title("🏭 Factory Operations Command Center")
st.markdown("---")
st.markdown("""
Welcome to the Predictive Maintenance Dashboard. 
This system monitors real-time telemetry from the shop floor and predicts machine failures before they cause unplanned downtime.

### Status Indicator Key
We use a standardized risk threshold system across the fleet:
- 🟢 **Normal (Green):** Machine operating within optimal parameters. Failure probability **< 30%**.
- 🟡 **Warning (Yellow):** Anomalous readings detected. Schedule inspection. Failure probability **30% - 75%**.
- 🔴 **Critical (Red):** Impending failure likely. Immediate action required. Failure probability **> 75%**.

👈 **Use the sidebar to navigate to the Fleet Overview or Diagnostics pages.**
""")

# 4. Sidebar Shell
st.sidebar.title("Navigation")
st.sidebar.success("Inference Engine: Online")
st.sidebar.info("Create a `pages/` directory to automatically populate this menu.")