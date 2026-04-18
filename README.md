# 🏭 Factory Operations — Predictive Maintenance Dashboard

A real-time predictive maintenance system that monitors factory machinery telemetry and forecasts failures **before** they cause unplanned downtime. Built with **XGBoost**, **SHAP Explainability**, and a multi-page **Streamlit** dashboard.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dashboard Pages](#dashboard-pages)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Retraining the Model](#retraining-the-model)
- [Dataset](#dataset)
- [License](#license)

---

## Overview

Manufacturing downtime is expensive. This project applies **supervised machine learning** to sensor telemetry data (temperature, rotational speed, torque, tool wear) to predict the probability of machine failure in real time.

Every prediction is accompanied by a **SHAP waterfall chart** that explains exactly *which* sensor readings are pushing the risk score higher or lower — making the AI transparent and actionable for floor managers.

### Risk Classification System

| Status | Threshold | Action |
|---|---|---|
| 🟢 **Normal** | Failure probability **< 30%** | Machine operating within optimal parameters |
| 🟡 **Warning** | Failure probability **30% – 75%** | Anomalous readings detected — schedule inspection |
| 🔴 **Critical** | Failure probability **> 75%** | Impending failure — immediate intervention required |

---

## Key Features

- **XGBoost Classifier** trained on the AI4I 2020 dataset with class-imbalance handling via `scale_pos_weight`
- **Explainable AI (XAI)** — SHAP TreeExplainer provides per-prediction feature attribution
- **Real-Time Inference** — cached model loading via `@st.cache_resource` for instant slider response
- **Interactive What-If Simulator** — drag sliders to stress-test virtual machines and watch risk shift live
- **Color-Coded Fleet Monitoring** — 100-machine overview sorted by failure risk with Red/Yellow/Green status
- **Plotly Gauge Charts** — per-sensor telemetry visualisation on the diagnostics page
- **Data Leakage Prevention** — failure mode flags (TWF, HDF, PWF, OSF, RNF) are dropped before training

---

## Dashboard Pages

### 1. 🏭 Command Center (`app.py`)
The landing page. Provides a system overview, the risk classification key, and navigation guidance.

### 2. 📊 Fleet Overview (`pages/1_Fleet_Overview.py`)
Simulates a live factory floor with 100 machines (10 known failures + 90 normal operations). Runs batch inference and displays a **color-coded, sortable data table** with KPI metrics for total machines, warnings, and critical alerts.

### 3. 🔍 Machine Diagnostics (`pages/2_Machine_Diagnostics.py`)
Deep-dive into a single machine. Select a machine by UDI to view:
- **Plotly gauge charts** for each sensor reading
- A **SHAP waterfall chart** explaining the AI's risk assessment

### 4. 🎛️ Manual Simulator (`pages/3_Manual_Simulator.py`)
A what-if analysis tool. Adjust operational parameters via sidebar sliders (temperature, RPM, torque, tool wear) and observe how the AI's failure prediction and SHAP explanation update in real time.

---

## Project Structure

```
predictive_maintenance_app/
│
├── app.py                          # Streamlit entry point (Command Center)
├── main.py                         # Placeholder main script
├── pyproject.toml                  # Project metadata & dependencies (uv)
├── requirements.txt                # Pinned dependencies (pip-compatible)
│
├── pages/                          # Streamlit multi-page app
│   ├── 1_Fleet_Overview.py         # Fleet-wide batch inference & status table
│   ├── 2_Machine_Diagnostics.py    # Single-machine drill-down with gauges + SHAP
│   └── 3_Manual_Simulator.py       # Interactive slider-based what-if simulator
│
├── src/                            # Core ML pipeline
│   ├── data_processing.py          # Data loading, cleaning, scaling, train/test split
│   ├── model_training.py           # XGBoost training, evaluation, artifact export
│   └── inference.py                # Production inference: scaling → prediction → SHAP
│
├── models/                         # Serialised ML artifacts (git-ignored)
│   ├── xgb_model.pkl               # Trained XGBoost classifier
│   ├── feature_scaler.pkl          # Fitted StandardScaler
│   ├── shap_explainer.pkl          # SHAP TreeExplainer
│   └── X_train_sample.pkl          # Baseline sample for SHAP plots
│
├── data/
│   └── raw/
│       └── ai4i2020.csv            # AI4I 2020 Predictive Maintenance Dataset (git-ignored)
│
└── notebooks/
    └── 01_EDA.ipynb                # Exploratory Data Analysis
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.13 |
| **ML Model** | XGBoost |
| **Explainability** | SHAP (TreeExplainer) |
| **Data Processing** | Pandas, Scikit-learn (StandardScaler, train_test_split) |
| **Dashboard** | Streamlit (multi-page app) |
| **Visualisation** | Plotly, Matplotlib, Seaborn |
| **Serialisation** | Joblib |
| **Package Manager** | uv |
| **Dev Tools** | Black (formatter), Ruff (linter) |

---

## Getting Started

### Prerequisites

- **Python 3.13+**
- [**uv**](https://docs.astral.sh/uv/) package manager (recommended), or `pip`

### 1. Clone the Repository

```bash
git clone https://github.com/AmarjeetJha17/predictive_maintenance_app.git
cd predictive_maintenance_app
```

### 2. Set Up the Environment

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Add the Dataset

Download the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) and place the CSV file at:

```
data/raw/ai4i2020.csv
```

### 4. Train the Model (First-Time Setup)

The trained model artifacts are git-ignored, so you must generate them before running the dashboard:

```bash
cd src
python model_training.py
cd ..
```

This will create the following files in the `models/` directory:
- `xgb_model.pkl` — trained classifier
- `feature_scaler.pkl` — fitted StandardScaler
- `shap_explainer.pkl` — SHAP TreeExplainer
- `X_train_sample.pkl` — baseline training sample

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Use the sidebar to navigate between pages.

---

### 🧠 Author
Amarjit Jha | Aspiring Data Scientist & ML Engineer | Targeting 2026 Internships in Pune [LinkedIn Profile Link] | [Portfolio Link]
