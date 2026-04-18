# 🏭 Industry 4.0: Predictive Maintenance Operations Center

## 📌 Business Context
In modern manufacturing (especially within automotive hubs like Pune), unplanned machine downtime costs lakhs of rupees per hour. Traditional "scheduled maintenance" is often wasteful, replacing parts that are still perfectly functional. 

This project is an end-to-end **Explainable AI (XAI)** dashboard designed for shop-floor managers. It ingests live telemetry data (RPM, Torque, Temperatures) from CNC machinery and predicts mechanical failures *before* they occur, shifting operations from a reactive to a proactive model.

## 🚀 Key Features
* **Machine Learning Engine:** An XGBoost binary classifier optimized for severe class imbalance (utilizing algorithmic `scale_pos_weight` rather than naive SMOTE).
* **Explainable AI (XAI):** Integration of `SHAP` (SHapley Additive exPlanations) to break the "black box" of AI. The dashboard visually explains exactly *why* a machine is failing (e.g., "High Torque + Low RPM indicates a jam").
* **Fleet Command Center:** A Streamlit-powered frontend featuring a Red/Yellow/Green prioritization logic engine to instantly alert managers to critical machinery.
* **Real-time Diagnostics:** Interactive Plotly telemetry gauges and a manual what-if simulator for stress-testing.

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

## 📸 Dashboard Previews

1. `![Fleet Overview](image/1.png)` - *Showing the R/Y/G tabular dashboard.*
2. `![Diagnostics](image/2.png)` - *Showing the Plotly Gauges and SHAP Waterfall.*
3. `![EDA Notebook](image/3.png)` - *Showing the Torque vs. RPM failure boundary.*

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
Amarjit Jha | Aspiring Data Scientist & ML Engineer | Targeting 2026 Internships in Pune [LinkedIn Profile](https://www.linkedin.com/in/amarjit-jha-556656280/) | [Resum Link](https://drive.google.com/file/d/14_12w6018r6825y7V032X9L697979797/view?usp=sharing) | [Portfolio Link](https://amarjeetjha17.github.io/Portfolio/)
