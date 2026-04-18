import joblib
import shap
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
from data_processing import load_and_preprocess_data

def train_and_export_model():
    print("Starting Model Training Pipeline...")
    
    # 1. Load the processed data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # 2. Calculate Class Imbalance Weight
    # XGBoost needs to know how imbalanced the data is. 
    # scale_pos_weight = (Number of Negative Cases) / (Number of Positive Cases)
    neg_cases = (y_train == 0).sum()
    pos_cases = (y_train == 1).sum()
    imbalance_weight = neg_cases / pos_cases
    print(f"Calculated scale_pos_weight for XGBoost: {imbalance_weight:.2f}")

    # 3. Initialize and Train XGBoost Classifier
    # We use area under the Precision-Recall curve (aucpr) as our evaluation metric 
    # because accuracy is useless on highly imbalanced datasets.
    xgb_model = XGBClassifier(
        scale_pos_weight=imbalance_weight,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="aucpr"
    )
    
    print("Training XGBoost model...")
    xgb_model.fit(X_train, y_train)
    
    # 4. Evaluate the Model
    predictions = xgb_model.predict(X_test)
    print("\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test, predictions))
    
    # 5. Fit the SHAP Explainer (Explainable AI)
    # This allows us to break down every single prediction by feature importance later in the UI
    print("Fitting SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(xgb_model)
    
    # 6. Export Artifacts for Streamlit
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    joblib.dump(explainer, "models/shap_explainer.pkl")
    
    # We also save a small baseline sample of X_train to be used by SHAP's plotting functions
    X_train.head(100).to_pickle("models/X_train_sample.pkl")
    
    print("\nModel, Explainer, and Baseline Data successfully saved to models/ directory.")

if __name__ == "__main__":
    train_and_export_model()