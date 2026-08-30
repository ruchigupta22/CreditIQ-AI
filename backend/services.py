import json
import joblib
import pandas as pd
from pathlib import Path
import shap
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load config
with open(MODELS_DIR / "config.json", "r") as f:
    config = json.load(f)

# Feature order
FEATURE_ORDER = config["feature_order"]
FEATURE_MAP = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit Utilisation Ratio",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30–59 Days Past Due",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines",
    "NumberOfTimes90DaysLate": "90+ Days Late",
    "NumberRealEstateLoansOrLines": "Real Estate Loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89 Days Past Due",
    "NumberOfDependents": "Dependents",
}

REASON_CODE_TEMPLATES = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving credit utilisation ratio (reported: {value}) was a significant contributor to the risk assessment.",
    "age": "Applicant age (reported: {value}) was a significant contributor to the risk assessment.",
    "NumberOfTime30-59DaysPastDueNotWorse": "History of payments 30-59 days past due (reported: {value} occurrences) was a significant contributor to the risk assessment.",
    "DebtRatio": "Debt-to-income ratio (reported: {value}) was a significant contributor to the risk assessment.",
    "MonthlyIncome": "Monthly income level (reported: {value}) was a significant contributor to the risk assessment.",
    "NumberOfOpenCreditLinesAndLoans": "Number of open credit lines and loans (reported: {value}) was a significant contributor to the risk assessment.",
    "NumberOfTimes90DaysLate": "History of payments 90+ days late (reported: {value} occurrences) was a significant contributor to the risk assessment.",
    "NumberRealEstateLoansOrLines": "Number of real estate loans or lines (reported: {value}) was a significant contributor to the risk assessment.",
    "NumberOfTime60-89DaysPastDueNotWorse": "History of payments 60-89 days past due (reported: {value} occurrences) was a significant contributor to the risk assessment.",
    "NumberOfDependents": "Number of dependents (reported: {value}) was a significant contributor to the risk assessment.",
}

# Cost assumption for cost-sensitive decisioning (see notebooks/04_threshold_optimization.ipynb)
COST_FN = 5.0
COST_FP = 1.0

# Load models
xgb_model = joblib.load(MODELS_DIR / config["xgboost_model"])
xgb_explainer = shap.TreeExplainer(xgb_model)

# Calibrated model - used only for probability/expected-loss in the CalibratedCostOptimized
# strategy. SHAP explainability continues to use the original xgb_model above, since
# CalibratedClassifierCV wraps the model in a way TreeExplainer can't interpret directly.
xgb_model_calibrated = joblib.load(MODELS_DIR / config["xgboost_calibrated_model"])

logistic_models = {
    name: joblib.load(MODELS_DIR / path)
    for name, path in config["logistic_models"].items()
}

scaler = joblib.load(MODELS_DIR / config["scaler"])


def predict_risk(data: dict):
    """
    Predict risk using selected model and strategy.
    """

    model_name = data.pop("model")
    strategy = data.pop("strategy")

    # Convert schema names to training feature names
    data["NumberOfTime30-59DaysPastDueNotWorse"] = data.pop(
        "NumberOfTime30_59DaysPastDueNotWorse"
    )

    data["NumberOfTime60-89DaysPastDueNotWorse"] = data.pop(
        "NumberOfTime60_89DaysPastDueNotWorse"
    )

    X = pd.DataFrame([data])[FEATURE_ORDER]

    # Select model
    if model_name == "XGBoost":
        if strategy == "CalibratedCostOptimized":
            model = xgb_model_calibrated
        else:
            model = xgb_model
        prob = model.predict_proba(X)[0][1]

    else:
        model = logistic_models[model_name]

        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0][1]

    threshold = config["thresholds"][strategy]

    prediction = int(prob >= threshold)

    # Risk categories
    if prob < 0.30:
        risk = "Low Risk"
    elif prob < 0.60:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    decision = "Reject" if prediction == 1 else "Approve"

    if decision == "Approve":
        expected_loss_note = round(prob * COST_FN, 3)
    else:
        expected_loss_note = round((1 - prob) * COST_FP, 3)

    return {
        "probability": round(float(prob), 3),
        "prediction": prediction,
        "threshold": threshold,
        "risk_category": risk,
        "decision": decision,
        "model": model_name,
        "strategy": strategy,
        "expected_loss_estimate": expected_loss_note,
        "cost_assumption": {"cost_fn": COST_FN, "cost_fp": COST_FP},
    }
def explain_prediction(data: dict):
    """
    Generate SHAP explanations for XGBoost predictions.
    """

    data.pop("model", None)
    data.pop("strategy", None)

    data["NumberOfTime30-59DaysPastDueNotWorse"] = data.pop(
        "NumberOfTime30_59DaysPastDueNotWorse"
    )

    data["NumberOfTime60-89DaysPastDueNotWorse"] = data.pop(
        "NumberOfTime60_89DaysPastDueNotWorse"
    )

    original_data = dict(data)

    X = pd.DataFrame([data])[FEATURE_ORDER]

    shap_values = xgb_explainer.shap_values(X)

    drivers = []

    for feature, value in zip(FEATURE_ORDER, shap_values[0]):

        drivers.append({
            "feature": FEATURE_MAP[feature],
            "shap": round(float(value), 4),
        })

    drivers = sorted(
        drivers,
        key=lambda x: abs(x["shap"]),
        reverse=True,
    )

    base_value = xgb_explainer.expected_value

    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[0])

    reason_codes = generate_reason_codes(drivers, original_data)

    return {
        "base_value": round(float(base_value), 4),
        "drivers": drivers,
        "reason_codes": reason_codes,
    }


def generate_reason_codes(drivers, borrower_data, top_n=4):
    """
    Convert top SHAP drivers into human-readable lending reason codes.
    Only considers features with positive SHAP (i.e. pushing risk upward).
    """
    risk_increasing = [d for d in drivers if d["shap"] > 0]
    top_drivers = risk_increasing[:top_n]

    reason_codes = []
    for driver in top_drivers:
        feature_key = next(
            (k for k, v in FEATURE_MAP.items() if v == driver["feature"]),
            None
        )
        if feature_key and feature_key in REASON_CODE_TEMPLATES:
            value = borrower_data.get(feature_key, "N/A")
            reason_text = REASON_CODE_TEMPLATES[feature_key].format(value=value)
            reason_codes.append({
                "feature": driver["feature"],
                "shap_value": driver["shap"],
                "reason": reason_text,
            })

    return reason_codes
