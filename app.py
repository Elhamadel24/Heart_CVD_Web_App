"""
app.py
======
Flask web application for the AI-Powered Cardiovascular Disease
Prediction & Analytics project.

This app NEVER retrains a model - it loads the pipeline and metrics
already produced by `train_model.py` (model/best_model.pkl,
model/metrics.json) and serves:

    /              Home / landing page
    /prediction    Interactive CVD risk prediction form
    /analytics     Data Analytics / EDA page
    /performance   Model performance & comparison page
    /powerbi       Power BI Analytics page (static dashboard export)
    /about         About the project

Run with:
    python app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------------
MODEL_PATH = MODEL_DIR / "best_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    raise RuntimeError(
        "Model artifacts not found. Please run `python train_model.py` first "
        "to train the model before starting the web app."
    )

pipeline = joblib.load(MODEL_PATH)
with open(METRICS_PATH) as f:
    METRICS = json.load(f)

FEATURES = METRICS["features"]
NUMERIC_FEATURES = METRICS["numeric_features"]
CATEGORICAL_FEATURES = METRICS["categorical_features"]
CATEGORICAL_OPTIONS = METRICS["categorical_options"]
BEST_MODEL_NAME = METRICS["best_model"]
BEST_METRICS = METRICS["results"][BEST_MODEL_NAME]
DATASET_INFO = METRICS["dataset_info"]

DASHBOARD_PAGES = [
    {"file": "powerbi_page-1.png", "title": "Cover Page"},
    {"file": "powerbi_page-2.png", "title": "Cardiovascular Risk Factor Analysis"},
    {"file": "powerbi_page-3.png", "title": "Lifestyle & Clinical Risk Patterns"},
    {"file": "powerbi_page-4.png", "title": "Personalized Risk Estimator"},
]


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/")
def home():
    stats = {
        "total_records": DATASET_INFO["n_rows"],
        "cvd_pct": DATASET_INFO["target_distribution"]["cvd_pct"],
        "non_cvd_pct": DATASET_INFO["target_distribution"]["non_cvd_pct"],
        "best_model": BEST_MODEL_NAME,
        "accuracy": round(BEST_METRICS["accuracy"] * 100, 1),
        "roc_auc": BEST_METRICS["roc_auc"],
        "n_features": len(FEATURES),
    }
    return render_template("index.html", stats=stats)


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None
    form_values = None

    if request.method == "POST":
        try:
            form_values = {
                "age": int(request.form["age"]),
                "gender": request.form["gender"],
                "height": float(request.form["height"]),
                "weight": float(request.form["weight"]),
                "ap_hi": float(request.form["ap_hi"]),
                "ap_lo": float(request.form["ap_lo"]),
                "cholesterol": request.form["cholesterol"],
                "gluc": request.form["gluc"],
                "smoke": request.form["smoke"],
                "alco": request.form["alco"],
                "active": request.form["active"],
            }

            # Same feature engineering used during training: BMI from height/weight
            bmi = form_values["weight"] / ((form_values["height"] / 100) ** 2)

            input_row = {
                "age": form_values["age"],
                "height": form_values["height"],
                "weight": form_values["weight"],
                "BMI": round(bmi, 2),
                "ap_hi": form_values["ap_hi"],
                "ap_lo": form_values["ap_lo"],
                "gender": form_values["gender"],
                "cholesterol": form_values["cholesterol"],
                "gluc": form_values["gluc"],
                "smoke": form_values["smoke"],
                "alco": form_values["alco"],
                "active": form_values["active"],
            }
            X_input = pd.DataFrame([input_row])[FEATURES]

            pred = int(pipeline.predict(X_input)[0])
            proba = pipeline.predict_proba(X_input)[0]
            prob_non_cvd, prob_cvd = float(proba[0]), float(proba[1])

            if prob_cvd >= 0.66:
                risk_level, risk_class = "HIGH RISK", "risk-high"
            elif prob_cvd >= 0.4:
                risk_level, risk_class = "MODERATE RISK", "risk-moderate"
            else:
                risk_level, risk_class = "LOW RISK", "risk-low"

            result = {
                "prediction": "High Risk of CVD" if pred == 1 else "Low Risk of CVD",
                "is_cvd": bool(pred == 1),
                "prob_cvd": round(prob_cvd * 100, 1),
                "prob_non_cvd": round(prob_non_cvd * 100, 1),
                "risk_level": risk_level,
                "risk_class": risk_class,
                "bmi": round(bmi, 1),
                "model_used": BEST_MODEL_NAME,
            }
        except (KeyError, ValueError) as e:
            result = {"error": f"Invalid input: {e}"}

    return render_template(
        "prediction.html",
        options=CATEGORICAL_OPTIONS,
        result=result,
        form_values=form_values,
    )


@app.route("/analytics")
def analytics():
    eda_charts = [
        {"file": "target_distribution.png", "title": "Target Distribution (CVD vs Non-CVD)"},
        {"file": "age_distribution.png", "title": "Age Distribution by CVD Status"},
        {"file": "numerical_distributions.png", "title": "Numerical Feature Distributions"},
        {"file": "correlation_matrix.png", "title": "Correlation Matrix"},
        {"file": "bp_class.png", "title": "CVD by Blood Pressure Class"},
        {"file": "cholesterol.png", "title": "CVD by Cholesterol Level"},
        {"file": "glucose.png", "title": "CVD by Glucose Status"},
        {"file": "lifestyle_factors.png", "title": "Lifestyle Risk Factors vs CVD Rate"},
        {"file": "bmi_class.png", "title": "CVD by BMI Class"},
    ]
    return render_template(
        "analytics.html",
        dataset_info=DATASET_INFO,
        eda_summary=METRICS["eda_summary"],
        eda_charts=eda_charts,
    )


@app.route("/performance")
def performance():
    return render_template(
        "performance.html",
        results=METRICS["results"],
        best_model=BEST_MODEL_NAME,
        best_metrics=BEST_METRICS,
        n_train=METRICS["n_train"],
        n_test=METRICS["n_test"],
    )


@app.route("/powerbi")
def powerbi():
    return render_template("powerbi.html")


@app.route("/about")
def about():
    return render_template(
        "about.html",
        dataset_info=DATASET_INFO,
        features=FEATURES,
        best_model=BEST_MODEL_NAME,
        best_metrics=BEST_METRICS,
        all_results=METRICS["results"],
    )


@app.route("/download/pbix")
def download_pbix():
    return send_from_directory(
        BASE_DIR / "data", "dashboard.pbix",
        as_attachment=True,
        download_name="Where_Heart_Data_Becomes_Insight.pbix",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint (optional, for programmatic access)."""
    data = request.get_json(force=True)
    try:
        bmi = float(data["weight"]) / ((float(data["height"]) / 100) ** 2)
        row = {
            "age": int(data["age"]),
            "height": float(data["height"]),
            "weight": float(data["weight"]),
            "BMI": round(bmi, 2),
            "ap_hi": float(data["ap_hi"]),
            "ap_lo": float(data["ap_lo"]),
            "gender": data["gender"],
            "cholesterol": data["cholesterol"],
            "gluc": data["gluc"],
            "smoke": data["smoke"],
            "alco": data["alco"],
            "active": data["active"],
        }
        X_input = pd.DataFrame([row])[FEATURES]
        pred = int(pipeline.predict(X_input)[0])
        proba = pipeline.predict_proba(X_input)[0]
        return jsonify({
            "prediction": "High Risk of CVD" if pred == 1 else "Low Risk of CVD",
            "probability_cvd": round(float(proba[1]) * 100, 1),
            "probability_non_cvd": round(float(proba[0]) * 100, 1),
            "model_used": BEST_MODEL_NAME,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print(f"Loaded model: {BEST_MODEL_NAME} (Accuracy={BEST_METRICS['accuracy']}, ROC-AUC={BEST_METRICS['roc_auc']})")
    app.run(debug=True, host="0.0.0.0", port=5000)
