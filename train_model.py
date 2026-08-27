"""
train_model.py
================
Cardiovascular Disease (CVD) Prediction - Model Training Script.

This script performs the FULL machine learning workflow from scratch,
based entirely on the real dataset extracted from the project's
Power BI model (data/heart_data.xlsx, 270,064 real patient records):

    1. Load & inspect the dataset
    2. Clean it (duplicates / missing values check)
    3. Exploratory Data Analysis (EDA) -> saves charts to static/images/eda/
    4. Feature engineering + preprocessing pipeline (encoding/scaling)
    5. Train/test split (stratified)
    6. Train multiple classification models
    7. Evaluate every model on the same held-out test set
    8. Select the best model based on ROC-AUC (tie-break: F1)
    9. Persist the winning full pipeline (preprocessing + model) to
       model/best_model.pkl, and all metrics/metadata to model/metrics.json

Run with:
    python train_model.py

The Flask app (app.py) never retrains - it just loads the artifacts
produced here.
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart_data.xlsx"
MODEL_DIR = BASE_DIR / "model"
EDA_DIR = BASE_DIR / "static" / "images" / "eda"
MODEL_DIR.mkdir(exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

# Brand colors matching the original Power BI dashboard palette
COLOR_CVD = "#C8102E"       # deep red
COLOR_NONCVD = "#5A0A14"    # burgundy
PALETTE = [COLOR_CVD, COLOR_NONCVD]

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading dataset")
print("=" * 70)

df_raw = pd.read_excel(DATA_PATH)
print(f"Loaded dataset: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
print(f"Columns: {list(df_raw.columns)}")

n_missing = int(df_raw.isnull().sum().sum())
n_dupes_all_cols = int(df_raw.duplicated().sum())
n_dupes_no_id = int(df_raw.duplicated(subset=[c for c in df_raw.columns if c != "id"]).sum())

print(f"Missing values: {n_missing}")
print(f"Fully duplicated rows: {n_dupes_all_cols}")
print(f"Duplicated rows (ignoring id): {n_dupes_no_id}")

dataset_info = {
    "n_rows": int(df_raw.shape[0]),
    "n_columns": int(df_raw.shape[1]),
    "columns": list(df_raw.columns),
    "missing_values": n_missing,
    "duplicate_rows": n_dupes_all_cols,
    "duplicate_rows_ignoring_id": n_dupes_no_id,
}

# --------------------------------------------------------------------------
# 2. CLEANING
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Cleaning")
print("=" * 70)

df = df_raw.copy()
before = len(df)
df = df.dropna()
df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"])
after = len(df)
print(f"Rows before cleaning: {before:,} | after cleaning: {after:,} | removed: {before - after:,}")

# Target variable: CVD -> 'cvd' / 'non-cvd'
df["target"] = df["CVD"].str.strip().str.lower().map({"cvd": 1, "non-cvd": 0})
assert df["target"].isnull().sum() == 0, "Unmapped target values found!"

target_dist = df["target"].value_counts().to_dict()
print(f"Target distribution -> CVD (1): {target_dist.get(1,0):,} | Non-CVD (0): {target_dist.get(0,0):,}")

dataset_info["rows_after_cleaning"] = int(after)
dataset_info["target_distribution"] = {
    "cvd": int(target_dist.get(1, 0)),
    "non_cvd": int(target_dist.get(0, 0)),
    "cvd_pct": round(100 * target_dist.get(1, 0) / after, 2),
    "non_cvd_pct": round(100 * target_dist.get(0, 0) / after, 2),
}

# --------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Exploratory Data Analysis")
print("=" * 70)

df_eda = df.copy()
df_eda["CVD_label"] = df_eda["target"].map({1: "CVD", 0: "Non-CVD"})

# --- 3.1 Target distribution ---
fig, ax = plt.subplots(figsize=(6, 5))
counts = df_eda["CVD_label"].value_counts()
ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=PALETTE,
       startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Target Distribution: CVD vs Non-CVD", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA_DIR / "target_distribution.png", dpi=130)
plt.close()

# --- 3.2 Age distribution by CVD ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df_eda, x="age", hue="CVD_label", bins=30, kde=True,
             palette=PALETTE, element="step", ax=ax)
ax.set_title("Age Distribution by CVD Status", fontsize=13, fontweight="bold")
ax.set_xlabel("Age (years)")
plt.tight_layout()
plt.savefig(EDA_DIR / "age_distribution.png", dpi=130)
plt.close()

# --- 3.3 Numerical feature distributions ---
num_cols = ["age", "height", "weight", "BMI", "ap_hi", "ap_lo"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), num_cols):
    sns.histplot(df_eda[col], bins=40, color=COLOR_CVD, ax=ax, kde=True)
    ax.set_title(col)
plt.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA_DIR / "numerical_distributions.png", dpi=130)
plt.close()

# --- 3.4 Correlation matrix ---
corr_df = df_eda[num_cols + ["target"]].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_df, annot=True, cmap="Reds", fmt=".2f", ax=ax, cbar=True,
            linewidths=0.5, linecolor="white")
ax.set_title("Correlation Matrix (Numerical Features & Target)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA_DIR / "correlation_matrix.png", dpi=130)
plt.close()

# --- 3.5 CVD by BP Class ---
fig, ax = plt.subplots(figsize=(8, 5))
order = ["Normal", "Elevated", "Stage 1", "Stage 2"]
ct = pd.crosstab(df_eda["BP_Class"], df_eda["CVD_label"]).reindex(order)
ct.plot(kind="bar", stacked=False, color=PALETTE, ax=ax)
ax.set_title("CVD vs Non-CVD by Blood Pressure Class", fontsize=13, fontweight="bold")
ax.set_xlabel("BP Class")
ax.set_ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(EDA_DIR / "bp_class.png", dpi=130)
plt.close()

# --- 3.6 CVD by Cholesterol ---
fig, ax = plt.subplots(figsize=(8, 5))
order_c = ["normal", "borderline high", "elevated"]
ct = pd.crosstab(df_eda["cholesterol"], df_eda["CVD_label"]).reindex(order_c)
ct.plot(kind="line", marker="o", color=PALETTE, ax=ax, linewidth=2.5)
ax.set_title("CVD vs Non-CVD by Cholesterol Level", fontsize=13, fontweight="bold")
ax.set_xlabel("Cholesterol")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(EDA_DIR / "cholesterol.png", dpi=130)
plt.close()

# --- 3.7 CVD by Glucose status ---
fig, ax = plt.subplots(figsize=(8, 5))
order_g = ["normal", "pre diabetic", "diabetic"]
ct = pd.crosstab(df_eda["gluc"], df_eda["CVD_label"]).reindex(order_g)
ct.plot(kind="bar", color=PALETTE, ax=ax)
ax.set_title("CVD vs Non-CVD by Glucose Status", fontsize=13, fontweight="bold")
ax.set_xlabel("Glucose Status")
ax.set_ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(EDA_DIR / "glucose.png", dpi=130)
plt.close()

# --- 3.8 Lifestyle risk factors (smoke/alco/active) vs CVD rate ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, title in zip(axes, ["smoke", "alco", "active"],
                           ["Smoking", "Alcohol", "Physical Activity"]):
    rate = df_eda.groupby(col)["target"].mean() * 100
    rate.plot(kind="bar", color=[COLOR_NONCVD, COLOR_CVD], ax=ax)
    ax.set_title(f"CVD Rate by {title}")
    ax.set_ylabel("CVD Rate (%)")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
plt.suptitle("Lifestyle Risk Factors vs CVD Rate", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA_DIR / "lifestyle_factors.png", dpi=130)
plt.close()

# --- 3.9 BMI class vs CVD ---
fig, ax = plt.subplots(figsize=(8, 5))
order_b = ["Under weight", "Normal weight", "Over weight", "Obese"]
ct = pd.crosstab(df_eda["BMI_Class"], df_eda["CVD_label"]).reindex(order_b)
ct.plot(kind="bar", color=PALETTE, ax=ax)
ax.set_title("CVD vs Non-CVD by BMI Class", fontsize=13, fontweight="bold")
ax.set_xlabel("BMI Class")
ax.set_ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(EDA_DIR / "bmi_class.png", dpi=130)
plt.close()

print(f"Saved 9 EDA charts to {EDA_DIR}")

eda_summary = {
    "numeric_feature_stats": {
        col: {
            "mean": round(float(df_eda[col].mean()), 2),
            "std": round(float(df_eda[col].std()), 2),
            "min": float(df_eda[col].min()),
            "max": float(df_eda[col].max()),
        }
        for col in num_cols
    },
    "categorical_value_counts": {
        col: df_eda[col].value_counts().to_dict()
        for col in ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "BMI_Class", "BP_Class"]
    },
}

# --------------------------------------------------------------------------
# 4. FEATURE ENGINEERING + PREPROCESSING
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Preprocessing")
print("=" * 70)

# We predict directly from the RAW vitals a patient/clinician would enter.
# BMI is engineered from height & weight (matches the dataset's own BMI
# calculation exactly - verified: max abs diff < 0.005).
# The pre-binned categorical columns (BMI_Class, BP_Class, Blood pressure,
# Age Distribution) are deterministic functions of these raw numeric
# features, so they are excluded from the model inputs to avoid redundant/
# collinear encodings - the model learns the same signal directly from the
# continuous features instead.

NUMERIC_FEATURES = ["age", "height", "weight", "BMI", "ap_hi", "ap_lo"]
CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

df["BMI_calc"] = df["weight"] / ((df["height"] / 100) ** 2)
df["BMI"] = df["BMI_calc"]  # use engineered BMI consistently (matches original)

X = df[FEATURES].copy()
y = df["target"].copy()

print(f"Feature set used for ML ({len(FEATURES)}): {FEATURES}")
print(f"Target: CVD (1 = cvd, 0 = non-cvd)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES),
    ]
)

# --------------------------------------------------------------------------
# 5. TRAIN MULTIPLE MODELS
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Training models")
print("=" * 70)

candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=14, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

results = {}
fitted_pipelines = {}

for name, clf in candidate_models.items():
    t0 = time.time()
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    results[name] = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
        "train_time_sec": round(train_time, 2),
        "roc_curve": {"fpr": fpr.tolist()[::5], "tpr": tpr.tolist()[::5]},
    }
    fitted_pipelines[name] = pipe

    print(f"[{name}] acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  "
          f"f1={f1:.4f}  roc_auc={auc:.4f}  ({train_time:.1f}s)")

# --------------------------------------------------------------------------
# 6. MODEL SELECTION
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Model selection")
print("=" * 70)

best_name = max(results.items(), key=lambda kv: (kv[1]["roc_auc"], kv[1]["f1_score"]))[0]
best_pipeline = fitted_pipelines[best_name]
best_metrics = results[best_name]

print(f"BEST MODEL: {best_name}")
print(json.dumps(best_metrics, indent=2))

# --------------------------------------------------------------------------
# 7. SAVE ARTIFACTS
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Saving artifacts")
print("=" * 70)

joblib.dump(best_pipeline, MODEL_DIR / "best_model.pkl")
joblib.dump(FEATURES, MODEL_DIR / "feature_list.pkl")

# Confusion matrix plot for the winning model
cm = np.array(best_metrics["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax,
            xticklabels=["Non-CVD", "CVD"], yticklabels=["Non-CVD", "CVD"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=130)
plt.savefig(EDA_DIR.parent / "confusion_matrix.png", dpi=130)
plt.close()

# ROC curve comparison for all models
fig, ax = plt.subplots(figsize=(7, 6))
colors_cycle = ["#C8102E", "#5A0A14", "#E8969E", "#7A1F2B"]
for (name, res), c in zip(results.items(), colors_cycle):
    ax.plot(res["roc_curve"]["fpr"], res["roc_curve"]["tpr"], label=f"{name} (AUC={res['roc_auc']:.3f})",
            color=c, linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(MODEL_DIR / "roc_curve.png", dpi=130)
plt.savefig(EDA_DIR.parent / "roc_curve.png", dpi=130)
plt.close()

metrics_payload = {
    "best_model": best_name,
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "categorical_options": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
    "test_size": 0.2,
    "random_state": RANDOM_STATE,
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "results": results,
    "dataset_info": dataset_info,
    "eda_summary": eda_summary,
}

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics_payload, f, indent=2)

print(f"Saved best_model.pkl, feature_list.pkl, metrics.json to {MODEL_DIR}")
print(f"Saved confusion_matrix.png and roc_curve.png")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Best model: {best_name}")
print(f"  Accuracy : {best_metrics['accuracy']}")
print(f"  Precision: {best_metrics['precision']}")
print(f"  Recall   : {best_metrics['recall']}")
print(f"  F1-score : {best_metrics['f1_score']}")
print(f"  ROC-AUC  : {best_metrics['roc_auc']}")
print("\nRun 'python app.py' to launch the web application.")
