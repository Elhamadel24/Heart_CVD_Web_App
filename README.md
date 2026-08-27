# CardioAI — AI-Powered Cardiovascular Disease Prediction & Analytics

A complete, end-to-end web application that combines data analysis, machine
learning, and the original Power BI dashboard ("Where Heart Data Becomes
Insight") into a single professional healthcare analytics platform.

---

## 1. Project Overview

CardioAI lets a user:

- Explore real exploratory data analysis (EDA) on a 270,064-record cardiovascular dataset
- Compare multiple trained machine learning classifiers head-to-head
- Enter a patient's vitals and get an instant, model-driven CVD risk prediction
- View the original Power BI dashboard pages, and download the original `.pbix` file

Everything shown in the app — dataset statistics, charts, model metrics, and
predictions — is generated from real data and a real trained model. Nothing
is hard-coded or simulated.

## 2. Dataset

The dataset (`data/heart_data.xlsx`) was extracted directly from the data
model embedded inside the project's own Power BI file
(`Where_Heart_Data_Becomes_Insight.pbix`), so the web app, the ML model, and
the Power BI dashboard are all built from the exact same source data.

- **270,064 rows**, 18 columns
- No missing values, no duplicate records
- Target: `CVD` (`cvd` / `non-cvd`) → 170,000 CVD (62.9%) / 100,064 Non-CVD (37.1%)

| Column | Description |
|---|---|
| age | Patient age (years) |
| gender | male / female |
| height, weight | cm / kg |
| BMI | Body Mass Index (weight / height²), engineered feature |
| ap_hi, ap_lo | Systolic / diastolic blood pressure (mmHg) |
| cholesterol | normal / borderline high / elevated |
| gluc | normal / pre diabetic / diabetic |
| smoke | smoke / no-smoke |
| alco | alco / non-alco |
| active | active / non-active |
| BMI_Class, BP_Class, Blood pressure, Age Distribution | Pre-binned categories derived from the numeric columns (used in the Power BI dashboard, not fed to the model directly — see below) |
| CVD | Target label |

## 3. Data Preprocessing

Handled in `train_model.py`:

- Missing-value and duplicate checks (dataset was already clean: 0 of each)
- Target encoding: `cvd` → 1, `non-cvd` → 0
- **Feature engineering**: BMI is (re-)computed from `height` and `weight`
  (verified to match the dataset's own BMI column to within 0.005)
- The model is trained on the **raw clinical inputs** a user would actually
  enter — `age, height, weight, BMI, ap_hi, ap_lo, gender, cholesterol,
  gluc, smoke, alco, active` — rather than the pre-binned category columns
  (`BMI_Class`, `BP_Class`, `Blood pressure`, `Age Distribution`), which are
  deterministic functions of those same raw numbers and would just add
  redundant, collinear encodings.
- Numeric features are standard-scaled; categorical features are one-hot
  encoded, all inside a single `ColumnTransformer` + `Pipeline` so the
  **exact same transformation is applied to training data and to live user
  input on the prediction page** — no train/serve skew.
- Stratified 80/20 train/test split (`random_state=42`) so class balance is
  preserved in both sets.

## 4. Exploratory Data Analysis

`train_model.py` generates 9 charts from the real data into
`static/images/eda/`:

- Target distribution (CVD vs Non-CVD)
- Age distribution by CVD status
- Numerical feature distributions (age, height, weight, BMI, ap_hi, ap_lo)
- Correlation matrix
- CVD by blood pressure class
- CVD by cholesterol level
- CVD by glucose status
- Lifestyle risk factors (smoking / alcohol / activity) vs CVD rate
- CVD by BMI class

These are all displayed on the **Data Analytics** page of the web app.

## 5. Machine Learning

Four classification models are trained on identical data with identical
preprocessing:

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Simple non-linear baseline |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting (`HistGradientBoostingClassifier`) | Boosted ensemble, tuned for large tabular data |

Each is evaluated on the same held-out test set using **Accuracy,
Precision, Recall, F1-score, ROC-AUC, and a Confusion Matrix**.

## 6. Model Selection

The best model is chosen automatically — **not assumed in advance** — by
sorting all trained models on **ROC-AUC** (tie-break: F1-score) and picking
the top one. In the most recent training run this was:

- **Gradient Boosting** — Accuracy 93.27%, Precision 94.95%, Recall 94.34%, F1 94.64%, ROC-AUC 0.983

(Re-run `train_model.py` to reproduce/refresh these numbers — they are also
saved to `model/metrics.json` and displayed live on the Model Performance
page.)

The full winning pipeline (preprocessing + model) is saved with `joblib` to
`model/best_model.pkl`, so the Flask app never retrains — it just loads this
file at startup.

## 7. How to Train the Model

```bash
pip install -r requirements.txt
python train_model.py
```

This will:
1. Load and clean `data/heart_data.xlsx`
2. Generate EDA charts into `static/images/eda/`
3. Train all four models
4. Evaluate and select the best one
5. Save `model/best_model.pkl`, `model/feature_list.pkl`, `model/metrics.json`
6. Save `static/images/confusion_matrix.png` and `static/images/roc_curve.png`
7. Print a full summary of results to the console

## 8. How to Run the Web App

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

> The app requires the artifacts produced by `train_model.py` to already
> exist in `model/`. If they're missing, run `train_model.py` first.

## 9. Pages

| Page | Route | Description |
|---|---|---|
| Home | `/` | Landing page with project stats and heart visual |
| CVD Prediction | `/prediction` | Interactive risk estimator form |
| Data Analytics | `/analytics` | Dataset statistics and EDA charts |
| Model Performance | `/performance` | Real model comparison, confusion matrix, ROC curve |
| Power BI Analytics | `/powerbi` | Original Power BI dashboard pages + `.pbix` download |
| About | `/about` | Project & pipeline summary |

## 10. Power BI Integration

The original Power BI report — **"Where Heart Data Becomes Insight"** — is
kept **unmodified** in `data/dashboard.pbix`, downloadable from the Power BI
Analytics page.

**Why static images instead of a live embed?** Power BI's "Publish to Web"
and Power BI Embedded both require either a Microsoft work/school
(organizational) account with Power BI Pro/Premium licensing, or an Azure
app registration with embedding credentials. None of that is required here:
the four report pages were exported as high-resolution PNGs directly from
the `.pbix` file and are served as static assets
(`static/dashboard/powerbi_page_*.png`), so the dashboard is fully viewable
offline, with zero authentication.

See [`powerbi_embed_future.md`](powerbi_embed_future.md) for how to upgrade
this to a live embedded report later, if embedding credentials become
available.

## 11. Project Structure

```
heart_cvd_web_app/
├── app.py                      # Flask application
├── train_model.py              # Full ML training pipeline
├── requirements.txt
├── README.md
├── powerbi_embed_future.md
│
├── data/
│   ├── heart_data.xlsx         # Real dataset (extracted from the .pbix data model)
│   └── dashboard.pbix          # Original, unmodified Power BI file
│
├── model/
│   ├── best_model.pkl          # Winning pipeline (preprocessing + model)
│   ├── feature_list.pkl
│   ├── metrics.json            # All training/evaluation results
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   ├── images/
│   │   ├── heart.png           # Anatomical heart illustration (from original dashboard)
│   │   ├── hero_bg.png         # Blood-cell hero background (from original dashboard)
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── eda/                # 9 EDA charts generated by train_model.py
│   └── dashboard/
│       ├── powerbi_page-1.png
│       ├── powerbi_page-2.png
│       ├── powerbi_page-3.png
│       └── powerbi_page-4.png
│
└── templates/
    ├── base.html
    ├── index.html
    ├── prediction.html
    ├── analytics.html
    ├── performance.html
    ├── powerbi.html
    └── about.html
```

## 12. Medical Disclaimer

> This tool is for educational and research purposes only and is not a
> substitute for professional medical advice, diagnosis, or treatment.
