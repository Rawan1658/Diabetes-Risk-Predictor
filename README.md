# Diabetes Risk Predictor

> Rawan Abdelelah · Student ID 211001658 · Machine Learning, May 2026

## TL;DR

A 21-question health survey is fed into a soft-voting ensemble of three calibrated classifiers, behind a small Flask UI. ROC-AUC on a held-out test split is **0.814**; at the F1-tuned operating point (probability cutoff = 0.24) the model captures ~60 % of diabetic respondents at ~39 % precision.

## Demo

```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```

Or on Windows: `run.bat`.

## Data

| Field | Value |
|---|---|
| Source | CDC BRFSS 2015 — Diabetes Health Indicators (Kaggle release) |
| Raw rows | 253,680 |
| After dropping duplicate questionnaires | 229,474 |
| Columns | 22 (21 features + binary target) |
| Target | `Diabetes_binary` (1 = diabetic, 0 = non-diabetic) |
| Positive rate | ≈ 14 % |

Cleaning is already applied to the shipped `data/diabetes.csv`. The columns are already numeric and fully populated — the only step that mattered was deduplication (24,206 exact duplicate rows dropped, since BRFSS responses are dense binary fields and identical questionnaires happen often).

Two engineered features sit on top of the 21 raw indicators:

| Feature | Formula | Range |
|---|---|---|
| `HealthyHabits` | `PhysActivity + Fruits + Veggies` | 0 → 3 |
| `RiskFactors`   | `HighBP + HighChol + Stroke + HeartDiseaseorAttack` | 0 → 4 |

Both are computed inside `app.py` from the raw form fields too, so the form stays at 21 questions.

## Pipeline

```
StandardScaler  ──►  SelectKBest(k=15, score_func=f_classif)  ──►  Estimator
```

Train/test: stratified 80/20 with `random_state=42`. Positive rate matches the population in both halves.

## Model choice

Four candidates, all wrapped in the pipeline above. Ranked by ROC-AUC at the default 0.5 cutoff:

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.855 | 0.167 | 0.260 | **0.815** |
| Random Forest | 0.854 | 0.118 | 0.198 | 0.813 |
| Logistic Regression | 0.850 | 0.145 | 0.229 | 0.807 |
| K-Nearest Neighbors | 0.848 | 0.166 | 0.250 | 0.770 |

The deployed model is a soft `VotingClassifier` over Logistic Regression + Random Forest + Gradient Boosting. KNN was excluded for poorer calibration.

## Threshold tuning and final metrics

Threshold sweep 0.05 → 0.80 (step 0.01), pick the F1-maximiser → **0.24**.

| Metric | Value |
|---|---:|
| Accuracy | 0.795 |
| Precision (diabetic class) | 0.390 |
| Recall (diabetic class) | 0.599 |
| F1 (diabetic class) | 0.472 |
| ROC AUC | 0.814 |

## Web UI

The Flask app loads `model.pkl`, `features.pkl`, `threshold.pkl`, and `metadata.json` at startup and renders the 21 questions split into four colour-coded cards (about you · medical history · daily life · access to care). Tooltips translate BRFSS jargon into plain English. A sticky progress bar fills as the user answers, the result animates with a count-up percentage, and `prefers-reduced-motion` is honoured.

## Figures

| File | Description |
|---|---|
| `01_target_distribution.png` | Class imbalance |
| `02_bmi_distribution.png` | BMI histogram by class |
| `03_age_distribution.png` | Age band distribution by class |
| `04_diabetes_vs_highbp.png` | Diabetes rate by HighBP flag |
| `05_diabetes_vs_genhlth.png` | Diabetes rate by general health |
| `06_correlation_heatmap.png` | Feature correlation heatmap |
| `07_model_comparison.png` | Five metrics across the four candidates |
| `08_confusion_matrix.png` | Confusion matrix at default 0.5 |
| `09_roc_curve.png` | ROC curve |
| `10_precision_recall.png` | Precision-recall curve |
| `11_confusion_matrix_tuned.png` | Confusion matrix at the tuned 0.24 cutoff |

## Submission layout

```
submission/
├── README.md
├── requirements.txt
├── run.bat
│
├── notebook.ipynb
├── app.py
├── templates/index.html
│
├── data/diabetes.csv
├── figures/  (11 PNGs)
│
├── model.pkl
├── features.pkl
├── threshold.pkl
└── metadata.json
```

## Tech stack

Python 3.12, Flask 3.0, scikit-learn 1.4, pandas 2.2, numpy 1.26, joblib 1.3, matplotlib 3.8, seaborn 0.13.

## Reproducibility

`RANDOM_STATE = 42` for the split, the random forest, and the gradient boosting member of the ensemble. Paths are relative throughout.

## Limitations

- BRFSS responses are self-reported. People underreport weight, alcohol use, and smoking.
- US-only and skewed older.
- The model predicts a *survey response*, not a clinical diagnosis.
- Probabilities are not calibrated.

---

> ⚠️ Coursework artefact. Not a medical device.
