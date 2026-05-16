"""Flask backend for the Diabetes Risk Predictor.

The notebook saves a single sklearn ``Pipeline`` (StandardScaler + SelectKBest +
VotingClassifier) as ``model.pkl``. This file just feeds form data into that
pipeline in the right column order and applies the tuned probability threshold.
"""

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load artifacts produced by notebook.ipynb / build_diabetes_predictor_project.py
# ---------------------------------------------------------------------------
model = joblib.load("model.pkl")          # full Pipeline (scaler + selector + voter)
features = joblib.load("features.pkl")    # ordered input columns
threshold = joblib.load("threshold.pkl")  # tuned probability cutoff


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read all raw form fields. The two engineered features are computed
        # below from the raw inputs so the form stays simple.
        raw_columns = [c for c in features if c not in ("HealthyHabits", "RiskFactors")]
        data = {col: float(request.form[col]) for col in raw_columns}

        # Engineered features (must match the recipe used during training).
        data["HealthyHabits"] = (
            data["PhysActivity"] + data["Fruits"] + data["Veggies"]
        )
        data["RiskFactors"] = (
            data["HighBP"]
            + data["HighChol"]
            + data["Stroke"]
            + data["HeartDiseaseorAttack"]
        )

        # Build a one-row dataframe in the exact column order the pipeline
        # was fit on. The Pipeline handles scaling and feature selection.
        row = pd.DataFrame([data])[features]

        probability = float(model.predict_proba(row)[0, 1])
        prediction = 1 if probability >= threshold else 0

        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        risk = "high" if prediction == 1 else "low"
        prob_pct = round(probability * 100, 1)

        return render_template(
            "index.html",
            prediction=result,
            probability=prob_pct,
            risk=risk,
            form_data=request.form,
        )

    except Exception as exc:  # pragma: no cover - surfaced in the UI
        return render_template("index.html", error=str(exc))


if __name__ == "__main__":
    app.run(debug=True)
