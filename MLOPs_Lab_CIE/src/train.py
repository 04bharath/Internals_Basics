"""
Task 1 — Experiment Tracking & Model Comparison
Trains LinearRegression and Ridge, logs to MLflow, selects best by MAE.
"""

import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

EXPERIMENT_NAME = "biosynth-compound-yield-mg"
FEATURES = ["reaction_temp_c", "catalyst_concentration", "reaction_time_hours", "ph_level"]
TARGET = "compound_yield_mg"


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    # MAPE — guard against zero
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 4), "mape": round(mape, 4)}


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    models_cfg = [
        ("LinearRegression", LinearRegression(), {}),
        ("Ridge", Ridge(alpha=1.0), {"alpha": 1.0}),
    ]

    results_models = []

    for name, model, params in models_cfg:
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("team", "ml_engineering")

            # Log params
            if params:
                mlflow.log_params(params)
            else:
                mlflow.log_param("fit_intercept", True)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path=name)

            results_models.append({"name": name, **metrics})
            print(f"{name}: {metrics}")

    # Best by MAE (lower is better)
    best = min(results_models, key=lambda m: m["mae"])

    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": results_models,
        "best_model": best["name"],
        "best_metric_name": "mae",
        "best_metric_value": best["mae"],
    }

    out_path = os.path.join(RESULTS_DIR, "step1_s1.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
