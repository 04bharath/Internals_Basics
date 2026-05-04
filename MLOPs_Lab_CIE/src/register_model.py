"""
Task 3 — Model Versioning
Trains the best-tuned RandomForest, registers it in the MLflow Model Registry,
and records version number + run_id.
"""

import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

EXPERIMENT_NAME = "biosynth-compound-yield-mg"
REGISTERED_MODEL_NAME = "biosynth-compound-yield-mg-predictor"
FEATURES = ["reaction_temp_c", "catalyst_concentration", "reaction_time_hours", "ph_level"]
TARGET = "compound_yield_mg"

# Best params from Task 2 — will be overridden if step2 result exists
DEFAULT_BEST_PARAMS = {"n_estimators": 200, "max_depth": 7, "min_samples_split": 2}


def load_best_params():
    step2_path = os.path.join(RESULTS_DIR, "step2_s2.json")
    if os.path.exists(step2_path):
        with open(step2_path) as f:
            data = json.load(f)
        params = data.get("best_params", DEFAULT_BEST_PARAMS)
        print(f"Loaded best params from step2: {params}")
        return params
    print("step2_s2.json not found — using default params.")
    return DEFAULT_BEST_PARAMS


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_params = load_best_params()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="register-best-model") as run:
        mlflow.set_tag("team", "ml_engineering")
        mlflow.log_params(best_params)

        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("mae", round(mae, 4))

        # Log + register in one call
        model_uri = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        ).model_uri

        run_id = run.info.run_id
        print(f"Run ID: {run_id} | MAE: {mae:.4f}")

    # Fetch version number
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    version_number = max(int(v.version) for v in versions)

    output = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version": version_number,
        "run_id": run_id,
        "source_metric": "mae",
        "source_metric_value": round(mae, 4),
    }

    out_path = os.path.join(RESULTS_DIR, "step3_s6.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
