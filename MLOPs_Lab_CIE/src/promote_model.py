"""
Task 4 — Model Promotion
Trains a challenger model (random_state=99), registers as version 2,
assigns "champion" alias to the better version.
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
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENT_NAME = "biosynth-compound-yield-mg"
REGISTERED_MODEL_NAME = "biosynth-compound-yield-mg-predictor"
ALIAS_NAME = "champion"
FEATURES = ["reaction_temp_c", "catalyst_concentration", "reaction_time_hours", "ph_level"]
TARGET = "compound_yield_mg"

DEFAULT_BEST_PARAMS = {"n_estimators": 200, "max_depth": 7, "min_samples_split": 2}


def load_step3():
    path = os.path.join(RESULTS_DIR, "step3_s6.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_best_params():
    step2_path = os.path.join(RESULTS_DIR, "step2_s2.json")
    if os.path.exists(step2_path):
        with open(step2_path) as f:
            data = json.load(f)
        return data.get("best_params", DEFAULT_BEST_PARAMS)
    return DEFAULT_BEST_PARAMS


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_params = load_best_params()
    step3 = load_step3()

    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Version 1 MAE (from step3 result or re-evaluate) ──────────────────────
    if step3:
        v1_mae = step3["source_metric_value"]
        v1_version = step3["version"]
    else:
        # Fallback: train with random_state=42 and measure
        model_v1 = RandomForestRegressor(**best_params, random_state=42)
        model_v1.fit(X_train, y_train)
        v1_mae = float(mean_absolute_error(y_test, model_v1.predict(X_test)))
        v1_version = 1

    print(f"Version 1 MAE: {v1_mae:.4f}")

    # ── Train challenger (random_state=99) ────────────────────────────────────
    with mlflow.start_run(run_name="challenger-random-state-99") as run:
        mlflow.set_tag("team", "ml_engineering")
        mlflow.log_params({**best_params, "random_state": 99})

        model_v2 = RandomForestRegressor(**best_params, random_state=99)
        model_v2.fit(X_train, y_train)
        v2_mae = float(mean_absolute_error(y_test, model_v2.predict(X_test)))
        mlflow.log_metric("mae", round(v2_mae, 4))

        mlflow.sklearn.log_model(
            model_v2,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        challenger_run_id = run.info.run_id

    print(f"Version 2 MAE: {v2_mae:.4f}")

    # Fetch actual version 2 number
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    challenger_version = max(int(v.version) for v in versions)

    # ── Decide champion ───────────────────────────────────────────────────────
    if v2_mae < v1_mae:
        champion_version = challenger_version
        action = "promoted"
        print(f"Challenger (v{challenger_version}) is better → promoting to champion.")
    else:
        champion_version = v1_version
        action = "kept"
        print(f"Champion (v{v1_version}) retained — challenger is not better.")

    # Assign alias
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=ALIAS_NAME,
        version=str(champion_version),
    )
    print(f'Alias "{ALIAS_NAME}" set to version {champion_version}.')

    output = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "alias_name": ALIAS_NAME,
        "champion_version": champion_version,
        "challenger_version": challenger_version,
        "action": action,
    }

    out_path = os.path.join(RESULTS_DIR, "step4_s7.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
