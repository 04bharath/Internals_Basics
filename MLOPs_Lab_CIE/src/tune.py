"""
Task 2 — Hyperparameter Tuning
Random search with 3-fold CV on RandomForestRegressor (best model class from Task 1).
Logs each trial as a nested run under parent "tuning-biosynth".
"""

import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, ParameterSampler
from sklearn.metrics import mean_absolute_error

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENT_NAME = "biosynth-compound-yield-mg"
FEATURES = ["reaction_temp_c", "catalyst_concentration", "reaction_time_hours", "ph_level"]
TARGET = "compound_yield_mg"

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 7, 15],
    "min_samples_split": [2, 4],
}
N_ITER = 9          # all combinations that fit random search budget
N_FOLDS = 3
PARENT_RUN_NAME = "tuning-biosynth"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    param_list = list(ParameterSampler(PARAM_GRID, n_iter=N_ITER, random_state=42))
    total_trials = len(param_list)

    best_cv_mae = float("inf")
    best_params = {}
    best_test_mae = float("inf")

    with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent_run:
        mlflow.set_tag("team", "ml_engineering")

        for i, params in enumerate(param_list):
            with mlflow.start_run(run_name=f"trial_{i+1}", nested=True):
                mlflow.log_params(params)

                model = RandomForestRegressor(**params, random_state=42)

                # 3-fold CV MAE (negated scorer → flip sign)
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=N_FOLDS,
                    scoring="neg_mean_absolute_error"
                )
                cv_mae = float(-cv_scores.mean())

                # Also fit on full train, eval on test
                model.fit(X_train, y_train)
                test_mae = float(mean_absolute_error(y_test, model.predict(X_test)))

                mlflow.log_metric("cv_mae", round(cv_mae, 4))
                mlflow.log_metric("test_mae", round(test_mae, 4))
                mlflow.sklearn.log_model(model, artifact_path=f"trial_{i+1}")

                print(f"Trial {i+1} | params={params} | cv_mae={cv_mae:.4f} | test_mae={test_mae:.4f}")

                if cv_mae < best_cv_mae:
                    best_cv_mae = cv_mae
                    best_params = params
                    best_test_mae = test_mae

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_mae", round(best_cv_mae, 4))
        mlflow.log_metric("best_test_mae", round(best_test_mae, 4))

    output = {
        "search_type": "random",
        "n_folds": N_FOLDS,
        "total_trials": total_trials,
        "best_params": best_params,
        "best_mae": round(best_test_mae, 4),
        "best_cv_mae": round(best_cv_mae, 4),
        "parent_run_name": PARENT_RUN_NAME,
    }

    out_path = os.path.join(RESULTS_DIR, "step2_s2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
