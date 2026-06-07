"""Time-based holdout validation for the FanPulse MMM.

Trains on the 2019-2023 seasons and evaluates on the held-out 2024 season,
so the reported fit metrics reflect genuine out-of-sample performance rather
than in-sample overfit. Writes the results to ``validation_metrics.json`` so
the dashboard can display reproducible numbers instead of hardcoded claims.

Run with:
    python -m backend.models.validate
"""
import json
import os

import arviz as az
import numpy as np

from backend.models import mmm
from backend.models.predict import (
    posterior_means,
    compute_mu,
    regression_metrics,
)

TRAIN_SEASONS = [2019, 2021, 2022, 2023]
TEST_SEASON = 2024
METRICS_PATH = "backend/data/processed/validation_metrics.json"


def run_validation(draws=1000, tune=2000, chains=2):
    print("loading data...")
    df = mmm.load_data()

    train_raw = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test_raw = df[df["season"] == TEST_SEASON].copy()
    print(f"train rows: {len(train_raw)}  test rows: {len(test_raw)}")

    # fit scalers + transforms on train only, then reuse on test (no leakage)
    train, scalers = mmm.preprocess(train_raw)
    test = mmm.transform_with_scalers(test_raw, scalers)

    print(f"fitting model on {len(train)} training games (this takes a while)...")
    model = mmm.build_model(train)
    with model:
        trace = mmm.pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=0.92,
            return_inferencedata=True,
            progressbar=True,
        )

    params = posterior_means(trace)

    train_pred = np.exp(compute_mu(train, params))
    test_pred = np.exp(compute_mu(test, params))

    train_metrics = regression_metrics(train["attendance"].values, train_pred)
    holdout_metrics = regression_metrics(test["attendance"].values, test_pred)

    summary = az.summary(trace)
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    divergences = int(trace.sample_stats["diverging"].values.sum())

    result = {
        "train_seasons": TRAIN_SEASONS,
        "test_season": TEST_SEASON,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train": train_metrics,
        "holdout": holdout_metrics,
        "convergence": {
            "max_rhat": max_rhat,
            "min_ess_bulk": min_ess,
            "divergences": divergences,
        },
        "sampling": {"draws": draws, "tune": tune, "chains": chains},
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== holdout validation results ===")
    print(json.dumps(result, indent=2))
    print(f"\nsaved metrics to {METRICS_PATH}")
    return result


if __name__ == "__main__":
    run_validation()
