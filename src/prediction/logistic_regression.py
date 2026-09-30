# src/prediction/logistic_regression.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.prediction.baselines import split_by_season, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS
from src.features.feature_config import (
    PRE_GAME_FEATURES,
    EXPECTED_ROSTER_FEATURES,
    ACTUAL_ABSENCE_FEATURES,
)

FEATURE_PATH = Path("data/processed/features.parquet")
RESULTS_PATH = Path("data/results/logistic_regression.json")

C_GRID = [0.001, 0.01, 0.1, 1]

# Ablation feature sets, built by filtering PRE_GAME_FEATURES so the
# config stays the single source of truth.
FEATURE_SETS = {
    "one_feature": ["last_10_net_rating_difference"],
    "full": PRE_GAME_FEATURES,
    "no_actual_absence": [
        f for f in PRE_GAME_FEATURES if f not in ACTUAL_ABSENCE_FEATURES
    ],
    "no_injury_production": [
        f for f in PRE_GAME_FEATURES
        if f not in ACTUAL_ABSENCE_FEATURES and f not in EXPECTED_ROSTER_FEATURES
    ],
}

BASELINE_REFERENCE = {
    "one_feature_win_acc": {"train": 0.623, "val": 0.628, "test": 0.652},
    "constant_prob_log_loss": {"train": 0.683, "val": 0.691, "test": 0.689},
    "constant_prob_brier": {"train": 0.245, "val": 0.249, "test": 0.248},
}

def get_X_y(split_df: pd.DataFrame, features: list):
    X = split_df[features].astype(float)
    y = split_df["home_win"].astype(int)
    return (X, y)


def build_model(C: float) -> Pipeline:
     return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=C, max_iter=1000)),
    ])


def score_model(model, X, y) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "brier": float(brier_score_loss(y, probs)),
        "log_loss": float(log_loss(y, probs))
    }


def tune_C(train: pd.DataFrame, val: pd.DataFrame, features: list, c_grid: list = C_GRID) -> float:
    X_train, y_train = get_X_y(train, features)
    X_val, y_val = get_X_y(val, features)
    scores = {}
    for C in c_grid:
        model = build_model(C)
        model.fit(X_train, y_train)
        metrics = score_model(model, X_val, y_val)
        scores[C] = metrics["log_loss"]
        print(f"{C}, {metrics['log_loss']}")
    return min(scores, key=scores.get)
        


def run_experiment(train, val, test, features: list) -> dict:
    best_C = tune_C(train, val, features)
    X_train, y_train = get_X_y(train, features)
    X_val, y_val = get_X_y(val, features)
    X_test, y_test = get_X_y(test, features)
    model = build_model(best_C)
    model.fit(X_train, y_train)
    return {
        "best_C": best_C,
        "n_features": len(features),
        "train": score_model(model, X_train, y_train),
        "val": score_model(model, X_val, y_val),
        "test": score_model(model, X_test, y_test)
    }


def inspect_coefficients(model, features: list, top_n: int = 15) -> pd.DataFrame:
    """Top-n standardized coefficients (by absolute size) from a fitted pipeline."""
    coefs = model.named_steps["logreg"].coef_[0]
    df = pd.DataFrame({"feature": features, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def save_results(results: dict, path: Path = RESULTS_PATH) -> None:
    """Write the results dict to JSON, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def print_comparison(results: dict) -> None:
    """Print accuracy / log loss / Brier per feature set and split, plus baselines."""
    header = (f"{'feature set':<22}{'n':>4}{'C':>8}"
              f"{'acc tr':>8}{'acc va':>8}{'acc te':>8}"
              f"{'ll va':>8}{'ll te':>8}{'br va':>8}{'br te':>8}")
    print(header)
    print("-" * len(header))

    for name, r in results.items():
        print(f"{name:<22}{r['n_features']:>4}{r['best_C']:>8}"
              f"{r['train']['accuracy']:>8.3f}{r['val']['accuracy']:>8.3f}{r['test']['accuracy']:>8.3f}"
              f"{r['val']['log_loss']:>8.3f}{r['test']['log_loss']:>8.3f}"
              f"{r['val']['brier']:>8.3f}{r['test']['brier']:>8.3f}")

    print("-" * len(header))
    b = BASELINE_REFERENCE
    acc = b["one_feature_win_acc"]
    ll = b["constant_prob_log_loss"]
    br = b["constant_prob_brier"]
    print(f"{'one-feature baseline':<22}{1:>4}{'-':>8}"
          f"{acc['train']:>8.3f}{acc['val']:>8.3f}{acc['test']:>8.3f}"
          f"{'-':>8}{'-':>8}{'-':>8}{'-':>8}")
    print(f"{'constant probability':<22}{0:>4}{'-':>8}"
          f"{'-':>8}{'-':>8}{'-':>8}"
          f"{ll['val']:>8.3f}{ll['test']:>8.3f}{br['val']:>8.3f}{br['test']:>8.3f}")


if __name__ == "__main__":
    df = pd.read_parquet(FEATURE_PATH)
    train, val, test = split_by_season(df, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS)

    # Sanity check 1: no NaNs in any split for the full feature set
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        X, _ = get_X_y(split_df, PRE_GAME_FEATURES)
        n_nan = int(X.isna().sum().sum())
        assert n_nan == 0, f"{split_name} has {n_nan} NaNs in PRE_GAME_FEATURES"

    # Sanity check 2: each ablation actually removes features
    n_full = len(FEATURE_SETS["full"])
    n_no_abs = len(FEATURE_SETS["no_actual_absence"])
    n_no_inj = len(FEATURE_SETS["no_injury_production"])
    assert n_full > n_no_abs > n_no_inj, (
        f"Ablation sizes not strictly decreasing: {n_full}, {n_no_abs}, {n_no_inj}"
    )

    results = {}
    for name, features in FEATURE_SETS.items():
        print(f"\nRunning {name} ({len(features)} features)...")
        results[name] = run_experiment(train, val, test, features)

    save_results(results)
    print()
    print_comparison(results)

    # Refit the full model once to look at coefficients
    X_train, y_train = get_X_y(train, PRE_GAME_FEATURES)
    model = build_model(results["full"]["best_C"])
    model.fit(X_train, y_train)
    print("\nTop coefficients (full feature set):")
    print(inspect_coefficients(model, PRE_GAME_FEATURES).to_string())