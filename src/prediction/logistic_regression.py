# src/prediction/logistic_regression.py
"""
Win-probability logistic regression on features_with_odds.parquet, with and
without the market's no-vig moneyline probability as a feature.

Run:
    python -m src.prediction.logistic_regression
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.prediction.baselines import split_by_season
from src.features.feature_config import (
    PRE_GAME_FEATURES,
    ACTUAL_ABSENCE_FEATURES,
)

FEATURE_PATH = Path("data/processed/features_with_odds.parquet")
RESULTS_PATH = Path("data/results/logistic_regression_odds.json")

# Reduced split: the odds source has no 2015-16, 2019-20, or 2020-21.
ODDS_TRAIN_SEASONS = ["2016-17", "2017-18", "2018-19", "2021-22", "2022-23"]
ODDS_VAL_SEASONS = ["2023-24"]
ODDS_TEST_SEASONS = ["2024-25", "2025-26"]

C_GRID = [0.001, 0.01, 0.1, 1]

NO_ABSENCE = [f for f in PRE_GAME_FEATURES if f not in ACTUAL_ABSENCE_FEATURES]

FEATURE_SETS = {
    "market_only": ["market_logit"],
    "one_feature": ["last_10_net_rating_difference"],
    "no_actual_absence": NO_ABSENCE,
    "full": PRE_GAME_FEATURES,
    "market_plus_no_absence": ["market_logit"] + NO_ABSENCE,
    "market_plus_full": ["market_logit"] + PRE_GAME_FEATURES,
}

REFERENCE_SET = "market_only"


def add_market_logit(df: pd.DataFrame) -> pd.DataFrame:
    """log-odds of the no-vig market home win probability. Row-wise, no fitted state."""
    df = df.copy()
    p = df["home_implied_prob"]
    assert p.notna().all(), "NaN in home_implied_prob"
    assert ((p > 0) & (p < 1)).all(), "home_implied_prob outside (0, 1)"
    df["market_logit"] = np.log(p / (1 - p))
    return df


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
        "log_loss": float(log_loss(y, probs)),
    }


def raw_market_scores(split_df: pd.DataFrame) -> dict:
    """Scores of the market's no-vig probability itself, with no model fitted."""
    y = split_df["home_win"].astype(int)
    p = split_df["home_implied_prob"]
    return {
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }


def tune_C(train: pd.DataFrame, val: pd.DataFrame, features: list, c_grid: list = C_GRID) -> float:
    """Pick C by val log loss. Never receives test."""
    X_train, y_train = get_X_y(train, features)
    X_val, y_val = get_X_y(val, features)
    scores = {}
    for C in c_grid:
        model = build_model(C)
        model.fit(X_train, y_train)
        metrics = score_model(model, X_val, y_val)
        scores[C] = metrics["log_loss"]
        print(f"  C={C}: val log loss {metrics['log_loss']:.4f}")
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
        "test": score_model(model, X_test, y_test),
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


def print_comparison(results: dict, market_raw: dict) -> None:
    """Accuracy / log loss / Brier per feature set, deltas vs. the reference set,
    and the raw (unfitted) market as a final row."""
    ref = results[REFERENCE_SET]
    header = (f"{'feature set':<24}{'n':>4}{'C':>8}"
              f"{'acc tr':>8}{'acc va':>8}{'acc te':>8}"
              f"{'ll va':>8}{'ll te':>8}{'br va':>8}{'br te':>8}"
              f"{'d ll va':>9}{'d ll te':>9}")
    print(header)
    print("-" * len(header))

    for name, r in results.items():
        d_va = r["val"]["log_loss"] - ref["val"]["log_loss"]
        d_te = r["test"]["log_loss"] - ref["test"]["log_loss"]
        print(f"{name:<24}{r['n_features']:>4}{r['best_C']:>8}"
              f"{r['train']['accuracy']:>8.3f}{r['val']['accuracy']:>8.3f}{r['test']['accuracy']:>8.3f}"
              f"{r['val']['log_loss']:>8.3f}{r['test']['log_loss']:>8.3f}"
              f"{r['val']['brier']:>8.3f}{r['test']['brier']:>8.3f}"
              f"{d_va:>+9.4f}{d_te:>+9.4f}")

    print("-" * len(header))
    mv, mt = market_raw["val"], market_raw["test"]
    print(f"{'market raw (no fit)':<24}{'-':>4}{'-':>8}"
          f"{'-':>8}{mv['accuracy']:>8.3f}{mt['accuracy']:>8.3f}"
          f"{mv['log_loss']:>8.3f}{mt['log_loss']:>8.3f}"
          f"{mv['brier']:>8.3f}{mt['brier']:>8.3f}"
          f"{mv['log_loss'] - ref['val']['log_loss']:>+9.4f}"
          f"{mt['log_loss'] - ref['test']['log_loss']:>+9.4f}")
    print(f"\nd ll = log loss minus the '{REFERENCE_SET}' row (negative = better than it).")


if __name__ == "__main__":
    df = pd.read_parquet(FEATURE_PATH)
    df = add_market_logit(df)
    train, val, test = split_by_season(
        df, ODDS_TRAIN_SEASONS, ODDS_VAL_SEASONS, ODDS_TEST_SEASONS
    )
    print(f"split sizes: train {len(train)}, val {len(val)}, test {len(test)}")

    # Sanity check 1: no NaNs in any split, for every column any feature set uses
    all_cols = sorted({c for feats in FEATURE_SETS.values() for c in feats})
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        X, _ = get_X_y(split_df, all_cols)
        n_nan = int(X.isna().sum().sum())
        assert n_nan == 0, f"{split_name} has {n_nan} NaNs in feature columns"

    # Sanity check 2: the absence ablation actually removes features
    assert len(FEATURE_SETS["full"]) > len(FEATURE_SETS["no_actual_absence"]), (
        "no_actual_absence removed nothing; check ACTUAL_ABSENCE_FEATURES"
    )

    results = {}
    for name, features in FEATURE_SETS.items():
        print(f"\nRunning {name} ({len(features)} features)...")
        results[name] = run_experiment(train, val, test, features)

    market_raw = {"val": raw_market_scores(val), "test": raw_market_scores(test)}
    results["market_raw"] = market_raw  # saved alongside; excluded from the table loop below

    save_results(results)
    results_for_table = {k: v for k, v in results.items() if k != "market_raw"}
    print()
    print_comparison(results_for_table, market_raw)

    # Refit market_plus_full once to look at coefficients
    feats = FEATURE_SETS["market_plus_full"]
    X_train, y_train = get_X_y(train, feats)
    model = build_model(results["market_plus_full"]["best_C"])
    model.fit(X_train, y_train)
    print("\nTop coefficients (market_plus_full):")
    print(inspect_coefficients(model, feats).to_string())