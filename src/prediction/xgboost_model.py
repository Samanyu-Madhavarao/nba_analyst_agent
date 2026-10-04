# src/prediction/xgboost_model.py
"""
XGBoost win-probability model on features_with_odds.parquet.
Same reduced split as the logistic run; compared against market_only.

Run:
    python -m src.prediction.xgboost_model
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.prediction.baselines import split_by_season
from src.prediction.logistic_regression import (
    get_X_y, score_model, raw_market_scores, add_market_logit,
    FEATURE_SETS, ODDS_TRAIN_SEASONS, ODDS_VAL_SEASONS, ODDS_TEST_SEASONS,
    FEATURE_PATH,
)

RESULTS_PATH = Path("data/results/xgboost_odds.json")
LOGREG_RESULTS_PATH = Path("data/results/logistic_regression_odds.json")

N_ESTIMATORS = 1000
LEARNING_RATE = 0.03
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42

# Feature sets for this run (market_only / one_feature skipped on purpose)
XGB_FEATURE_SETS = ["no_actual_absence", "full",
                    "market_plus_no_absence", "market_plus_full"]

# Small grid: 2 x 2 x 2 = 8 configs
PARAM_GRID = {
    "max_depth": [2, 3],
    "min_child_weight": [20, 50],
    "reg_lambda": [1, 10],
}
FIXED_PARAMS = {"subsample": 0.8, "colsample_bytree": 0.7}


def build_model(params: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                             eval_metric="logloss", early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                             random_state=RANDOM_STATE, **FIXED_PARAMS, **params)


def fit_model(model, X_train, y_train, X_val, y_val):
    return model.fit( X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


def iter_param_grid(grid: dict):
    """Yield one params dict per combination (itertools.product over the grid)."""
    keys = grid.keys()
    combos = itertools.product(*grid.values())
    for combo in combos:
        yield dict(zip(keys, combo))


def tune_params(train, val, features: list) -> tuple:
    X_train, y_train = get_X_y(train, features)
    X_val, y_val = get_X_y(val, features)
    best_loss = float("inf")
    best_params = None
    best_iteration = None
    for params in iter_param_grid(PARAM_GRID):
        model = build_model(params)
        model = fit_model(model, X_train, y_train, X_val, y_val)
        val_loss = score_model(model, X_val, y_val)["log_loss"]
        print(f"params: {params}, val_loss: {val_loss:.4f}, best iteration: {model.best_iteration}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params
            best_iteration = model.best_iteration
    return best_params, best_iteration


def run_experiment(train, val, test, features: list) -> tuple:
    """Tune, refit the best config on train (val still used for early stopping),
    then score train/val/test with the shared score_model()."""
    best_params, _ = tune_params(train, val, features)
    X_train, y_train = get_X_y(train, features)
    X_val, y_val = get_X_y(val, features)
    X_test, y_test = get_X_y(test, features)
    model = build_model(best_params)
    model = fit_model(model, X_train, y_train, X_val, y_val)
    result_dict =  {
        "n_features": len(features),
        "best_params": best_params,
        "best_iteration": int(model.best_iteration),
        "train": score_model(model, X_train, y_train),
        "val": score_model(model, X_val, y_val),
        "test": score_model(model, X_test, y_test),
    }
    return result_dict, model


def paired_logloss_diff(model, X, y, market_p) -> dict:
    """Per-game log loss of model minus per-game log loss of the market.
    Negative mean_diff means the model beat the market."""
    eps = 1e-6
    y_arr = np.asarray(y, dtype=float)
    p_model = np.clip(model.predict_proba(X)[:, 1], eps, 1 - eps)
    p_market = np.clip(np.asarray(market_p, dtype=float), eps, 1 - eps)

    model_ll = -(y_arr * np.log(p_model) + (1 - y_arr) * np.log(1 - p_model))
    market_ll = -(y_arr * np.log(p_market) + (1 - y_arr) * np.log(1 - p_market))

    diff = model_ll - market_ll
    n = len(diff)

    return {
        "mean_diff": float(diff.mean()),
        "std_err": float(diff.std(ddof=1) / np.sqrt(n)),
        "n": int(n),
    }


def top_importances(model, features: list, top_n: int = 15) -> pd.DataFrame:
    """Gain-based importances, sorted. Features never split on get gain 0."""
    gain = model.get_booster().get_score(importance_type="gain")
    df = pd.DataFrame({
        "feature": features,
        "gain": [gain.get(f, 0.0) for f in features],
    })
    df = df.sort_values("gain", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def save_results(results: dict, path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def print_comparison(results: dict, logreg_results: dict) -> None:
    """XGBoost vs logistic per feature set, with deltas vs the logistic market_only."""
    ref_va = logreg_results["market_only"]["val"]["log_loss"]
    ref_te = logreg_results["market_only"]["test"]["log_loss"]

    header = (f"{'feature set':<24}{'n':>4}{'iters':>7}"
              f"{'xgb va':>9}{'xgb te':>9}{'lr va':>9}{'lr te':>9}"
              f"{'d va':>9}{'d te':>9}")
    print(header)
    print("-" * len(header))

    for name, r in results.items():
        lr = logreg_results[name]
        xva, xte = r["val"]["log_loss"], r["test"]["log_loss"]
        print(f"{name:<24}{r['n_features']:>4}{r['best_iteration']:>7}"
              f"{xva:>9.4f}{xte:>9.4f}"
              f"{lr['val']['log_loss']:>9.4f}{lr['test']['log_loss']:>9.4f}"
              f"{xva - ref_va:>+9.4f}{xte - ref_te:>+9.4f}")

    print("-" * len(header))
    print(f"market_only reference: val {ref_va:.4f}, test {ref_te:.4f}")
    print("d = XGBoost log loss minus logistic market_only (negative = better than the market).")

    print("\nPaired test log loss vs raw market (model minus market, per game):")
    for name, r in results.items():
        p = r.get("paired_vs_market_test")
        if p:
            z = p["mean_diff"] / p["std_err"]
            print(f"  {name:<24} mean_diff {p['mean_diff']:+.4f}  "
                  f"std_err {p['std_err']:.4f}  ({z:+.1f} SE)  n={p['n']}")


if __name__ == "__main__":
    # 1. Load features_with_odds, add_market_logit
    # 2. split_by_season with the ODDS_* season lists, print split sizes
    # 3. NaN sanity check on every column used by XGB_FEATURE_SETS
    #    (XGBoost tolerates NaN, but you want to know, since logistic didn't)
    # 4. Loop XGB_FEATURE_SETS: run_experiment, store results
    # 5. Paired log loss diff vs market on test for market_plus_* sets
    # 6. Load logistic JSON, print_comparison, save_results
    # 7. Top importances for market_plus_full
    df = add_market_logit(pd.read_parquet(FEATURE_PATH))
    train, val, test = split_by_season(df, ODDS_TRAIN_SEASONS, ODDS_VAL_SEASONS, ODDS_TEST_SEASONS)
    print(f"train size: {len(train)}, val size: {len(val)}, test size: {len(test)}")

    all_cols = sorted({c for name in XGB_FEATURE_SETS for c in FEATURE_SETS[name]})
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        X, _ = get_X_y(split_df, all_cols)
        n_nan = int(X.isna().sum().sum())
        print(f"{split_name}: {n_nan} NaNs in feature columns")

    results = {}
    models = {}
    for name in XGB_FEATURE_SETS:
        features = FEATURE_SETS[name]
        print(f"\nRunning {name} ({len(features)} features)...")
        result, model = run_experiment(train, val, test, features)

        if name.startswith("market_plus"):
            X_test, y_test = get_X_y(test, features)
            result["paired_vs_market_test"] = paired_logloss_diff(
                model, X_test, y_test, test["home_implied_prob"]
            )

        results[name] = result
        models[name] = model

    with open(LOGREG_RESULTS_PATH) as f:
        logreg_results = json.load(f)

    print()
    print_comparison(results, logreg_results)
    save_results(results)
    print(f"\nsaved results to {RESULTS_PATH}")

    print("\nTop gain importances (market_plus_full):")
    print(top_importances(models["market_plus_full"], FEATURE_SETS["market_plus_full"]).to_string())
    