from pathlib import Path
import pandas as pd
import numpy as np
import json

FEATURES_PATH = Path("data/processed/features.parquet")
RESULTS_PATH = Path("data/results/baselines.json")

TRAIN_SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"
]
VAL_SEASONS = ["2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

def split_by_season(df, train_seasons, val_seasons, test_seasons):
    train = df[df["season"].isin(train_seasons)].copy()
    val   = df[df["season"].isin(val_seasons)].copy()
    test  = df[df["season"].isin(test_seasons)].copy()

    assert set(train["season"]).isdisjoint(val["season"])
    assert set(train["season"]).isdisjoint(test["season"])
    assert set(val["season"]).isdisjoint(test["season"])
    assert len(train) + len(val) + len(test) == len(df), "some rows unaccounted for"

    return train, val, test

def constant_margin_baseline(train, val, test):
    home_margin_mean = train["actual_margin"].mean()
    print(home_margin_mean)
    results = {}

    for name, split in [("train", train), ("val", val), ("test", test)]:
        margin = split["actual_margin"]
        results[f"{name}_MAE"] = (margin - home_margin_mean).abs().mean()
        results[f"{name}_RMSE"] = ((margin - home_margin_mean) ** 2).mean() ** 0.5
        results[f"{name}_win_accuracy"] = (margin > 0).mean()

    return results

def constant_win_probability(train, val, test):
    home_win_rate = train["home_win"].mean()
    print(home_win_rate)
    results = {}

    for name, split in [("train", train), ("val", val), ("test", test)]:
        home_win = split["home_win"]
        results[f"{name}_brier_score"] = ((home_win_rate - home_win) ** 2).mean()
        results[f"{name}_log_loss"] = -(split["home_win"] * np.log(home_win_rate) + (1 - split["home_win"]) * np.log(1 - home_win_rate)).mean()

    return results

def one_feature_baseline(train, val, test, feature="last_10_net_rating_difference"):
    slope, intercept = np.polyfit(train[feature], train["actual_margin"], 1)
    print(slope)
    print(intercept)
    results = {}

    for name, split in [("train", train), ("val", val), ("test", test)]:
        predicted_margin = slope * split[feature] + intercept
        actual_margin = split["actual_margin"]

        results[f"{name}_MAE"] = (actual_margin - predicted_margin).abs().mean()
        results[f"{name}_RMSE"] = ((actual_margin - predicted_margin) ** 2).mean() ** 0.5
        results[f"{name}_win_accuracy"] = ((predicted_margin > 0) == (actual_margin > 0)).mean()

    return results

def save_results(results: dict, path=RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        baseline_name: {k: float(v) for k, v, in metrics.items()}
        for baseline_name, metrics in results.items()
    }

    with open(path, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"saved results to {path}")

if __name__ == "__main__":
    df = pd.read_parquet(FEATURES_PATH)
    train, val, test = split_by_season(df, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS)
    all_results = {
        "constant_margin": constant_margin_baseline(train, val, test),\
        "constant_win_probability": constant_win_probability(train, val, test),
        "one_feature_net_rating": one_feature_baseline(train, val, test)
    }
    save_results(all_results)