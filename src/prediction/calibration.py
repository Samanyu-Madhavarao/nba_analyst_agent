import pandas as pd
import numpy as np

from src.prediction.baselines import split_by_season
from src.prediction.logistic_regression import (
    FEATURE_PATH, ODDS_TRAIN_SEASONS, ODDS_VAL_SEASONS, ODDS_TEST_SEASONS,
)

def calibration_table(y, p, n_bins=10):
    assert len(y) == len(p), "y an p must be the same length"
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    assert not df.isna().any().any(), "NaNs in y or p"

    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")

    table = df.groupby("bin", observed=True).agg(
        n=("y", "size"),
        mean_pred=("p", "mean"),
        actual_rate=("y", "mean"),
    )

    table["gap"] = table["actual_rate"] - table["mean_pred"]
    table["std_err"] = np.sqrt(table["mean_pred"] * (1 - table["mean_pred"]) / table["n"])
    table["flag"] = table["gap"].abs() > 2 * table["std_err"]

    return table.reset_index()

if __name__ == "__main__":
    df = pd.read_parquet(FEATURE_PATH)
    train, val, test = split_by_season(
        df, ODDS_TRAIN_SEASONS, ODDS_VAL_SEASONS, ODDS_TEST_SEASONS
    )

    for name, split in [("val", val), ("test", test)]:
        table = calibration_table(split["home_win"], split["home_implied_prob"])
        ece = (table["n"] * table["gap"].abs()).sum() / table["n"].sum()
        print(f"\n{name} (n={len(split)}), ECE = {ece:.4f}")
        print(table.to_string())
