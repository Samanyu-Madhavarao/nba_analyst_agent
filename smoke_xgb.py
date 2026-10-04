import pandas as pd

from src.prediction.baselines import split_by_season
from src.prediction.logistic_regression import (
    get_X_y, score_model, add_market_logit, FEATURE_SETS,
    ODDS_TRAIN_SEASONS, ODDS_VAL_SEASONS, ODDS_TEST_SEASONS, FEATURE_PATH,
)
from src.prediction.xgboost_model import (build_model, fit_model, 
                tune_params, run_experiment, paired_logloss_diff)

df = add_market_logit(pd.read_parquet(FEATURE_PATH))
train, val, test = split_by_season(
    df, ODDS_TRAIN_SEASONS, ODDS_VAL_SEASONS, ODDS_TEST_SEASONS
)
print(f"split sizes: train {len(train)}, val {len(val)}, test {len(test)}")

features = FEATURE_SETS["no_actual_absence"]
X_train, y_train = get_X_y(train, features)
X_val, y_val = get_X_y(val, features)

params = {"max_depth": 2, "min_child_weight": 20, "reg_lambda": 1}
model = fit_model(build_model(params), X_train, y_train, X_val, y_val)

"""
print("best_iteration:", model.best_iteration)
print("val scores:", score_model(model, X_val, y_val))
print("train scores:", score_model(model, X_train, y_train))

best_params, best_iteration = tune_params(train, val, FEATURE_SETS["no_actual_absence"])
print("best:", best_params, best_iteration)
"""

result, model = run_experiment(train, val, test, FEATURE_SETS["no_actual_absence"])
print(result)

features = FEATURE_SETS["no_actual_absence"]
X_test, y_test = get_X_y(test, features)
print(paired_logloss_diff(model, X_test, y_test, test["home_implied_prob"]))