"""Optunaによるハイパーパラメータ最適化"""

import optuna
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import get_feature_columns
from config import TRAIN_RATIO, OPTUNA_N_TRIALS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_lgbm(df: pd.DataFrame, n_trials: int = OPTUNA_N_TRIALS) -> dict:
    """LightGBMのパラメータを最適化"""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["Target"]
    split_idx = int(len(df) * TRAIN_RATIO)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.log_evaluation(0)])
        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n=== LightGBM 最適化結果 ===")
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")
    return {**study.best_params, "objective": "binary", "metric": "auc", "verbosity": -1}


def optimize_xgb(df: pd.DataFrame, n_trials: int = OPTUNA_N_TRIALS) -> dict:
    """XGBoostのパラメータを最適化"""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["Target"]
    split_idx = int(len(df) * TRAIN_RATIO)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n=== XGBoost 最適化結果 ===")
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")
    return {**study.best_params, "objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0}


def optimize_hold_days(df_raw: pd.DataFrame, days_range: list[int] = None) -> int:
    """最適な保有日数を探索"""
    from features import prepare_features
    if days_range is None:
        days_range = [3, 5, 7, 10, 15, 20]

    best_auc = 0
    best_days = 5

    for days in days_range:
        import config
        original = config.HOLD_DAYS
        config.HOLD_DAYS = days

        df = prepare_features(df_raw, include_fundamentals=False,
                              include_sentiment=False, include_market=False)
        feature_cols = get_feature_columns(df)
        X = df[feature_cols]
        y = df["Target"]
        split_idx = int(len(df) * TRAIN_RATIO)

        model = lgb.LGBMClassifier(**{
            "objective": "binary", "metric": "auc", "verbosity": -1,
            "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
        })
        model.fit(X.iloc[:split_idx], y.iloc[:split_idx])
        y_prob = model.predict_proba(X.iloc[split_idx:])[:, 1]
        auc = roc_auc_score(y.iloc[split_idx:], y_prob)

        print(f"  保有日数 {days:>2d}日: AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_days = days

        config.HOLD_DAYS = original

    print(f"\n  最適保有日数: {best_days}日 (AUC={best_auc:.4f})")
    return best_days
