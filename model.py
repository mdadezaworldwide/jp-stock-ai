"""LightGBMによる売買シグナル予測モデル"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from config import LGBM_PARAMS, TRAIN_RATIO
from features import get_feature_columns


MODEL_PATH = Path(__file__).parent / "trained_model.pkl"


def train_model(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """モデルを訓練して返す"""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["Target"]

    # 時系列で分割（未来のデータでリーク防止）
    split_idx = int(len(df) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"訓練データ: {len(X_train)} 行, テストデータ: {len(X_test)} 行")
    print(f"正例比率 - 訓練: {y_train.mean():.3f}, テスト: {y_test.mean():.3f}")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(50)],
    )

    # 評価
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== テストデータ評価 ===")
    print(f"  AUC:       {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")

    return model


def save_model(model: lgb.LGBMClassifier, path: Path = MODEL_PATH):
    """モデルを保存"""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"モデル保存: {path}")


def load_model(path: Path = MODEL_PATH) -> lgb.LGBMClassifier:
    """モデルを読み込み"""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_signals(model: lgb.LGBMClassifier, df: pd.DataFrame) -> pd.DataFrame:
    """最新データに対してシグナルを予測"""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    df = df.copy()
    df["Signal_prob"] = model.predict_proba(X)[:, 1]
    df["Signal"] = model.predict(X)
    return df


def show_feature_importance(model: lgb.LGBMClassifier, df: pd.DataFrame, top_n: int = 20):
    """特徴量の重要度を表示"""
    feature_cols = get_feature_columns(df)
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False).head(top_n)
    print(f"\n=== 特徴量重要度 TOP {top_n} ===")
    for name, val in importance.items():
        print(f"  {name:25s} {val:6.0f}")
    return importance
