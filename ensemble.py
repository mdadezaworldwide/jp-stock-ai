"""マルチモデル アンサンブル（LightGBM + XGBoost）"""

import pickle
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from features import get_feature_columns
from config import LGBM_PARAMS, XGB_PARAMS, TRAIN_RATIO

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


class EnsembleModel:
    """LightGBM + XGBoost のアンサンブルモデル"""

    def __init__(self, lgbm_params: dict = None, xgb_params: dict = None,
                 lgbm_weight: float = 0.5):
        self.lgbm_params = lgbm_params or LGBM_PARAMS
        self.xgb_params = xgb_params or XGB_PARAMS
        self.lgbm_weight = lgbm_weight
        self.xgb_weight = 1.0 - lgbm_weight
        self.lgbm_model: lgb.LGBMClassifier | None = None
        self.xgb_model: xgb.XGBClassifier | None = None
        self.feature_cols: list[str] = []

    def train(self, df: pd.DataFrame) -> dict:
        """両モデルを訓練"""
        self.feature_cols = get_feature_columns(df)
        X = df[self.feature_cols]
        y = df["Target"]

        split_idx = int(len(df) * TRAIN_RATIO)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"訓練データ: {len(X_train)} 行, テストデータ: {len(X_test)} 行")

        # LightGBM
        print("\n--- LightGBM 訓練 ---")
        self.lgbm_model = lgb.LGBMClassifier(**self.lgbm_params)
        self.lgbm_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(100)],
        )
        lgbm_prob = self.lgbm_model.predict_proba(X_test)[:, 1]
        lgbm_auc = roc_auc_score(y_test, lgbm_prob)
        print(f"  LightGBM AUC: {lgbm_auc:.4f}")

        # XGBoost
        print("\n--- XGBoost 訓練 ---")
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        xgb_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_prob)
        print(f"  XGBoost AUC:  {xgb_auc:.4f}")

        # アンサンブル
        ensemble_prob = lgbm_prob * self.lgbm_weight + xgb_prob * self.xgb_weight
        ensemble_auc = roc_auc_score(y_test, ensemble_prob)
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)

        metrics = {
            "lgbm_auc": lgbm_auc,
            "xgb_auc": xgb_auc,
            "ensemble_auc": ensemble_auc,
            "accuracy": accuracy_score(y_test, ensemble_pred),
            "precision": precision_score(y_test, ensemble_pred, zero_division=0),
            "recall": recall_score(y_test, ensemble_pred, zero_division=0),
        }

        print(f"\n=== アンサンブル評価 ===")
        print(f"  LightGBM AUC: {lgbm_auc:.4f}")
        print(f"  XGBoost AUC:  {xgb_auc:.4f}")
        print(f"  Ensemble AUC: {ensemble_auc:.4f}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")

        # 最適重みを自動調整
        self._optimize_weights(lgbm_prob, xgb_prob, y_test)

        return metrics

    def _optimize_weights(self, lgbm_prob, xgb_prob, y_test):
        """アンサンブル重みを最適化"""
        best_auc = 0
        best_w = 0.5
        for w in np.arange(0.1, 0.95, 0.05):
            combo = lgbm_prob * w + xgb_prob * (1 - w)
            auc = roc_auc_score(y_test, combo)
            if auc > best_auc:
                best_auc = auc
                best_w = w
        self.lgbm_weight = best_w
        self.xgb_weight = 1.0 - best_w
        print(f"  最適重み: LightGBM={best_w:.2f}, XGBoost={1-best_w:.2f} (AUC={best_auc:.4f})")

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """訓練時の特徴量に合わせる（不足は-999埋め）"""
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            for c in missing:
                X = X.copy()
                X[c] = -999
        return X[self.feature_cols]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """アンサンブル確率を返す"""
        X = self._align_features(X)
        lgbm_prob = self.lgbm_model.predict_proba(X)[:, 1]
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        return lgbm_prob * self.lgbm_weight + xgb_prob * self.xgb_weight

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """アンサンブル予測"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """シグナル予測"""
        X = df[get_feature_columns(df)]
        df = df.copy()
        df["Signal_prob"] = self.predict_proba(X)
        df["Signal"] = self.predict(X)
        return df

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        """統合特徴量重要度"""
        lgbm_imp = pd.Series(self.lgbm_model.feature_importances_, index=self.feature_cols)
        xgb_imp = pd.Series(self.xgb_model.feature_importances_, index=self.feature_cols)
        # 正規化して平均
        lgbm_norm = lgbm_imp / lgbm_imp.sum()
        xgb_norm = xgb_imp / xgb_imp.sum()
        combined = lgbm_norm * self.lgbm_weight + xgb_norm * self.xgb_weight
        top = combined.sort_values(ascending=False).head(top_n)
        print(f"\n=== 統合特徴量重要度 TOP {top_n} ===")
        for name, val in top.items():
            print(f"  {name:25s} {val:.4f}")
        return top

    def save(self, path: Path = MODEL_DIR / "ensemble.pkl"):
        """モデル保存"""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"アンサンブルモデル保存: {path}")

    @staticmethod
    def load(path: Path = MODEL_DIR / "ensemble.pkl") -> "EnsembleModel":
        """モデル読み込み"""
        with open(path, "rb") as f:
            return pickle.load(f)
