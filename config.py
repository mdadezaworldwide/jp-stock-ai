"""設定ファイル"""
import os

# === 対象銘柄（東証プライム時価総額上位500銘柄） ===
from tse_prime_tickers import TICKERS, TICKER_NAMES, TICKER_SECTORS

# マーケット指標
MARKET_INDICES = {
    "^N225": "日経225",
    "1306.T": "TOPIX ETF",
    "USDJPY=X": "ドル円",
    "^VIX": "VIX",
}

# === データ取得 ===
DATA_PERIOD_YEARS = 5

# === モデル設定 ===
HOLD_DAYS = 5
TARGET_RETURN = 0.02
TRAIN_RATIO = 0.8

# === バックテスト設定 ===
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 3
POSITION_SIZE = 0.3

# === リスク管理 ===
MAX_DRAWDOWN_LIMIT = 0.10
MAX_SECTOR_EXPOSURE = 0.5
CORRELATION_THRESHOLD = 0.8
ATR_STOP_LOSS_MULTIPLIER = 2.0
ATR_TAKE_PROFIT_MULTIPLIER = 3.0

# === LightGBM パラメータ ===
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# === XGBoost パラメータ ===
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "verbosity": 0,
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# === リアルタイム設定 ===
RETRAIN_HOUR = 18
SIGNAL_CHECK_MINUTES = 30

# === 通知設定（環境変数） ===
LINE_NOTIFY_TOKEN = os.environ.get("LINE_NOTIFY_TOKEN", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# === X (Twitter) API ===
X_BEARER_TOKEN = os.environ.get("X_BEARER_TOKEN", "")
SENTIMENT_LOOKBACK_HOURS = 24

# === Claude API（ニュース分析） ===
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# === J-Quants API ===
JQUANTS_EMAIL = os.environ.get("JQUANTS_EMAIL", "")
JQUANTS_PASSWORD = os.environ.get("JQUANTS_PASSWORD", "")

# === Optuna ===
OPTUNA_N_TRIALS = 1500
