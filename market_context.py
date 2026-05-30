"""マーケット全体の状況分析"""

import pandas as pd
import numpy as np
from safe_yf import download as _yf_download, get_info as _yf_info, get_ticker as _yf_ticker
import ta
from datetime import datetime, timedelta

from config import MARKET_INDICES


def fetch_market_data(days: int = 120) -> pd.DataFrame:
    """市場指標データを取得"""
    end = datetime.now()
    start = end - timedelta(days=days)
    frames = {}

    for symbol, name in MARKET_INDICES.items():
        print(f"  市場データ取得: {name}")
        try:
            df = _yf_download(symbol, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                frames[symbol] = df["Close"]
        except Exception as e:
            print(f"  [WARN] {name} 取得失敗: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.DataFrame(frames)


def compute_market_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """市場全体の特徴量を算出"""
    features = pd.DataFrame(index=market_df.index)

    # 日経225の指標
    if "^N225" in market_df.columns:
        n225 = market_df["^N225"]
        features["N225_return_1d"] = n225.pct_change(1)
        features["N225_return_5d"] = n225.pct_change(5)
        features["N225_return_20d"] = n225.pct_change(20)
        features["N225_volatility_20d"] = n225.pct_change().rolling(20).std()
        features["N225_SMA20_dev"] = (n225 - n225.rolling(20).mean()) / n225.rolling(20).mean()
        features["N225_RSI"] = ta.momentum.rsi(n225, window=14)

        # 日経225が上昇トレンドか
        sma5 = n225.rolling(5).mean()
        sma20 = n225.rolling(20).mean()
        features["N225_trend"] = (sma5 > sma20).astype(int)

    # TOPIX ETF
    if "1306.T" in market_df.columns:
        topix = market_df["1306.T"]
        features["TOPIX_return_1d"] = topix.pct_change(1)
        features["TOPIX_return_5d"] = topix.pct_change(5)

    # NT倍率（日経/TOPIX ETF）
    if "^N225" in market_df.columns and "1306.T" in market_df.columns:
        features["NT_ratio"] = market_df["^N225"] / market_df["1306.T"]
        features["NT_ratio_change"] = features["NT_ratio"].pct_change(5)

    # ドル円
    if "USDJPY=X" in market_df.columns:
        usdjpy = market_df["USDJPY=X"]
        features["USDJPY_return_1d"] = usdjpy.pct_change(1)
        features["USDJPY_return_5d"] = usdjpy.pct_change(5)
        features["USDJPY_level"] = usdjpy  # 円安/円高レベル

    # VIX（恐怖指数）
    if "^VIX" in market_df.columns:
        vix = market_df["^VIX"]
        features["VIX_level"] = vix
        features["VIX_change"] = vix.pct_change(5)
        features["VIX_high"] = (vix > 25).astype(int)  # 高ボラ環境

    return features


def get_market_context() -> pd.DataFrame:
    """市場コンテキスト特徴量を返す"""
    market_df = fetch_market_data()
    if market_df.empty:
        return pd.DataFrame()
    return compute_market_features(market_df)


if __name__ == "__main__":
    ctx = get_market_context()
    print(ctx.tail(10).to_string())
