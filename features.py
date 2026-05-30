"""テクニカル指標 + ファンダメンタルズ + センチメント + 市場コンテキスト"""

import pandas as pd
import numpy as np
import ta
from config import HOLD_DAYS, TARGET_RETURN, TICKERS
from fundamentals import get_all_fundamentals
from sentiment import get_all_sentiments
from market_context import get_market_context


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を追加"""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # === トレンド系 ===
    for period in [5, 10, 20, 60]:
        df[f"SMA_{period}"] = ta.trend.sma_indicator(close, window=period)
        df[f"EMA_{period}"] = ta.trend.ema_indicator(close, window=period)

    for period in [5, 20, 60]:
        sma = df[f"SMA_{period}"]
        df[f"SMA_{period}_deviation"] = (close - sma) / sma

    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    df["ADX"] = ta.trend.adx(high, low, close)

    # === モメンタム系 ===
    df["RSI_14"] = ta.momentum.rsi(close, window=14)
    df["RSI_9"] = ta.momentum.rsi(close, window=9)

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    for period in [1, 3, 5, 10, 20]:
        df[f"ROC_{period}"] = ta.momentum.roc(close, window=period)

    # === ボラティリティ系 ===
    bb = ta.volatility.BollingerBands(close)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = bb.bollinger_wband()
    df["BB_position"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    df["ATR_14"] = ta.volatility.average_true_range(high, low, close)

    # === 出来高系 ===
    df["Volume_SMA_20"] = volume.rolling(20).mean()
    df["Volume_ratio"] = volume / df["Volume_SMA_20"]
    df["OBV"] = ta.volume.on_balance_volume(close, volume)

    # === 価格パターン ===
    df["Body_ratio"] = (close - df["Open"]) / (high - low + 1e-8)
    df["Upper_shadow"] = (high - np.maximum(close, df["Open"])) / (high - low + 1e-8)
    df["Lower_shadow"] = (np.minimum(close, df["Open"]) - low) / (high - low + 1e-8)

    for period in [10, 20]:
        df[f"High_{period}_pos"] = (close - low.rolling(period).min()) / (
            high.rolling(period).max() - low.rolling(period).min() + 1e-8
        )

    return df


def add_fundamental_features(df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """ファンダメンタルズ特徴量をマージ（インデックス保持）"""
    if fund_df.empty:
        return df
    fund_cols = [c for c in fund_df.columns if c != "Ticker"]
    for col in fund_cols:
        mapping = fund_df.set_index("Ticker")[col].to_dict()
        df[col] = df["Ticker"].map(mapping)
    return df


def add_sentiment_features(df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
    """センチメント特徴量をマージ（インデックス保持）"""
    if sent_df.empty:
        return df
    sent_cols = [c for c in sent_df.columns if c != "Ticker"]
    for col in sent_cols:
        mapping = sent_df.set_index("Ticker")[col].to_dict()
        df[col] = df["Ticker"].map(mapping)
    return df


def add_market_features(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """市場コンテキスト特徴量をマージ（インデックス保持）"""
    if market_df.empty:
        return df
    market_df = market_df.copy()
    # 日付インデックスで結合
    market_cols = market_df.columns.tolist()
    for col in market_cols:
        df[col] = df.index.map(market_df[col].to_dict())
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """目標変数を作成（config.HOLD_DAYSをリアルタイムで参照）"""
    import config
    hold = config.HOLD_DAYS
    target_ret = config.TARGET_RETURN
    df["Future_return"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.shift(-hold) / x - 1
    )
    df["Target"] = (df["Future_return"] >= target_ret).astype(int)
    return df


def add_news_features(df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """ニュース分析特徴量をマージ（インデックス保持）"""
    if news_df.empty:
        return df
    news_cols = [c for c in news_df.columns if c != "Ticker"]
    for col in news_cols:
        mapping = news_df.set_index("Ticker")[col].to_dict()
        df[col] = df["Ticker"].map(mapping)
    return df


def add_jquants_features(df: pd.DataFrame, jq_df: pd.DataFrame) -> pd.DataFrame:
    """J-Quants財務特徴量をマージ（インデックス保持）"""
    if jq_df.empty:
        return df
    jq_cols = [c for c in jq_df.columns if c != "Ticker"]
    for col in jq_cols:
        mapping = jq_df.set_index("Ticker")[col].to_dict()
        df[col] = df["Ticker"].map(mapping)
    return df


def prepare_features(
    df: pd.DataFrame,
    include_fundamentals: bool = True,
    include_sentiment: bool = True,
    include_market: bool = True,
    include_news: bool = True,
    include_jquants: bool = True,
) -> pd.DataFrame:
    """全データソースを統合して特徴量を生成"""
    # 1. テクニカル指標
    result = []
    for ticker, group in df.groupby("Ticker"):
        group = group.sort_index()
        group = add_technical_features(group)
        result.append(group)
    combined = pd.concat(result)

    # 2. ファンダメンタルズ
    if include_fundamentals:
        print("  ファンダメンタルズ取得中...")
        try:
            fund_df = get_all_fundamentals(combined["Ticker"].unique().tolist())
            combined = add_fundamental_features(combined, fund_df)
        except Exception as e:
            print(f"  [WARN] ファンダメンタルズ取得失敗: {e}")

    # 3. センチメント
    if include_sentiment:
        print("  センチメント分析中...")
        try:
            sent_df = get_all_sentiments(combined["Ticker"].unique().tolist())
            combined = add_sentiment_features(combined, sent_df)
        except Exception as e:
            print(f"  [WARN] センチメント取得失敗: {e}")

    # 4. 市場コンテキスト
    if include_market:
        print("  市場コンテキスト取得中...")
        try:
            market_df = get_market_context()
            combined = add_market_features(combined, market_df)
        except Exception as e:
            print(f"  [WARN] 市場コンテキスト取得失敗: {e}")

    # 5. Claude APIニュース分析
    if include_news:
        print("  ニュース分析中（Claude API）...")
        try:
            from news_analyzer import get_news_features
            news_df = get_news_features()
            combined = add_news_features(combined, news_df)
        except Exception as e:
            print(f"  [WARN] ニュース分析失敗: {e}")

    # 6. J-Quants財務データ
    if include_jquants:
        try:
            from jquants_fetcher import is_jquants_available, get_jquants_fundamentals
            if is_jquants_available():
                print("  J-Quants財務データ取得中...")
                jq_df = get_jquants_fundamentals(combined["Ticker"].unique().tolist())
                combined = add_jquants_features(combined, jq_df)
        except Exception as e:
            print(f"  [WARN] J-Quants財務取得失敗: {e}")

    # 7. 目標変数
    combined = create_target(combined)
    combined = combined.dropna(subset=["Target"])

    # NaNを-999で埋める
    feature_cols = get_feature_columns(combined)
    combined[feature_cols] = combined[feature_cols].fillna(-999)

    print(f"  特徴量数: {len(feature_cols)}, データ行数: {len(combined)}")

    return combined


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """特徴量カラム名のリストを返す"""
    exclude = {"Open", "High", "Low", "Close", "Adj Close", "Volume",
               "Ticker", "Future_return", "Target"}
    return [c for c in df.columns if c not in exclude]
