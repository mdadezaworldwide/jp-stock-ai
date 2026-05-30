"""日本株データの取得（J-Quants優先、フォールバックでyfinance）"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import TICKERS, DATA_PERIOD_YEARS


def fetch_stock_data(ticker: str, years: int = DATA_PERIOD_YEARS) -> pd.DataFrame:
    """1銘柄の株価データを取得"""
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        print(f"[WARN] {ticker} のデータ取得に失敗")
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["Ticker"] = ticker
    return df


def fetch_all_data(tickers: list[str] = TICKERS) -> pd.DataFrame:
    """全銘柄のデータを取得（J-Quants優先）"""
    from jquants_fetcher import is_jquants_available, fetch_all_data_jquants

    # J-Quantsが使用可能ならそちらを使う
    if is_jquants_available():
        print("データソース: J-Quants API")
        df = fetch_all_data_jquants(tickers)
        if not df.empty:
            return df
        print("  J-Quantsフォールバック → yfinance")

    # yfinance
    print("データソース: yfinance")
    frames = []
    for ticker in tickers:
        print(f"取得中: {ticker}")
        df = fetch_stock_data(ticker)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("データを取得できませんでした")
    return pd.concat(frames)


if __name__ == "__main__":
    data = fetch_all_data()
    print(f"取得完了: {len(data)} 行, 銘柄数: {data['Ticker'].nunique()}")
