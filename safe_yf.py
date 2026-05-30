"""yfinanceのセーフラッパー — API制限時にエラーを吸収"""

import pandas as pd


def download(ticker, **kwargs):
    """yf.downloadのセーフ版"""
    try:
        import yfinance as yf
        df = yf.download(ticker, progress=False, **kwargs)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def get_info(ticker) -> dict:
    """yf.Ticker(ticker).infoのセーフ版"""
    try:
        import yfinance as yf
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def get_ticker(ticker):
    """yf.Tickerのセーフ版"""
    try:
        import yfinance as yf
        return yf.Ticker(ticker)
    except Exception:
        return None
