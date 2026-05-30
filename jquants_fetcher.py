"""J-Quants API データ取得（REST API直接利用）"""

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

import pandas as pd

JQUANTS_API_KEY = os.environ.get("JQUANTS_API_KEY", "xpGYSVBrYTYl71ZBlXmQToqLdcY1jQOvazpM0llT8NY")
BASE_URL = "https://api.jquants.com/v2"


def _api_get(endpoint: str, params: dict = None) -> dict:
    """J-Quants API v2にGETリクエスト（x-api-key認証）"""
    if not JQUANTS_API_KEY:
        return {}

    url = f"{BASE_URL}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, headers={"x-api-key": JQUANTS_API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"  [WARN] J-Quants API エラー ({endpoint}, HTTP {e.code}): {body[:200]}")
        return {}
    except Exception as e:
        print(f"  [WARN] J-Quants API エラー ({endpoint}): {e}")
        return {}


def fetch_stock_data_jquants(ticker: str, years: int = 5) -> pd.DataFrame:
    """J-Quantsから株価データを取得"""
    code = ticker.replace(".T", "")
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    data = _api_get("/equities/bars/daily", {
        "code": code,
        "from": start.strftime("%Y%m%d"),
        "to": end.strftime("%Y%m%d"),
    })

    # v2のレスポンスキーを確認
    quotes_key = None
    for key in ["daily_quotes", "bars", "data"]:
        if key in data:
            quotes_key = key
            break
    if not data or not quotes_key:
        return pd.DataFrame()

    df = pd.DataFrame(data[quotes_key])
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # yfinance互換カラム名（v2: AdjO/AdjH/AdjL/AdjC/AdjVo）
    rename = {
        "AdjO": "Open", "AdjustmentOpen": "Open",
        "AdjH": "High", "AdjustmentHigh": "High",
        "AdjL": "Low", "AdjustmentLow": "Low",
        "AdjC": "Close", "AdjustmentClose": "Close",
        "AdjVo": "Volume", "AdjustmentVolume": "Volume",
    }
    df = df.rename(columns=rename)
    df["Ticker"] = ticker

    cols = ["Open", "High", "Low", "Close", "Volume", "Ticker"]
    return df[[c for c in cols if c in df.columns]]


def fetch_financial_statements_jquants(ticker: str) -> dict:
    """J-Quantsから財務データを取得"""
    code = ticker.replace(".T", "")
    data = _api_get("/equities/financial_statements", {"code": code})

    # v2のレスポンスキーを確認
    stmt_key = None
    for key in ["statements", "financial_statements", "data"]:
        if key in data:
            stmt_key = key
            break
    if not data or not stmt_key:
        return {}

    statements = data[stmt_key]
    if not statements:
        return {}

    latest = statements[-1]
    prev = statements[-2] if len(statements) >= 2 else None

    features = {}

    revenue = latest.get("NetSales")
    op_income = latest.get("OperatingProfit")
    net_income = latest.get("Profit")

    if revenue and op_income and revenue != 0:
        features["JQ_op_margin"] = op_income / revenue

    if prev:
        prev_revenue = prev.get("NetSales")
        prev_income = prev.get("Profit")
        if prev_revenue and revenue and prev_revenue != 0:
            features["JQ_revenue_growth"] = (revenue - prev_revenue) / abs(prev_revenue)
        if prev_income and net_income and prev_income != 0:
            features["JQ_income_growth"] = (net_income - prev_income) / abs(prev_income)

    eps = latest.get("EarningsPerShare")
    if eps:
        features["JQ_EPS"] = eps

    dividend = latest.get("DividendPerShare")
    if dividend:
        features["JQ_DPS"] = dividend

    return features


def fetch_all_data_jquants(tickers: list[str], years: int = 5) -> pd.DataFrame:
    """全銘柄のデータをJ-Quantsから取得"""
    if not JQUANTS_API_KEY:
        return pd.DataFrame()

    frames = []
    for ticker in tickers:
        print(f"  J-Quants取得: {ticker}")
        df = fetch_stock_data_jquants(ticker, years)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def get_jquants_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """全銘柄の財務データをJ-Quantsから取得"""
    if not JQUANTS_API_KEY:
        return pd.DataFrame()

    rows = []
    for ticker in tickers:
        print(f"  J-Quants財務: {ticker}")
        features = fetch_financial_statements_jquants(ticker)
        if features:
            features["Ticker"] = ticker
            rows.append(features)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c != "Ticker"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def is_jquants_available() -> bool:
    """J-Quantsが使用可能か"""
    return bool(JQUANTS_API_KEY)
