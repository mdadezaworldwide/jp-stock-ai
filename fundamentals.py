"""財務諸表分析 — yfinanceから財務データを取得し特徴量化"""

import pandas as pd
import numpy as np
import yfinance as yf
from config import TICKERS


def fetch_fundamentals(ticker: str) -> dict:
    """1銘柄の財務指標を取得"""
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    fundamentals = {
        # === バリュエーション ===
        "PER": info.get("trailingPE"),
        "Forward_PER": info.get("forwardPE"),
        "PBR": info.get("priceToBook"),
        "PSR": info.get("priceToSalesTrailing12Months"),
        "EV_EBITDA": info.get("enterpriseToEbitda"),
        "Dividend_yield": info.get("dividendYield"),

        # === 収益性 ===
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "Profit_margin": info.get("profitMargins"),
        "Operating_margin": info.get("operatingMargins"),
        "Gross_margin": info.get("grossMargins"),

        # === 成長性 ===
        "Revenue_growth": info.get("revenueGrowth"),
        "Earnings_growth": info.get("earningsGrowth"),

        # === 財務健全性 ===
        "Debt_to_equity": info.get("debtToEquity"),
        "Current_ratio": info.get("currentRatio"),
        "Quick_ratio": info.get("quickRatio"),

        # === 規模 ===
        "Market_cap": info.get("marketCap"),
        "Enterprise_value": info.get("enterpriseValue"),

        # === アナリスト ===
        "Target_mean_price": info.get("targetMeanPrice"),
        "Recommendation": _encode_recommendation(info.get("recommendationKey")),
        "Num_analysts": info.get("numberOfAnalystOpinions"),
    }

    # 目標株価との乖離率
    current = info.get("currentPrice")
    target = info.get("targetMeanPrice")
    if current and target:
        fundamentals["Target_upside"] = (target - current) / current

    return fundamentals


def fetch_financial_statements(ticker: str) -> dict:
    """損益計算書・BSから追加指標を算出"""
    stock = yf.Ticker(ticker)
    features = {}

    # 損益計算書
    income = stock.quarterly_income_stmt
    if income is not None and not income.empty:
        latest = income.iloc[:, 0]
        prev = income.iloc[:, 1] if income.shape[1] > 1 else None

        revenue = latest.get("Total Revenue")
        net_income = latest.get("Net Income")

        if revenue and prev is not None:
            prev_revenue = prev.get("Total Revenue")
            if prev_revenue and prev_revenue != 0:
                features["QoQ_revenue_growth"] = (revenue - prev_revenue) / abs(prev_revenue)

        if net_income and prev is not None:
            prev_income = prev.get("Net Income")
            if prev_income and prev_income != 0:
                features["QoQ_income_growth"] = (net_income - prev_income) / abs(prev_income)

    # バランスシート
    bs = stock.quarterly_balance_sheet
    if bs is not None and not bs.empty:
        latest_bs = bs.iloc[:, 0]
        total_assets = latest_bs.get("Total Assets")
        total_equity = latest_bs.get("Stockholders Equity")
        if total_assets and total_equity and total_assets != 0:
            features["Equity_ratio"] = total_equity / total_assets

    # キャッシュフロー
    cf = stock.quarterly_cashflow
    if cf is not None and not cf.empty:
        latest_cf = cf.iloc[:, 0]
        op_cf = latest_cf.get("Operating Cash Flow")
        cap_ex = latest_cf.get("Capital Expenditure")
        if op_cf is not None and cap_ex is not None:
            features["FCF"] = op_cf + cap_ex  # capexは負の値

    return features


def get_all_fundamentals(tickers: list[str] = TICKERS) -> pd.DataFrame:
    """全銘柄の財務指標をDataFrameで返す"""
    rows = []
    for ticker in tickers:
        print(f"  財務データ取得: {ticker}")
        try:
            base = fetch_fundamentals(ticker)
            extra = fetch_financial_statements(ticker)
            base.update(extra)
            base["Ticker"] = ticker
            rows.append(base)
        except Exception as e:
            print(f"  [WARN] {ticker} 財務データ取得失敗: {e}")
    df = pd.DataFrame(rows)
    # 数値以外をNaNに
    numeric_cols = [c for c in df.columns if c != "Ticker"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def _encode_recommendation(rec: str | None) -> float | None:
    """アナリスト推奨を数値化"""
    mapping = {
        "strongBuy": 2.0,
        "buy": 1.0,
        "hold": 0.0,
        "sell": -1.0,
        "strongSell": -2.0,
    }
    return mapping.get(rec)


if __name__ == "__main__":
    df = get_all_fundamentals()
    print(df.to_string())
