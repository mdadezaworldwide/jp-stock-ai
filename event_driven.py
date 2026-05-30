"""イベントドリブン分析（決算発表・権利確定日）"""

import pandas as pd
import numpy as np
from safe_yf import download as _yf_download, get_info as _yf_info, get_ticker as _yf_ticker
from datetime import datetime, timedelta

from config import TICKERS, TICKER_NAMES


def get_earnings_dates(ticker: str, limit: int = 8) -> pd.DataFrame:
    """決算発表日を取得"""
    stock = _yf_ticker(ticker)
    try:
        cal = stock.get_earnings_dates(limit=limit)
        if cal is not None and not cal.empty:
            return cal
    except Exception:
        pass
    return pd.DataFrame()


def analyze_earnings_pattern(ticker: str) -> dict:
    """決算前後の株価パターンを分析"""
    stock = _yf_ticker(ticker)
    name = TICKER_NAMES.get(ticker, ticker)

    # 過去の決算日
    try:
        earnings = stock.get_earnings_dates(limit=12)
    except Exception:
        return {"ticker": ticker, "name": name, "has_data": False}

    if earnings is None or earnings.empty:
        return {"ticker": ticker, "name": name, "has_data": False}

    # 株価データ
    end = datetime.now()
    start = end - timedelta(days=365 * 3)
    hist = _yf_download(ticker, start=start, end=end, progress=False)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    if hist.empty:
        return {"ticker": ticker, "name": name, "has_data": False}

    # 各決算日前後のリターンを計算
    pre_returns = []   # 決算5日前→当日
    post_returns = []  # 決算当日→5日後

    for date in earnings.index:
        date = pd.Timestamp(date).tz_localize(None)
        try:
            # 前後の取引日を特定
            before_mask = hist.index <= date
            after_mask = hist.index >= date

            if before_mask.sum() < 6 or after_mask.sum() < 6:
                continue

            pre_idx = hist.index[before_mask][-6]
            on_idx = hist.index[before_mask][-1]
            post_idx = hist.index[after_mask][min(5, after_mask.sum() - 1)]

            pre_ret = hist.loc[on_idx, "Close"] / hist.loc[pre_idx, "Close"] - 1
            post_ret = hist.loc[post_idx, "Close"] / hist.loc[on_idx, "Close"] - 1

            pre_returns.append(pre_ret)
            post_returns.append(post_ret)
        except (IndexError, KeyError):
            continue

    if not pre_returns:
        return {"ticker": ticker, "name": name, "has_data": False}

    # 次回決算日
    future_dates = earnings.index[earnings.index > pd.Timestamp.now(tz=earnings.index.tz)]
    next_earnings = future_dates[0] if len(future_dates) > 0 else None

    return {
        "ticker": ticker,
        "name": name,
        "has_data": True,
        "pre_mean": np.mean(pre_returns),
        "pre_win_rate": np.mean([r > 0 for r in pre_returns]),
        "post_mean": np.mean(post_returns),
        "post_win_rate": np.mean([r > 0 for r in post_returns]),
        "sample_count": len(pre_returns),
        "next_earnings": str(next_earnings.date()) if next_earnings else "未定",
    }


def analyze_all_events() -> pd.DataFrame:
    """全銘柄のイベント分析"""
    results = []
    for ticker in TICKERS:
        name = TICKER_NAMES.get(ticker, ticker)
        print(f"  イベント分析: {name}")
        result = analyze_earnings_pattern(ticker)
        results.append(result)

    df = pd.DataFrame(results)
    return df


def print_event_report():
    """イベント分析レポート表示"""
    df = analyze_all_events()

    print("\n" + "=" * 85)
    print("  イベントドリブン分析（決算前後のパターン）")
    print("=" * 85)
    print(f"  {'銘柄':>10s} {'決算前5日':>10s} {'勝率':>6s} {'決算後5日':>10s} {'勝率':>6s} {'件数':>4s} {'次回決算':>12s}")
    print("-" * 85)

    for _, row in df.iterrows():
        if not row.get("has_data"):
            print(f"  {row['name']:>10s}  データなし")
            continue
        print(
            f"  {row['name']:>10s}"
            f"  {row['pre_mean']:>+9.2%}"
            f"  {row['pre_win_rate']:>5.0%}"
            f"  {row['post_mean']:>+9.2%}"
            f"  {row['post_win_rate']:>5.0%}"
            f"  {row['sample_count']:>4.0f}"
            f"  {row['next_earnings']:>12s}"
        )
    print("=" * 85)

    # 決算が近い銘柄をハイライト
    if "next_earnings" in df.columns:
        upcoming = df[(df["has_data"] == True) & (df["next_earnings"] != "未定")]
        if not upcoming.empty:
            print(f"\n  直近の決算予定:")
            for _, row in upcoming.iterrows():
                bias = "強気" if row["pre_mean"] > 0 else "弱気"
                print(f"    {row['name']}: {row['next_earnings']} (過去パターン: {bias})")


if __name__ == "__main__":
    print_event_report()
