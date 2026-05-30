"""セクターローテーション分析"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# 東証セクター別ETF
SECTOR_ETFS = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1623.T": "自動車・輸送機",
    "1624.T": "鉄鋼・非鉄",
    "1625.T": "機械",
    "1626.T": "電機・精密",
    "1627.T": "情報通信・サービス",
    "1628.T": "電力・ガス",
    "1629.T": "運輸・物流",
    "1630.T": "商社・卸売",
    "1631.T": "小売",
    "1632.T": "銀行",
    "1633.T": "金融(除く銀行)",
    "1634.T": "不動産",
}


def fetch_sector_data(days: int = 120) -> pd.DataFrame:
    """セクターETFデータを取得"""
    end = datetime.now()
    start = end - timedelta(days=days)
    frames = {}

    for ticker, name in SECTOR_ETFS.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                frames[name] = df["Close"]
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


def analyze_sector_rotation(days: int = 120) -> pd.DataFrame:
    """セクターの強弱を分析"""
    print("  セクターデータ取得中...")
    data = fetch_sector_data(days)
    if data.empty:
        print("  [WARN] セクターデータ取得失敗")
        return pd.DataFrame()

    results = []
    for sector in data.columns:
        prices = data[sector].dropna()
        if len(prices) < 20:
            continue

        # 各期間のリターン
        ret_5d = prices.iloc[-1] / prices.iloc[-6] - 1 if len(prices) >= 6 else np.nan
        ret_20d = prices.iloc[-1] / prices.iloc[-21] - 1 if len(prices) >= 21 else np.nan
        ret_60d = prices.iloc[-1] / prices.iloc[-61] - 1 if len(prices) >= 61 else np.nan

        # モメンタム（短期 vs 長期）
        momentum = ret_5d - ret_60d if pd.notna(ret_5d) and pd.notna(ret_60d) else np.nan

        # ボラティリティ
        volatility = prices.pct_change().tail(20).std() * np.sqrt(252)

        # トレンド強度（SMA5 vs SMA20）
        sma5 = prices.rolling(5).mean().iloc[-1]
        sma20 = prices.rolling(20).mean().iloc[-1]
        trend = "上昇" if sma5 > sma20 else "下降"

        results.append({
            "セクター": sector,
            "5日リターン": ret_5d,
            "20日リターン": ret_20d,
            "60日リターン": ret_60d,
            "モメンタム": momentum,
            "ボラティリティ": volatility,
            "トレンド": trend,
        })

    df = pd.DataFrame(results)
    df = df.sort_values("モメンタム", ascending=False)
    return df


def get_strong_sectors(top_n: int = 5) -> list[str]:
    """強いセクターのリストを返す"""
    df = analyze_sector_rotation()
    if df.empty:
        return []
    strong = df[df["トレンド"] == "上昇"].head(top_n)
    return strong["セクター"].tolist()


def print_sector_report():
    """セクター分析レポート表示"""
    df = analyze_sector_rotation()
    if df.empty:
        print("  セクターデータなし")
        return

    print("\n" + "=" * 80)
    print("  セクターローテーション分析")
    print("=" * 80)
    print(f"  {'セクター':>14s} {'5日':>8s} {'20日':>8s} {'60日':>8s} {'モメンタム':>10s} {'Vol':>6s} {'トレンド':>6s}")
    print("-" * 80)

    for _, row in df.iterrows():
        print(
            f"  {row['セクター']:>14s}"
            f"  {row['5日リターン']:>+7.1%}"
            f"  {row['20日リターン']:>+7.1%}"
            f"  {row['60日リターン']:>+7.1%}"
            f"  {row['モメンタム']:>+9.3f}"
            f"  {row['ボラティリティ']:>5.1%}"
            f"  {row['トレンド']:>6s}"
        )
    print("=" * 80)

    strong = df[df["トレンド"] == "上昇"]
    weak = df[df["トレンド"] == "下降"]
    print(f"\n  有望セクター: {', '.join(strong['セクター'].head(3).tolist())}")
    print(f"  警戒セクター: {', '.join(weak['セクター'].tail(3).tolist())}")


if __name__ == "__main__":
    print_sector_report()
