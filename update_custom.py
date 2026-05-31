"""カスタム銘柄分析事前計算 (GitHub Actions向け)"""

import json
from pathlib import Path

import pandas as pd

from custom_stocks import load_custom_stocks, analyze_custom_stock

DATA_DIR = Path(__file__).parent / "signal_data"
HIST_DIR = DATA_DIR / "custom_hist"
INFO_DIR = DATA_DIR / "custom_info"
DATA_DIR.mkdir(exist_ok=True)
HIST_DIR.mkdir(exist_ok=True)
INFO_DIR.mkdir(exist_ok=True)


def update_custom():
    print("=== カスタム銘柄分析更新 ===")
    stocks = load_custom_stocks()
    if not stocks:
        print("カスタム銘柄なし")
        pd.DataFrame(columns=["銘柄", "ティッカー", "現在値", "判定"]).to_csv(
            DATA_DIR / "custom_analysis.csv", index=False
        )
        return

    rows = []
    for cs in stocks:
        ticker = cs["ticker"]
        name = cs.get("name", ticker)
        print(f"  分析中: {name} ({ticker})")
        try:
            result = analyze_custom_stock(ticker)
        except Exception as e:
            print(f"  失敗: {e}")
            continue

        if "error" in result:
            print(f"  エラー: {result['error']}")
            continue

        # チャート用OHLCVを保存
        hist = result.pop("hist", None)
        if hist is not None and not hist.empty:
            hist.to_csv(HIST_DIR / f"{ticker}.csv")

        # ファンダメンタルズを別JSON保存
        fundamentals = result.pop("fundamentals", {})
        with open(INFO_DIR / f"{ticker}.json", "w", encoding="utf-8") as f:
            json.dump(fundamentals, f, ensure_ascii=False, default=str)

        rows.append({
            "銘柄": name,
            "ティッカー": ticker,
            "現在値": result.get("current"),
            "判定": result.get("signal"),
            "スコア": result.get("score"),
            "トレンド": result.get("trend"),
            "RSI": result.get("rsi"),
            "RSI判定": result.get("rsi_label"),
            "MACD判定": result.get("macd_label"),
            "ADX": result.get("adx"),
            "推奨保有": result.get("hold_label"),
            "保有理由": result.get("hold_reason"),
            "5日リターン": result.get("ret_5d"),
            "20日リターン": result.get("ret_20d"),
        })

    df = pd.DataFrame(rows)
    out = DATA_DIR / "custom_analysis.csv"
    df.to_csv(out, index=False)
    print(f"保存: {out} ({len(df)}銘柄)")


if __name__ == "__main__":
    update_custom()
