"""シグナルデータを事前計算してCSVに保存（ダッシュボードはこれを読むだけ）"""

import config
from data_fetcher import fetch_all_data
from features import add_technical_features, get_feature_columns
from ensemble import EnsembleModel
from stock_scorer import calc_technical_score, score_to_label, get_hold_recommendation
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "signal_data"
DATA_DIR.mkdir(exist_ok=True)

PERIODS = {1: 0.01, 5: 0.02, 10: 0.03, 20: 0.04, 60: 0.06}


def update_all():
    print("=== シグナルデータ更新 ===")
    print("データ取得中...")
    raw_data = fetch_all_data()

    # テクニカル指標生成
    print("テクニカル指標生成中...")
    frames = []
    for ticker, group in raw_data.groupby("Ticker"):
        group = group.sort_index()
        group = add_technical_features(group)
        frames.append(group)
    df = pd.concat(frames).dropna()

    for days, target in PERIODS.items():
        print(f"\n--- {days}日モデル ---")
        model_path = Path(__file__).parent / "models" / f"ensemble_{days}d.pkl"
        if not model_path.exists():
            print(f"  モデルなし、スキップ")
            continue

        model = EnsembleModel.load(model_path)
        df_sig = model.predict_signals(df)

        signals = []
        for ticker in df_sig["Ticker"].unique():
            td = df_sig[df_sig["Ticker"] == ticker]
            if td.empty:
                continue
            latest = td.iloc[-1]
            rsi_val = latest.get("RSI_14", np.nan)
            macd_val = latest.get("MACD_hist", np.nan)

            if pd.notna(rsi_val) and rsi_val != -999:
                if rsi_val >= 70: rsi_label = "買われすぎ"
                elif rsi_val <= 30: rsi_label = "売られすぎ"
                elif rsi_val <= 40: rsi_label = "やや売られすぎ"
                elif rsi_val >= 60: rsi_label = "やや買われすぎ"
                else: rsi_label = "普通"
            else:
                rsi_label = "-"

            if pd.notna(macd_val) and macd_val != -999:
                if macd_val > 0:
                    macd_label = "上昇の勢い加速" if macd_val > 0.5 else "上昇の勢い鈍化"
                else:
                    macd_label = "下落の勢い加速" if macd_val < -0.5 else "下落止まりつつある"
            else:
                macd_label = "-"

            tech_score = calc_technical_score(latest)
            hold_label, hold_reason = get_hold_recommendation(latest, days)

            signals.append({
                "銘柄": config.TICKER_NAMES.get(ticker, ticker),
                "ティッカー": ticker,
                "セクター": config.TICKER_SECTORS.get(ticker, "その他"),
                "終値": latest["Close"],
                "シグナル確率": latest["Signal_prob"],
                "判定": "BUY" if latest["Signal"] == 1 else "-",
                "テクニカル": f"{tech_score:+d} ({score_to_label(tech_score)})",
                "ファンダ": "-",
                "RSI": rsi_val,
                "MACD判定": macd_label,
                "推奨保有": hold_label,
                "保有理由": hold_reason,
            })

        result = pd.DataFrame(signals)
        sort_order = {"BUY": 0}
        result["_sort"] = result["判定"].map(sort_order).fillna(9)
        result = result.sort_values(["_sort", "シグナル確率"], ascending=[True, False]).drop(columns=["_sort"])
        result = result.reset_index(drop=True)
        result.insert(0, "No.", range(1, len(result) + 1))

        csv_path = DATA_DIR / f"signals_{days}d.csv"
        result.to_csv(csv_path, index=False)
        print(f"  保存: {csv_path} ({len(result)}銘柄)")

    print("\n=== 全期間の更新完了 ===")


if __name__ == "__main__":
    update_all()
