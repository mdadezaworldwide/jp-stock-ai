"""銘柄ごとの最適保有期間アドバイザー"""

import pandas as pd
import numpy as np
import ta


def advise_hold_period(ticker_df: pd.DataFrame) -> dict:
    """
    テクニカル指標から最適な保有期間を推定

    短期(3-10日): モメンタムが強い、ボラティリティ高い
    中期(10-30日): トレンド形成中、移動平均上向き
    長期(30-180日): 強いトレンド＋ファンダメンタルズ良好
    """
    if len(ticker_df) < 60:
        return {"days": 5, "label": "短期(5日)", "reason": "データ不足"}

    close = ticker_df["Close"]
    latest = close.iloc[-1]

    # トレンド強度（ADX）
    adx = ticker_df.get("ADX")
    adx_val = float(adx.iloc[-1]) if adx is not None and not adx.empty and pd.notna(adx.iloc[-1]) else 20

    # 移動平均の並び
    sma5 = close.rolling(5).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma60 = close.rolling(60).mean().iloc[-1]

    # パーフェクトオーダー（短期>中期>長期）= 強いトレンド
    perfect_order = sma5 > sma20 > sma60

    # RSI
    rsi_val = ticker_df.get("RSI_14")
    rsi = float(rsi_val.iloc[-1]) if rsi_val is not None and not rsi_val.empty and pd.notna(rsi_val.iloc[-1]) else 50

    # ボラティリティ（20日リターンの標準偏差）
    volatility = close.pct_change().tail(20).std()

    # ROC（変化率）で短期の勢いを判定
    roc_5 = (latest / close.iloc[-6] - 1) if len(close) >= 6 else 0
    roc_20 = (latest / close.iloc[-21] - 1) if len(close) >= 21 else 0

    # 判定ロジック
    score_short = 0  # 短期向きスコア
    score_mid = 0    # 中期向きスコア
    score_long = 0   # 長期向きスコア

    # ボラティリティ高い → 短期
    if volatility > 0.025:
        score_short += 2
    elif volatility > 0.015:
        score_mid += 1
    else:
        score_long += 1

    # ADX高い（強いトレンド） → 中〜長期
    if adx_val > 30:
        score_mid += 1
        score_long += 2
    elif adx_val > 20:
        score_mid += 2
    else:
        score_short += 1

    # パーフェクトオーダー → 長期
    if perfect_order:
        score_long += 3
        score_mid += 1

    # RSI極端 → 短期リバウンド狙い
    if rsi < 30 or rsi > 70:
        score_short += 2

    # 短期の勢い強い → 短期で利確
    if abs(roc_5) > 0.05:
        score_short += 2

    # 中期の勢い安定 → 中期
    if 0.02 < roc_20 < 0.10:
        score_mid += 2
        score_long += 1

    # 判定
    scores = {"short": score_short, "mid": score_mid, "long": score_long}
    best = max(scores, key=scores.get)

    if best == "short":
        days = 5 if volatility > 0.03 else 7
        label = f"短期({days}日)"
        if rsi < 30:
            reason = "売られすぎのリバウンド狙い"
        elif volatility > 0.025:
            reason = "ボラティリティが高く短期決着が有利"
        else:
            reason = "短期モメンタムが強い"
    elif best == "mid":
        days = 15 if adx_val > 25 else 20
        label = f"中期({days}日)"
        if adx_val > 25:
            reason = "トレンド形成中、勢いに乗る"
        else:
            reason = "移動平均が上向き、じっくり保有"
    else:
        if perfect_order:
            days = 60
            label = "長期(60日)"
            reason = "パーフェクトオーダー成立、強い上昇トレンド"
        elif adx_val > 30:
            days = 40
            label = "長期(40日)"
            reason = "非常に強いトレンド、長めの保有が有利"
        else:
            days = 30
            label = "中長期(30日)"
            reason = "安定した上昇基調"

    return {
        "days": days,
        "label": label,
        "reason": reason,
        "scores": scores,
    }
