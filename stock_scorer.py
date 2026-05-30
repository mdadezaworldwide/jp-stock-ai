"""銘柄スコアリング — テクニカル・ファンダメンタルズを統一スコアで評価"""

import pandas as pd
import numpy as np
from safe_yf import get_info


def calc_technical_score(row) -> int:
    """テクニカル指標から-10〜+10のスコアを算出"""
    score = 0

    # RSI（売られすぎで加点、買われすぎで減点）
    rsi = row.get("RSI_14")
    if pd.notna(rsi) and rsi != -999:
        if rsi <= 25: score += 3
        elif rsi <= 35: score += 2
        elif rsi <= 45: score += 1
        elif rsi >= 75: score -= 3
        elif rsi >= 65: score -= 1

    # MACD（プラスで加点）
    macd = row.get("MACD_hist")
    if pd.notna(macd) and macd != -999:
        if macd > 1: score += 2
        elif macd > 0: score += 1
        elif macd < -1: score -= 2
        elif macd < 0: score -= 1

    # 移動平均の並び
    sma5 = row.get("SMA_5")
    sma20 = row.get("SMA_20")
    sma60 = row.get("SMA_60")
    close = row.get("Close")
    if all(pd.notna(v) and v != -999 for v in [sma5, sma20, sma60, close]):
        if sma5 > sma20 > sma60:
            score += 3  # パーフェクトオーダー
        elif sma5 > sma20:
            score += 1
        elif sma5 < sma20 < sma60:
            score -= 3  # 逆パーフェクトオーダー
        elif sma5 < sma20:
            score -= 1

    # ADX（トレンド強度）
    adx = row.get("ADX")
    if pd.notna(adx) and adx != -999:
        if adx > 30: score += 1  # 強いトレンド
        elif adx < 15: score -= 1  # トレンドなし

    # 出来高
    vol_ratio = row.get("Volume_ratio")
    if pd.notna(vol_ratio) and vol_ratio != -999:
        if vol_ratio > 2: score += 1  # 出来高急増

    return max(-10, min(10, score))


def calc_fundamental_score(ticker: str) -> int:
    """ファンダメンタルズから-10〜+10のスコアを算出"""
    info = get_info(ticker)
    if not info:
        return 0

    score = 0

    # PER
    per = info.get("trailingPE")
    if per is not None:
        if per < 10: score += 2
        elif per < 15: score += 1
        elif per > 40: score -= 2
        elif per > 25: score -= 1

    # PBR
    pbr = info.get("priceToBook")
    if pbr is not None:
        if pbr < 1.0: score += 2
        elif pbr < 1.5: score += 1
        elif pbr > 5: score -= 1

    # ROE
    roe = info.get("returnOnEquity")
    if roe is not None:
        if roe > 0.15: score += 2
        elif roe > 0.08: score += 1
        elif roe < 0: score -= 2

    # 売上成長
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        if rev_growth > 0.15: score += 2
        elif rev_growth > 0.05: score += 1
        elif rev_growth < -0.05: score -= 1

    # 利益成長
    earn_growth = info.get("earningsGrowth")
    if earn_growth is not None:
        if earn_growth > 0.20: score += 2
        elif earn_growth > 0.05: score += 1
        elif earn_growth < -0.10: score -= 2

    # 配当
    div_yield = info.get("dividendYield")
    if div_yield is not None:
        if div_yield > 0.04: score += 1

    # アナリスト推奨
    rec = info.get("recommendationKey")
    if rec == "strongBuy": score += 2
    elif rec == "buy": score += 1
    elif rec == "sell": score -= 1
    elif rec == "strongSell": score -= 2

    # 目標株価との乖離
    target = info.get("targetMeanPrice")
    current = info.get("currentPrice")
    if target and current and current > 0:
        upside = (target - current) / current
        if upside > 0.20: score += 2
        elif upside > 0.10: score += 1
        elif upside < -0.10: score -= 1

    return max(-10, min(10, score))


def score_to_label(score: int) -> str:
    """スコアを文字ラベルに変換"""
    if score >= 7: return "非常に強い"
    elif score >= 4: return "強い"
    elif score >= 2: return "やや強い"
    elif score >= -1: return "中立"
    elif score >= -3: return "やや弱い"
    elif score >= -6: return "弱い"
    else: return "非常に弱い"


def get_hold_recommendation(row, selected_days: int) -> tuple[str, str]:
    """選択された保有期間に整合した推奨を返す"""
    rsi = row.get("RSI_14", 50)
    macd = row.get("MACD_hist", 0)
    adx = row.get("ADX", 20)
    signal_prob = row.get("Signal_prob", 0)

    if pd.isna(rsi) or rsi == -999: rsi = 50
    if pd.isna(macd) or macd == -999: macd = 0
    if pd.isna(adx) or adx == -999: adx = 20

    # 選択期間に応じた推奨理由
    if selected_days <= 7:
        # 短期
        if rsi < 30:
            return f"短期({selected_days}日)", "売られすぎからの反発狙い"
        elif macd > 0 and adx > 20:
            return f"短期({selected_days}日)", "モメンタム良好、短期で利確狙い"
        elif signal_prob >= 0.5:
            return f"短期({selected_days}日)", "シグナル確率が高く短期で上昇期待"
        else:
            return f"短期({selected_days}日)", "短期的な値動きに注目"
    elif selected_days <= 20:
        # 中期
        sma5 = row.get("SMA_5", 0)
        sma20 = row.get("SMA_20", 0)
        if pd.notna(sma5) and pd.notna(sma20) and sma5 != -999 and sma20 != -999 and sma5 > sma20:
            return f"中期({selected_days}日)", "上昇トレンド継続中、勢いに乗る"
        elif rsi < 40:
            return f"中期({selected_days}日)", "調整局面、回復を待ってリターン狙い"
        else:
            return f"中期({selected_days}日)", "中期的な値動きを見極める"
    else:
        # 長期
        sma5 = row.get("SMA_5", 0)
        sma20 = row.get("SMA_20", 0)
        sma60 = row.get("SMA_60", 0)
        if all(pd.notna(v) and v != -999 for v in [sma5, sma20, sma60]) and sma5 > sma20 > sma60:
            return f"長期({selected_days}日)", "パーフェクトオーダー、強い上昇トレンド"
        elif adx > 25:
            return f"長期({selected_days}日)", "トレンドが明確、長期保有で利益拡大"
        else:
            return f"長期({selected_days}日)", "長期的な成長に期待"
