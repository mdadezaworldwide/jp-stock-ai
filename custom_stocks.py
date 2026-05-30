"""カスタム銘柄管理 — ユーザーが自由に銘柄を追加・分析"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
import ta

CUSTOM_FILE = Path(__file__).parent / "custom_stocks.json"


def load_custom_stocks() -> list[dict]:
    """カスタム銘柄リストを読み込み"""
    if CUSTOM_FILE.exists():
        with open(CUSTOM_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_custom_stocks(stocks: list[dict]):
    """カスタム銘柄リストを保存"""
    with open(CUSTOM_FILE, "w", encoding="utf-8") as f:
        json.dump(stocks, f, ensure_ascii=False, indent=2)


def add_custom_stock(ticker: str, name: str = ""):
    """カスタム銘柄を追加"""
    stocks = load_custom_stocks()

    # 重複チェック
    if any(s["ticker"] == ticker for s in stocks):
        return False, "既に追加済みです"

    # 銘柄が存在するか確認
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        if not name:
            name = info.get("shortName", info.get("longName", ticker))
    except Exception:
        pass

    if not name:
        name = ticker

    stocks.append({"ticker": ticker, "name": name})
    save_custom_stocks(stocks)
    return True, f"{name} ({ticker}) を追加しました"


def remove_custom_stock(ticker: str):
    """カスタム銘柄を削除"""
    stocks = load_custom_stocks()
    stocks = [s for s in stocks if s["ticker"] != ticker]
    save_custom_stocks(stocks)


def analyze_custom_stock(ticker: str) -> dict:
    """カスタム銘柄を分析（モデル不要、テクニカル指標ベース）"""
    try:
        hist = yf.download(ticker, period="1y", progress=False)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if hist.empty or len(hist) < 20:
            return {"error": "データ取得失敗"}
    except Exception as e:
        return {"error": str(e)}

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    current = float(close.iloc[-1])

    # テクニカル指標
    rsi = float(ta.momentum.rsi(close, window=14).iloc[-1]) if len(hist) >= 14 else np.nan
    macd_obj = ta.trend.MACD(close)
    macd_hist = float(macd_obj.macd_diff().iloc[-1]) if len(hist) >= 26 else np.nan
    adx = float(ta.trend.adx(high, low, close).iloc[-1]) if len(hist) >= 14 else np.nan
    atr = float(ta.volatility.average_true_range(high, low, close).iloc[-1]) if len(hist) >= 14 else current * 0.02

    # 移動平均
    sma5 = float(close.rolling(5).mean().iloc[-1]) if len(hist) >= 5 else current
    sma20 = float(close.rolling(20).mean().iloc[-1]) if len(hist) >= 20 else current
    sma60 = float(close.rolling(60).mean().iloc[-1]) if len(hist) >= 60 else current

    # ボリンジャーバンド
    bb = ta.volatility.BollingerBands(close)
    bb_upper = float(bb.bollinger_hband().iloc[-1])
    bb_lower = float(bb.bollinger_lband().iloc[-1])

    # リターン
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(hist) >= 6 else np.nan
    ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1) if len(hist) >= 21 else np.nan

    # トレンド判定
    perfect_order = sma5 > sma20 > sma60
    if perfect_order:
        trend = "強い上昇トレンド"
    elif sma5 > sma20:
        trend = "上昇トレンド"
    elif sma5 < sma20 < sma60:
        trend = "強い下降トレンド"
    elif sma5 < sma20:
        trend = "下降トレンド"
    else:
        trend = "横ばい"

    # RSI判定
    if pd.notna(rsi):
        if rsi >= 70:
            rsi_label = "買われすぎ"
        elif rsi <= 30:
            rsi_label = "売られすぎ"
        elif rsi <= 40:
            rsi_label = "やや売られすぎ"
        elif rsi >= 60:
            rsi_label = "やや買われすぎ"
        else:
            rsi_label = "普通"
    else:
        rsi_label = "-"

    # MACD判定
    if pd.notna(macd_hist):
        if macd_hist > 0:
            macd_label = "上昇の勢い加速" if macd_hist > 0.5 else "上昇の勢い鈍化"
        else:
            macd_label = "下落の勢い加速" if macd_hist < -0.5 else "下落止まりつつある"
    else:
        macd_label = "-"

    # 総合シグナル（テクニカルスコア）
    score = 0
    if pd.notna(rsi):
        if rsi < 40:
            score += 1
        if rsi < 30:
            score += 1
        if rsi > 70:
            score -= 2
    if pd.notna(macd_hist) and macd_hist > 0:
        score += 1
    if sma5 > sma20:
        score += 1
    if perfect_order:
        score += 1
    if pd.notna(adx) and adx > 25:
        score += 1
    if pd.notna(ret_5d) and ret_5d > 0:
        score += 1

    # スコア → 判定
    if score >= 5:
        signal = "強い買い"
        signal_color = "green"
    elif score >= 3:
        signal = "買い"
        signal_color = "lightgreen"
    elif score >= 1:
        signal = "やや買い"
        signal_color = "yellow"
    elif score >= -1:
        signal = "中立"
        signal_color = "gray"
    else:
        signal = "売り"
        signal_color = "red"

    # 保有期間アドバイス
    from hold_advisor import advise_hold_period
    # hold_advisorにはDataFrameを渡す必要がある
    hist_with_indicators = hist.copy()
    hist_with_indicators["ADX"] = ta.trend.adx(high, low, close)
    hist_with_indicators["RSI_14"] = ta.momentum.rsi(close, window=14)
    hold = advise_hold_period(hist_with_indicators)

    # ファンダメンタルズ
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        fundamentals = {
            "PER": info.get("trailingPE"),
            "PBR": info.get("priceToBook"),
            "ROE": info.get("returnOnEquity"),
            "配当利回り": info.get("dividendYield"),
            "時価総額": info.get("marketCap"),
        }
    except Exception:
        fundamentals = {}

    return {
        "current": current,
        "rsi": rsi,
        "rsi_label": rsi_label,
        "macd_hist": macd_hist,
        "macd_label": macd_label,
        "adx": adx,
        "atr": atr,
        "trend": trend,
        "sma5": sma5,
        "sma20": sma20,
        "sma60": sma60,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "signal": signal,
        "score": score,
        "hold_label": hold["label"],
        "hold_reason": hold["reason"],
        "fundamentals": fundamentals,
        "hist": hist,
    }
