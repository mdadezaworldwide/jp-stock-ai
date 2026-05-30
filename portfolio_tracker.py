"""実ポートフォリオ管理 — 買った銘柄を記録し、売り時をAIが判定"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

from config import TICKER_NAMES, ATR_STOP_LOSS_MULTIPLIER, ATR_TAKE_PROFIT_MULTIPLIER

PORTFOLIO_DIR = Path(__file__).parent / "portfolio"
PORTFOLIO_DIR.mkdir(exist_ok=True)
HOLDINGS_FILE = PORTFOLIO_DIR / "holdings.json"
TRADE_HISTORY_FILE = PORTFOLIO_DIR / "trade_history.csv"


def load_holdings() -> list[dict]:
    """保有銘柄を読み込み"""
    if HOLDINGS_FILE.exists():
        with open(HOLDINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_holdings(holdings: list[dict]):
    """保有銘柄を保存"""
    with open(HOLDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(holdings, f, ensure_ascii=False, indent=2)


def add_holding(ticker: str, buy_price: float, shares: int, buy_date: str = None):
    """買い記録を追加"""
    holdings = load_holdings()
    holding = {
        "ticker": ticker,
        "buy_price": buy_price,
        "shares": shares,
        "buy_date": buy_date or datetime.now().strftime("%Y-%m-%d"),
        "added_at": datetime.now().isoformat(),
    }
    holdings.append(holding)
    save_holdings(holdings)
    name = TICKER_NAMES.get(ticker, ticker)
    print(f"  [記録] {name}: {buy_price:,.0f}円 x {shares}株")
    return holding


def remove_holding(ticker: str, sell_price: float = None):
    """売り記録（保有から削除）"""
    holdings = load_holdings()
    removed = None
    remaining = []
    for h in holdings:
        if h["ticker"] == ticker and removed is None:
            removed = h
        else:
            remaining.append(h)

    if removed is None:
        print(f"  [WARN] {ticker} は保有していません")
        return None

    save_holdings(remaining)

    # 売買履歴に記録
    if sell_price:
        pnl = (sell_price - removed["buy_price"]) * removed["shares"]
        ret = sell_price / removed["buy_price"] - 1
        trade = {
            "Ticker": ticker,
            "Buy_date": removed["buy_date"],
            "Buy_price": removed["buy_price"],
            "Sell_date": datetime.now().strftime("%Y-%m-%d"),
            "Sell_price": sell_price,
            "Shares": removed["shares"],
            "PnL": pnl,
            "Return": ret,
        }
        df = pd.DataFrame([trade])
        if TRADE_HISTORY_FILE.exists():
            df.to_csv(TRADE_HISTORY_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(TRADE_HISTORY_FILE, index=False)

        name = TICKER_NAMES.get(ticker, ticker)
        pnl_str = f"+{pnl:,.0f}" if pnl >= 0 else f"{pnl:,.0f}"
        print(f"  [売却] {name}: {sell_price:,.0f}円 ({ret:+.1%}) 損益: {pnl_str}円")

    return removed


def check_sell_signals() -> list[dict]:
    """全保有銘柄の売り判定"""
    holdings = load_holdings()
    if not holdings:
        return []

    # カスタム銘柄名も含める
    from custom_stocks import load_custom_stocks
    all_names = dict(TICKER_NAMES)
    for cs in load_custom_stocks():
        all_names[cs["ticker"]] = cs["name"]

    results = []
    for h in holdings:
        ticker = h["ticker"]
        name = all_names.get(ticker, ticker)
        buy_price = h["buy_price"]
        buy_date = h["buy_date"]

        # 現在の株価を取得
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="60d")
            if hist.empty:
                continue
            current = float(hist["Close"].iloc[-1])
            high = float(hist["High"].iloc[-1])
            low = float(hist["Low"].iloc[-1])
        except Exception:
            continue

        # ATRを計算
        if len(hist) >= 14:
            import ta as ta_lib
            atr = float(ta_lib.volatility.average_true_range(
                hist["High"], hist["Low"], hist["Close"], window=14
            ).iloc[-1])
        else:
            atr = current * 0.02

        # 損益計算
        pnl_pct = current / buy_price - 1
        days_held = (datetime.now() - datetime.strptime(buy_date, "%Y-%m-%d")).days

        # RSI
        if len(hist) >= 14:
            rsi = float(ta_lib.momentum.rsi(hist["Close"], window=14).iloc[-1])
        else:
            rsi = 50.0

        # 移動平均
        close = hist["Close"]
        sma5 = float(close.rolling(5).mean().iloc[-1]) if len(hist) >= 5 else current
        sma20 = float(close.rolling(20).mean().iloc[-1]) if len(hist) >= 20 else current
        sma60 = float(close.rolling(60).mean().iloc[-1]) if len(hist) >= 60 else current

        # 出来高分析
        vol_sma20 = float(hist["Volume"].rolling(20).mean().iloc[-1]) if len(hist) >= 20 else 0
        vol_current = float(hist["Volume"].iloc[-1])
        vol_ratio = vol_current / vol_sma20 if vol_sma20 > 0 else 1.0

        # SMA20からの乖離率
        sma20_dev = (current - sma20) / sma20 if sma20 > 0 else 0

        # ADX（トレンド強度）
        if len(hist) >= 14:
            adx = float(ta_lib.trend.adx(hist["High"], hist["Low"], hist["Close"]).iloc[-1])
        else:
            adx = 20.0

        # ファンダメンタルズチェック
        try:
            info = stock.info or {}
            roe = info.get("returnOnEquity")
            earnings_growth = info.get("earningsGrowth")
            revenue_growth = info.get("revenueGrowth")
            fundamentals_healthy = (
                (roe is not None and roe > 0.05) or
                (earnings_growth is not None and earnings_growth > 0) or
                (revenue_growth is not None and revenue_growth > 0)
            )
        except Exception:
            fundamentals_healthy = False
            roe = None
            earnings_growth = None

        # 損切り・利確ライン
        stop_loss = buy_price - atr * ATR_STOP_LOSS_MULTIPLIER
        take_profit = buy_price + atr * ATR_TAKE_PROFIT_MULTIPLIER

        # === 判定ロジック ===
        action = "保有継続"
        reason = ""
        urgency = "低"  # 低/中/高

        # --- 買い増し判定（売り判定より先にチェック） ---
        is_nanpin = False

        # 買い増し条件1: RSI売られすぎ + ファンダ健全 + 下落幅が限定的
        if rsi <= 30 and fundamentals_healthy and pnl_pct > -0.15:
            is_nanpin = True
            reason_parts = [f"RSI {rsi:.0f}(売られすぎ)"]
            if roe: reason_parts.append(f"ROE {roe:.1%}")
            action = "買い増しチャンス"
            reason = " / ".join(reason_parts) + " / ファンダ健全で反発期待"
            urgency = "中"

        # 買い増し条件2: SMA20から大きく乖離 + 出来高減少（売り枯れ）+ ファンダ健全
        elif sma20_dev < -0.08 and vol_ratio < 0.8 and fundamentals_healthy and pnl_pct > -0.20:
            is_nanpin = True
            action = "買い増しチャンス"
            reason = f"SMA20から{sma20_dev:.1%}乖離 / 出来高減少(売り枯れ) / ファンダ健全"
            urgency = "中"

        # 買い増し条件3: 市場全体の下げに連動 + 個別ファンダ良好 + RSIが低め
        elif pnl_pct < -0.05 and rsi < 40 and fundamentals_healthy and adx < 25:
            is_nanpin = True
            action = "買い増し検討"
            reason = f"RSI {rsi:.0f} / トレンド弱(ADX {adx:.0f}) / ファンダ健全で一時的な下げの可能性"
            urgency = "低"

        # --- 売り判定 ---
        if not is_nanpin:
            # 1. 損切りライン到達
            if current <= stop_loss:
                action = "売り（損切り）"
                reason = f"損切りライン({stop_loss:,.0f}円)を下回りました"
                urgency = "高"

            # 2. 利確ライン到達
            elif current >= take_profit:
                action = "売り（利確）"
                reason = f"利確ライン({take_profit:,.0f}円)に到達しました"
                urgency = "高"

            # 3. RSI過熱
            elif rsi >= 75 and pnl_pct > 0.03:
                action = "売り検討"
                reason = f"RSI {rsi:.0f}で過熱、利益{pnl_pct:+.1%}あり"
                urgency = "中"

            # 4. デッドクロス + 強い下降トレンド
            elif sma5 < sma20 and adx > 30 and pnl_pct < -0.05:
                action = "売り検討（損切り）"
                reason = f"デッドクロス + 強い下降トレンド(ADX {adx:.0f})"
                urgency = "高"

            # 5. デッドクロス（利益あり）
            elif sma5 < sma20 and pnl_pct > 0:
                action = "売り検討"
                reason = "移動平均がデッドクロス、トレンド転換の兆候"
                urgency = "中"

            # 6. 大きな含み損 + ファンダ悪化
            elif pnl_pct <= -0.10 and not fundamentals_healthy:
                action = "売り検討（損切り）"
                reason = f"含み損{pnl_pct:.1%} + 業績悪化"
                urgency = "高"

            # 7. 大きな含み損だがファンダ健全
            elif pnl_pct <= -0.10 and fundamentals_healthy:
                action = "保有継続（様子見）"
                reason = f"含み損{pnl_pct:.1%}だがファンダ健全、安易な損切りは避ける"
                urgency = "中"

            # 8. 長期保有で停滞
            elif days_held > 90 and abs(pnl_pct) < 0.02:
                action = "売り検討"
                reason = f"{days_held}日保有で横ばい（資金効率低下）"
                urgency = "低"

            # 9. 利益 + 上昇トレンド継続
            elif pnl_pct > 0.05 and sma5 > sma20:
                action = "保有継続"
                reason = f"利益{pnl_pct:+.1%}、上昇トレンド継続中"
                urgency = "低"

            else:
                action = "保有継続"
                reason = "特に売りシグナルなし"
                urgency = "低"

        results.append({
            "銘柄": name,
            "ティッカー": ticker,
            "買値": buy_price,
            "現在値": current,
            "損益": pnl_pct,
            "保有日数": days_held,
            "RSI": rsi,
            "損切ライン": stop_loss,
            "利確ライン": take_profit,
            "判定": action,
            "理由": reason,
            "緊急度": urgency,
        })

    return results
