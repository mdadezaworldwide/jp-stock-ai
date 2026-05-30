"""バックテストエンジン（リスク管理・動的損切り/利確対応）"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    INITIAL_CAPITAL, MAX_POSITIONS, POSITION_SIZE, HOLD_DAYS,
    ATR_STOP_LOSS_MULTIPLIER, ATR_TAKE_PROFIT_MULTIPLIER,
)
from risk_manager import RiskManager


def run_backtest(df: pd.DataFrame, signal_threshold: float = 0.5,
                 use_risk_management: bool = True) -> dict:
    """
    シグナルに基づくバックテスト

    ルール:
    - Signal_prob >= signal_threshold の銘柄を買い
    - ATRベース損切り / 利確 or HOLD_DAYS後に自動売却
    - ドローダウン制限・セクター分散・相関チェック
    """
    capital = INITIAL_CAPITAL
    positions = []
    trade_log = []
    daily_equity = []

    risk = RiskManager() if use_risk_management else None

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    last_price = {}
    dates = sorted(df.index.unique())

    for date in dates:
        day_rows = df.loc[[date]]

        # 最新価格を更新
        for _, row in day_rows.iterrows():
            last_price[row["Ticker"]] = {
                "close": float(row["Close"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
            }

        # ドローダウンチェック
        current_equity = capital + sum(
            last_price.get(p["ticker"], {}).get("close", p["buy_price"]) * p["shares"]
            for p in positions
        )
        if risk and not risk.check_drawdown(current_equity):
            # 全ポジション強制決済
            for pos in positions:
                price = last_price.get(pos["ticker"], {}).get("close", pos["buy_price"])
                pnl = (price - pos["buy_price"]) * pos["shares"]
                capital += price * pos["shares"]
                trade_log.append(_trade_entry(pos, date, price, pnl, "ドローダウン強制決済"))
            positions = []
            daily_equity.append({"Date": date, "Equity": capital, "Cash": capital, "Positions": 0})
            continue

        # 売却チェック
        still_holding = []
        for pos in positions:
            days_held = (date - pos["buy_date"]).days
            prices = last_price.get(pos["ticker"], {})
            close = prices.get("close", pos["buy_price"])
            high = prices.get("high", close)
            low = prices.get("low", close)

            sell = False
            reason = ""

            # ATR損切り/利確
            if pos.get("stop_loss") and low <= pos["stop_loss"]:
                sell, reason = True, "損切り"
                close = pos["stop_loss"]  # 損切り価格で約定
            elif pos.get("take_profit") and high >= pos["take_profit"]:
                sell, reason = True, "利確"
                close = pos["take_profit"]
            elif days_held >= HOLD_DAYS:
                sell, reason = True, "保有期限"

            if sell:
                pnl = (close - pos["buy_price"]) * pos["shares"]
                capital += close * pos["shares"]
                trade_log.append(_trade_entry(pos, date, close, pnl, reason))
            else:
                still_holding.append(pos)
        positions = still_holding

        # 買いシグナルチェック
        if len(positions) < MAX_POSITIONS:
            candidates = day_rows[day_rows["Signal_prob"] >= signal_threshold].copy()
            candidates = candidates.sort_values("Signal_prob", ascending=False)

            for _, row in candidates.iterrows():
                if len(positions) >= MAX_POSITIONS:
                    break
                ticker = row["Ticker"]
                if any(p["ticker"] == ticker for p in positions):
                    continue

                # リスクチェック
                if risk:
                    if not risk.check_sector_exposure(positions, ticker):
                        continue
                    if not risk.check_correlation(df, positions, ticker):
                        continue

                buy_price = float(row["Close"])
                atr = float(row.get("ATR_14", buy_price * 0.02))

                # ポジションサイズ
                if risk:
                    shares = risk.get_position_size(capital, atr, buy_price)
                else:
                    invest = INITIAL_CAPITAL * POSITION_SIZE
                    shares = int(invest / buy_price / 100) * 100

                if shares <= 0 or buy_price * shares > capital:
                    continue

                stop_loss = buy_price - atr * ATR_STOP_LOSS_MULTIPLIER
                take_profit = buy_price + atr * ATR_TAKE_PROFIT_MULTIPLIER

                capital -= buy_price * shares
                positions.append({
                    "ticker": ticker,
                    "buy_date": date,
                    "buy_price": buy_price,
                    "shares": shares,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                })

        # 日次評価額
        holdings_value = sum(
            last_price.get(p["ticker"], {}).get("close", p["buy_price"]) * p["shares"]
            for p in positions
        )
        daily_equity.append({
            "Date": date,
            "Equity": capital + holdings_value,
            "Cash": capital,
            "Positions": len(positions),
        })

    return {
        "trades": pd.DataFrame(trade_log),
        "equity": pd.DataFrame(daily_equity).set_index("Date"),
        "final_capital": daily_equity[-1]["Equity"] if daily_equity else INITIAL_CAPITAL,
    }


def _trade_entry(pos, sell_date, sell_price, pnl, reason):
    return {
        "Ticker": pos["ticker"],
        "Buy_date": pos["buy_date"],
        "Buy_price": pos["buy_price"],
        "Sell_date": sell_date,
        "Sell_price": sell_price,
        "Shares": pos["shares"],
        "PnL": pnl,
        "Return": sell_price / pos["buy_price"] - 1,
        "Reason": reason,
    }


def print_backtest_report(result: dict):
    """バックテスト結果のレポート表示"""
    trades = result["trades"]
    equity = result["equity"]
    final = result["final_capital"]

    print("\n" + "=" * 60)
    print("  バックテスト結果")
    print("=" * 60)
    print(f"  初期資金:     {INITIAL_CAPITAL:>12,} 円")
    print(f"  最終資産:     {final:>12,.0f} 円")
    print(f"  損益:         {final - INITIAL_CAPITAL:>12,.0f} 円")
    print(f"  リターン:     {(final / INITIAL_CAPITAL - 1) * 100:>11.2f} %")

    if not trades.empty:
        print(f"\n  トレード数:   {len(trades):>12}")
        print(f"  勝率:         {(trades['PnL'] > 0).mean() * 100:>11.2f} %")
        print(f"  平均リターン: {trades['Return'].mean() * 100:>11.2f} %")
        print(f"  最大利益:     {trades['PnL'].max():>12,.0f} 円")
        print(f"  最大損失:     {trades['PnL'].min():>12,.0f} 円")
        print(f"  プロフィットファクター: {_profit_factor(trades):>8.2f}")

        if not equity.empty:
            max_dd = _max_drawdown(equity["Equity"])
            print(f"  最大ドローダウン: {max_dd * 100:>8.2f} %")

        # 決済理由の内訳
        if "Reason" in trades.columns:
            print(f"\n  決済理由:")
            for reason, count in trades["Reason"].value_counts().items():
                subset = trades[trades["Reason"] == reason]
                win_rate = (subset["PnL"] > 0).mean() * 100
                print(f"    {reason:>12s}: {count:>4d}件 (勝率 {win_rate:.1f}%)")
    else:
        print("  トレードなし")
    print("=" * 60)


def plot_backtest(result: dict, save_path: str = "backtest_result.png"):
    """バックテスト結果のグラフ出力"""
    equity = result["equity"]
    trades = result["trades"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

    axes[0].plot(equity.index, equity["Equity"], label="Total Equity", linewidth=1.5)
    axes[0].axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title("Portfolio Equity Curve", fontsize=14)
    axes[0].set_ylabel("JPY")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(equity.index, equity["Positions"], alpha=0.5, color="orange")
    axes[1].set_title("Active Positions")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    if not trades.empty:
        colors = ["green" if x > 0 else "red" for x in trades["PnL"]]
        axes[2].bar(range(len(trades)), trades["PnL"], color=colors, alpha=0.7)
        axes[2].set_title("Trade PnL")
        axes[2].set_ylabel("JPY")
        axes[2].axhline(y=0, color="black", linewidth=0.5)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(__file__).parent / save_path
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nグラフ保存: {path}")


def _profit_factor(trades: pd.DataFrame) -> float:
    wins = trades[trades["PnL"] > 0]["PnL"].sum()
    losses = abs(trades[trades["PnL"] <= 0]["PnL"].sum())
    return wins / losses if losses > 0 else float("inf")


def _max_drawdown(equity_series: pd.Series) -> float:
    peak = equity_series.expanding().max()
    dd = (equity_series - peak) / peak
    return dd.min()
