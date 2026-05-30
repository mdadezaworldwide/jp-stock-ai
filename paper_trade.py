"""ペーパートレード（仮想売買）"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import INITIAL_CAPITAL, TICKER_NAMES
from risk_manager import RiskManager
from alerts import notify_trade

PAPER_DIR = Path(__file__).parent / "paper_trades"
PAPER_DIR.mkdir(exist_ok=True)
STATE_FILE = PAPER_DIR / "state.json"
TRADE_LOG = PAPER_DIR / "trades.csv"


class PaperTrader:
    """仮想売買トラッカー"""

    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.positions: list[dict] = []
        self.trade_history: list[dict] = []
        self.risk_manager = RiskManager()
        self._load_state()

    def _load_state(self):
        """保存状態を復元"""
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.capital = state.get("capital", INITIAL_CAPITAL)
            self.positions = state.get("positions", [])
            self.risk_manager.peak_equity = state.get("peak_equity", INITIAL_CAPITAL)
            print(f"  ペーパートレード復元: 資金={self.capital:,.0f}円, ポジション={len(self.positions)}件")

    def _save_state(self):
        """状態を保存"""
        state = {
            "capital": self.capital,
            "positions": self.positions,
            "peak_equity": self.risk_manager.peak_equity,
            "updated_at": datetime.now().isoformat(),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def buy(self, ticker: str, price: float, shares: int,
            stop_loss: float = None, take_profit: float = None, signal_prob: float = 0):
        """仮想買い注文"""
        cost = price * shares
        if cost > self.capital:
            print(f"  [SKIP] 資金不足: {ticker} ({cost:,.0f}円 > {self.capital:,.0f}円)")
            return False

        self.capital -= cost
        pos = {
            "ticker": ticker,
            "buy_date": datetime.now().isoformat(),
            "buy_price": price,
            "shares": shares,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "signal_prob": signal_prob,
        }
        self.positions.append(pos)

        name = TICKER_NAMES.get(ticker, ticker)
        print(f"  [BUY] {name}: {price:,.0f}円 x {shares}株 = {cost:,.0f}円")
        notify_trade("買い", ticker, price, shares, f"確率={signal_prob:.1%}")

        self._save_state()
        return True

    def sell(self, ticker: str, price: float, reason: str = ""):
        """仮想売り注文"""
        pos = None
        for p in self.positions:
            if p["ticker"] == ticker:
                pos = p
                break

        if pos is None:
            print(f"  [WARN] {ticker} のポジションなし")
            return

        proceeds = price * pos["shares"]
        pnl = (price - pos["buy_price"]) * pos["shares"]
        ret = price / pos["buy_price"] - 1

        self.capital += proceeds
        self.positions.remove(pos)

        trade = {
            "Ticker": ticker,
            "Buy_date": pos["buy_date"],
            "Buy_price": pos["buy_price"],
            "Sell_date": datetime.now().isoformat(),
            "Sell_price": price,
            "Shares": pos["shares"],
            "PnL": pnl,
            "Return": ret,
            "Reason": reason,
        }
        self.trade_history.append(trade)
        self._log_trade(trade)

        name = TICKER_NAMES.get(ticker, ticker)
        pnl_str = f"+{pnl:,.0f}" if pnl >= 0 else f"{pnl:,.0f}"
        print(f"  [SELL] {name}: {price:,.0f}円 x {pos['shares']}株 = {pnl_str}円 ({ret:+.1%}) [{reason}]")
        notify_trade("売り", ticker, price, pos["shares"], f"{reason} / {pnl_str}円")

        self._save_state()

    def check_exits(self, price_data: dict[str, dict]):
        """損切り・利確チェック"""
        to_sell = []
        for pos in self.positions:
            ticker = pos["ticker"]
            if ticker not in price_data:
                continue

            prices = price_data[ticker]
            should_exit, reason = self.risk_manager.should_exit(
                pos, prices["close"], prices["high"], prices["low"]
            )
            if should_exit:
                to_sell.append((ticker, prices["close"], reason))

        for ticker, price, reason in to_sell:
            self.sell(ticker, price, reason)

    def equity(self, current_prices: dict[str, float]) -> float:
        """現在の評価額"""
        holdings = sum(
            current_prices.get(p["ticker"], p["buy_price"]) * p["shares"]
            for p in self.positions
        )
        return self.capital + holdings

    def report(self, current_prices: dict[str, float] = None):
        """ペーパートレードレポート"""
        if current_prices is None:
            current_prices = {}

        total_equity = self.equity(current_prices)

        print("\n" + "=" * 60)
        print("  ペーパートレード状況")
        print("=" * 60)
        print(f"  現金:         {self.capital:>12,.0f} 円")
        print(f"  評価額:       {total_equity:>12,.0f} 円")
        print(f"  損益:         {total_equity - INITIAL_CAPITAL:>+12,.0f} 円")
        print(f"  リターン:     {(total_equity / INITIAL_CAPITAL - 1) * 100:>+11.2f} %")

        if self.positions:
            print(f"\n  保有ポジション ({len(self.positions)}件):")
            for pos in self.positions:
                name = TICKER_NAMES.get(pos["ticker"], pos["ticker"])
                cur = current_prices.get(pos["ticker"], pos["buy_price"])
                pnl = (cur - pos["buy_price"]) * pos["shares"]
                ret = cur / pos["buy_price"] - 1
                print(f"    {name:>10s}  買値:{pos['buy_price']:>8,.0f}  現在:{cur:>8,.0f}  "
                      f"損益:{pnl:>+8,.0f}円 ({ret:>+6.1%})")

        # 取引履歴サマリー
        if TRADE_LOG.exists():
            trades = pd.read_csv(TRADE_LOG)
            if not trades.empty:
                print(f"\n  取引履歴: {len(trades)}件")
                print(f"  勝率: {(trades['PnL'] > 0).mean()*100:.1f}%")
                print(f"  平均リターン: {trades['Return'].mean()*100:+.2f}%")
                print(f"  累計損益: {trades['PnL'].sum():+,.0f}円")
        print("=" * 60)

    def _log_trade(self, trade: dict):
        """取引をCSVに記録"""
        df = pd.DataFrame([trade])
        if TRADE_LOG.exists():
            df.to_csv(TRADE_LOG, mode="a", header=False, index=False)
        else:
            df.to_csv(TRADE_LOG, index=False)
