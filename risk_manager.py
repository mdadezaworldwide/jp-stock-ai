"""リスク管理モジュール"""

import pandas as pd
import numpy as np
from config import (
    MAX_DRAWDOWN_LIMIT, MAX_SECTOR_EXPOSURE, CORRELATION_THRESHOLD,
    ATR_STOP_LOSS_MULTIPLIER, ATR_TAKE_PROFIT_MULTIPLIER,
    TICKER_SECTORS, INITIAL_CAPITAL,
)


class RiskManager:
    """ポートフォリオのリスク管理"""

    def __init__(self):
        self.peak_equity = INITIAL_CAPITAL
        self.trading_halted = False
        self.halt_reason = ""

    def check_drawdown(self, current_equity: float) -> bool:
        """ドローダウン制限チェック。Trueなら取引継続OK"""
        self.peak_equity = max(self.peak_equity, current_equity)
        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown >= MAX_DRAWDOWN_LIMIT:
            self.trading_halted = True
            self.halt_reason = f"ドローダウン {drawdown:.1%} が制限 {MAX_DRAWDOWN_LIMIT:.1%} を超過"
            return False
        return True

    def check_sector_exposure(self, positions: list[dict], new_ticker: str) -> bool:
        """セクター集中度チェック。Trueなら追加OK"""
        if not positions:
            return True

        new_sector = TICKER_SECTORS.get(new_ticker, "不明")
        sector_count = {}

        for pos in positions:
            sector = TICKER_SECTORS.get(pos["ticker"], "不明")
            sector_count[sector] = sector_count.get(sector, 0) + 1

        # 追加後の比率をチェック
        sector_count[new_sector] = sector_count.get(new_sector, 0) + 1
        total = sum(sector_count.values())

        if sector_count[new_sector] / total > MAX_SECTOR_EXPOSURE:
            return False
        return True

    def check_correlation(self, df: pd.DataFrame, positions: list[dict],
                          new_ticker: str, lookback: int = 60) -> bool:
        """相関チェック。既存ポジションと高相関の銘柄を除外"""
        if not positions:
            return True

        held_tickers = [p["ticker"] for p in positions]

        for held in held_tickers:
            held_data = df[df["Ticker"] == held]["Close"].tail(lookback)
            new_data = df[df["Ticker"] == new_ticker]["Close"].tail(lookback)

            if len(held_data) < 20 or len(new_data) < 20:
                continue

            # 日次リターンの相関
            held_ret = held_data.pct_change().dropna()
            new_ret = new_data.pct_change().dropna()

            min_len = min(len(held_ret), len(new_ret))
            if min_len < 10:
                continue

            corr = np.corrcoef(
                held_ret.values[-min_len:],
                new_ret.values[-min_len:]
            )[0, 1]

            if abs(corr) > CORRELATION_THRESHOLD:
                return False

        return True

    def calculate_stop_loss(self, buy_price: float, atr: float) -> float:
        """ATRベースの損切りライン"""
        return buy_price - atr * ATR_STOP_LOSS_MULTIPLIER

    def calculate_take_profit(self, buy_price: float, atr: float) -> float:
        """ATRベースの利確ライン"""
        return buy_price + atr * ATR_TAKE_PROFIT_MULTIPLIER

    def should_exit(self, pos: dict, current_price: float, current_high: float,
                    current_low: float) -> tuple[bool, str]:
        """ポジションを閉じるべきか判定"""
        stop_loss = pos.get("stop_loss")
        take_profit = pos.get("take_profit")

        if stop_loss and current_low <= stop_loss:
            return True, "損切り"
        if take_profit and current_high >= take_profit:
            return True, "利確"
        return False, ""

    def get_position_size(self, capital: float, atr: float, price: float) -> int:
        """ATRベースのポジションサイズ（リスク額固定方式）"""
        if price <= 0:
            return 0

        # 1トレードのリスク額 = 資本の2%
        risk_per_trade = capital * 0.02
        # 1株あたりのリスク = ATR × 損切り倍率
        risk_per_share = atr * ATR_STOP_LOSS_MULTIPLIER

        if risk_per_share > 0:
            shares = risk_per_trade / risk_per_share
            shares = int(shares / 100) * 100
        else:
            shares = 100

        # 資金上限: 最大40%まで（100株が最低ロット）
        max_cost = capital * 0.4
        if price * 100 <= max_cost:
            max_shares = int(max_cost / price / 100) * 100
        else:
            return 0  # 100株すら買えない

        shares = max(min(shares, max_shares), 100)

        # 実際に買えるか
        if price * shares > capital:
            shares = int(capital / price / 100) * 100

        return shares

    def reset(self):
        """リスク管理状態をリセット"""
        self.peak_equity = INITIAL_CAPITAL
        self.trading_halted = False
        self.halt_reason = ""

    def status(self) -> dict:
        """現在のリスク状態"""
        return {
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "peak_equity": self.peak_equity,
            "drawdown_limit": MAX_DRAWDOWN_LIMIT,
        }
