"""リアルタイム学習・分析スケジューラー（全機能統合版）"""

import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_fetcher import fetch_all_data
from features import prepare_features, get_feature_columns
from ensemble import EnsembleModel
from risk_manager import RiskManager
from paper_trade import PaperTrader
from alerts import notify_signals, notify_risk_alert
from config import (
    TICKERS, TICKER_NAMES, RETRAIN_HOUR, SIGNAL_CHECK_MINUTES,
    HOLD_DAYS, TARGET_RETURN,
)

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "realtime.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class RealtimeTrader:
    """リアルタイム学習・シグナル生成・ペーパートレード"""

    def __init__(self):
        self.model: EnsembleModel | None = None
        self.paper = PaperTrader()
        self.risk = RiskManager()
        self._load_existing_model()

    def _load_existing_model(self):
        try:
            self.model = EnsembleModel.load()
            logger.info("アンサンブルモデルを読み込みました")
        except FileNotFoundError:
            logger.info("既存モデルなし — 初回訓練が必要です")

    def train(self):
        """フル訓練"""
        logger.info("=" * 50)
        logger.info("アンサンブル訓練開始")

        raw_data = fetch_all_data()
        df = prepare_features(raw_data)

        self.model = EnsembleModel()
        self.model.train(df)
        self.model.feature_importance()
        self.model.save()

        logger.info(f"訓練完了: {datetime.now()}")

    def generate_signals(self) -> pd.DataFrame:
        """最新データでシグナル生成 + ペーパートレード"""
        if self.model is None:
            logger.warning("モデル未訓練")
            return pd.DataFrame()

        logger.info("シグナル生成中...")
        raw_data = fetch_all_data()
        df = prepare_features(raw_data, include_fundamentals=False,
                              include_sentiment=True, include_market=False)
        df_sig = self.model.predict_signals(df)

        # 各銘柄の最新シグナル
        signals = []
        current_prices = {}
        for ticker in TICKERS:
            td = df_sig[df_sig["Ticker"] == ticker]
            if td.empty:
                continue
            latest = td.iloc[-1]
            current_prices[ticker] = {
                "close": float(latest["Close"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
            }
            atr = float(latest.get("ATR_14", latest["Close"] * 0.02))
            stop = self.risk.calculate_stop_loss(latest["Close"], atr)
            tp = self.risk.calculate_take_profit(latest["Close"], atr)

            signals.append({
                "Ticker": ticker,
                "Close": latest["Close"],
                "Signal_prob": latest["Signal_prob"],
                "Signal": "BUY" if latest["Signal_prob"] >= 0.5 else "HOLD",
                "RSI_14": latest.get("RSI_14"),
                "MACD_hist": latest.get("MACD_hist"),
                "stop_loss": stop,
                "take_profit": tp,
            })

        signals_df = pd.DataFrame(signals)
        self._display_signals(signals_df)

        # ペーパートレード: 損切り/利確チェック
        self.paper.check_exits(current_prices)

        # ペーパートレード: 新規買い
        buy_signals = signals_df[signals_df["Signal"] == "BUY"]
        for _, row in buy_signals.iterrows():
            ticker = row["Ticker"]
            if any(p["ticker"] == ticker for p in self.paper.positions):
                continue
            if not self.risk.check_sector_exposure(self.paper.positions, ticker):
                continue

            atr = row["Close"] * 0.02
            shares = self.risk.get_position_size(self.paper.capital, atr, row["Close"])
            if shares > 0:
                self.paper.buy(
                    ticker, row["Close"], shares,
                    stop_loss=row["stop_loss"],
                    take_profit=row["take_profit"],
                    signal_prob=row["Signal_prob"],
                )

        # 通知
        notify_signals(signals_df)

        # レポート
        price_dict = {t: p["close"] for t, p in current_prices.items()}
        self.paper.report(price_dict)

        # シグナルログ
        self._log_signals(signals_df)

        return signals_df

    def _display_signals(self, df: pd.DataFrame):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  シグナル ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
        logger.info("=" * 70)
        logger.info(f"  {'銘柄':>10s} {'終値':>10s} {'確率':>8s} {'RSI':>6s} {'損切':>8s} {'利確':>8s} {'判定':>8s}")
        logger.info("-" * 70)

        for _, row in df.iterrows():
            name = TICKER_NAMES.get(row["Ticker"], row["Ticker"])
            mark = ">>> BUY" if row["Signal"] == "BUY" else "    ---"
            rsi = f"{row['RSI_14']:.1f}" if pd.notna(row.get("RSI_14")) else "N/A"
            logger.info(
                f"  {name:>10s} {row['Close']:>10,.0f} {row['Signal_prob']:>8.3f} "
                f"{rsi:>6s} {row['stop_loss']:>8,.0f} {row['take_profit']:>8,.0f} {mark}"
            )
        logger.info("=" * 70)

    def _log_signals(self, df: pd.DataFrame):
        log_path = LOG_DIR / "signal_history.csv"
        df = df.copy()
        df["Timestamp"] = datetime.now().isoformat()
        if log_path.exists():
            df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            df.to_csv(log_path, index=False)

    def run_scheduler(self):
        """メインループ"""
        logger.info("=" * 50)
        logger.info("リアルタイムトレーダー起動")
        logger.info(f"  シグナルチェック: {SIGNAL_CHECK_MINUTES}分間隔")
        logger.info(f"  再訓練: 毎日 {RETRAIN_HOUR}:00")
        logger.info("  Ctrl+C で停止")
        logger.info("=" * 50)

        if self.model is None:
            self.train()

        last_retrain_date = ""

        while True:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            # 毎日再訓練
            if now.hour >= RETRAIN_HOUR and last_retrain_date != today:
                try:
                    self.train()
                    last_retrain_date = today
                except Exception as e:
                    logger.error(f"再訓練エラー: {e}")

            # シグナル生成（9:00-16:00）
            if 9 <= now.hour <= 16:
                try:
                    self.generate_signals()
                except Exception as e:
                    logger.error(f"シグナル生成エラー: {e}")
            else:
                logger.info(f"市場時間外 ({now.strftime('%H:%M')})")

            logger.info(f"次回チェック: {SIGNAL_CHECK_MINUTES}分後")
            time.sleep(SIGNAL_CHECK_MINUTES * 60)


def main():
    trader = RealtimeTrader()
    trader.run_scheduler()


if __name__ == "__main__":
    main()
