"""リアルタイム急騰検知スキャナー

kabuステーションAPI（auカブコム証券）でリアルタイム株価を監視し、
急騰初動・出来高急増を検知してアラートを送信

準備:
1. auカブコム証券の口座開設
2. kabuステーションのインストール・起動
3. 環境変数設定:
   set KABU_API_PASSWORD=あなたのAPIパスワード

使い方:
    python realtime_scanner.py
"""

import os
import json
import time
import urllib.request
import urllib.parse
import logging
from datetime import datetime
from pathlib import Path

from config import TICKERS, TICKER_NAMES
from alerts import send_line, send_discord

KABU_API_PASSWORD = os.environ.get("KABU_API_PASSWORD", "")
KABU_BASE_URL = "http://localhost:18080/kabusapi"  # kabuステーションのローカルAPI

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "realtime_scanner.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class KabuStationClient:
    """kabuステーションAPI クライアント"""

    def __init__(self):
        self.token = ""

    def login(self) -> bool:
        """APIトークン取得"""
        if not KABU_API_PASSWORD:
            logger.error("KABU_API_PASSWORD が未設定です")
            return False

        try:
            data = json.dumps({"APIPassword": KABU_API_PASSWORD}).encode()
            req = urllib.request.Request(
                f"{KABU_BASE_URL}/token",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
                self.token = result.get("Token", "")
                logger.info(f"kabuステーション ログイン成功")
                return bool(self.token)
        except Exception as e:
            logger.error(f"kabuステーション ログイン失敗: {e}")
            return False

    def get_price(self, ticker: str) -> dict:
        """リアルタイム株価を取得"""
        code = ticker.replace(".T", "")
        try:
            req = urllib.request.Request(
                f"{KABU_BASE_URL}/board/{code}@1",
                headers={"X-API-KEY": self.token},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return {}

    def get_board(self, ticker: str) -> dict:
        """板情報を取得"""
        return self.get_price(ticker)  # boardエンドポイントで板情報も含まれる


class RealtimeScanner:
    """リアルタイム急騰検知スキャナー"""

    def __init__(self):
        self.client = KabuStationClient()
        self.prev_prices: dict[str, float] = {}
        self.prev_volumes: dict[str, float] = {}
        self.alerted_today: set[str] = set()

    def scan_all(self) -> list[dict]:
        """全銘柄をスキャンして急騰・出来高急増を検知"""
        alerts = []

        for ticker in TICKERS:
            data = self.client.get_price(ticker)
            if not data:
                continue

            current = data.get("CurrentPrice")
            volume = data.get("TradingVolume")
            prev_close = data.get("PreviousClose")
            high = data.get("HighPrice")
            low = data.get("LowPrice")

            if not current or not prev_close:
                continue

            name = TICKER_NAMES.get(ticker, ticker)
            change_pct = (current - prev_close) / prev_close

            # === 検知条件 ===

            # 1. 前日比+2%以上の急騰
            if change_pct >= 0.02 and ticker not in self.alerted_today:
                alerts.append({
                    "type": "急騰",
                    "銘柄": name,
                    "ティッカー": ticker,
                    "現在値": current,
                    "前日比": change_pct,
                    "理由": f"前日比{change_pct:+.1%}の急騰",
                })
                self.alerted_today.add(ticker)

            # 2. 直近5分で+1%以上の急上昇
            prev = self.prev_prices.get(ticker)
            if prev and current > prev:
                short_change = (current - prev) / prev
                if short_change >= 0.01 and ticker not in self.alerted_today:
                    alerts.append({
                        "type": "急上昇",
                        "銘柄": name,
                        "ティッカー": ticker,
                        "現在値": current,
                        "前日比": change_pct,
                        "理由": f"直近で{short_change:+.1%}急上昇",
                    })
                    self.alerted_today.add(ticker)

            # 3. 出来高急増（前回比2倍以上）
            prev_vol = self.prev_volumes.get(ticker)
            if prev_vol and volume and prev_vol > 0:
                vol_ratio = volume / prev_vol
                if vol_ratio >= 2.0 and change_pct > 0 and ticker not in self.alerted_today:
                    alerts.append({
                        "type": "出来高急増",
                        "銘柄": name,
                        "ティッカー": ticker,
                        "現在値": current,
                        "前日比": change_pct,
                        "理由": f"出来高{vol_ratio:.1f}倍に急増",
                    })
                    self.alerted_today.add(ticker)

            # 4. 板情報：大口買い検知
            buy1_qty = data.get("Buy1", {}).get("Qty", 0) if isinstance(data.get("Buy1"), dict) else 0
            sell1_qty = data.get("Sell1", {}).get("Qty", 0) if isinstance(data.get("Sell1"), dict) else 0
            if buy1_qty > 0 and sell1_qty > 0:
                buy_sell_ratio = buy1_qty / sell1_qty
                if buy_sell_ratio >= 3.0 and ticker not in self.alerted_today:
                    alerts.append({
                        "type": "大口買い",
                        "銘柄": name,
                        "ティッカー": ticker,
                        "現在値": current,
                        "前日比": change_pct,
                        "理由": f"買い板が売り板の{buy_sell_ratio:.1f}倍（大口買い集め）",
                    })
                    self.alerted_today.add(ticker)

            # 価格・出来高を記録
            self.prev_prices[ticker] = current
            if volume:
                self.prev_volumes[ticker] = volume

        return alerts

    def send_alerts(self, alerts: list[dict]):
        """アラートを送信"""
        if not alerts:
            return

        now = datetime.now().strftime("%H:%M")
        lines = [f"\n[リアルタイムアラート] {now}"]
        lines.append("-" * 30)

        for a in alerts:
            lines.append(
                f"[{a['type']}] {a['銘柄']}"
                f"\n  {a['現在値']:,.0f}円 (前日比{a['前日比']:+.1%})"
                f"\n  {a['理由']}"
            )

        message = "\n".join(lines)
        logger.info(message)
        send_line(message)
        send_discord(message)

    def run(self, interval_sec: int = 30):
        """メインループ"""
        if not self.client.login():
            logger.error("kabuステーションに接続できません")
            logger.info("1. kabuステーションを起動してください")
            logger.info("2. set KABU_API_PASSWORD=あなたのパスワード")
            return

        logger.info("=" * 50)
        logger.info("リアルタイムスキャナー起動")
        logger.info(f"  監視銘柄数: {len(TICKERS)}")
        logger.info(f"  スキャン間隔: {interval_sec}秒")
        logger.info("  検知: 急騰 / 急上昇 / 出来高急増 / 大口買い")
        logger.info("=" * 50)

        while True:
            now = datetime.now()
            # 市場時間のみ（9:00-15:00）
            if 9 <= now.hour < 15 or (now.hour == 15 and now.minute == 0):
                alerts = self.scan_all()
                if alerts:
                    self.send_alerts(alerts)

                # 15時にアラート済みリセット
                if now.hour >= 15:
                    self.alerted_today.clear()
            else:
                if now.hour == 8 and now.minute == 59:
                    self.alerted_today.clear()
                    logger.info("市場開場前リセット")

            time.sleep(interval_sec)


if __name__ == "__main__":
    scanner = RealtimeScanner()
    scanner.run()
