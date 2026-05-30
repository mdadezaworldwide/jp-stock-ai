"""リアルタイムアラート監視システム

使い方:
    python alert_monitor.py              - デイトレアラート監視（引け後にシグナル→翌朝通知）
    python alert_monitor.py --realtime   - リアルタイム監視（kabuステーションAPI必要）
"""

import time
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import TICKERS, TICKER_NAMES
from data_fetcher import fetch_all_data
from features import add_technical_features, get_feature_columns
from ensemble import EnsembleModel
from alerts import send_line, send_discord

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "alert_monitor.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def scan_daily_signals(top_n: int = 10) -> list[dict]:
    """全銘柄をスキャンして短期BUYシグナルを検出"""
    logger.info("=== デイトレシグナルスキャン開始 ===")

    # 1日モデル読み込み
    model_path = Path(__file__).parent / "models" / "ensemble_1d.pkl"
    if not model_path.exists():
        logger.error("1日モデルが見つかりません。train_multi.py を実行してください")
        return []

    model = EnsembleModel.load(model_path)
    raw_data = fetch_all_data()

    # テクニカル指標生成
    frames = []
    for ticker, group in raw_data.groupby("Ticker"):
        group = group.sort_index()
        group = add_technical_features(group)
        frames.append(group)
    df = pd.concat(frames).dropna()

    # 予測
    df_sig = model.predict_signals(df)

    # 各銘柄の最新シグナルを取得
    alerts = []
    for ticker in df_sig["Ticker"].unique():
        td = df_sig[df_sig["Ticker"] == ticker]
        if td.empty:
            continue
        latest = td.iloc[-1]
        prob = latest["Signal_prob"]

        if prob >= 0.45:  # 45%以上なら候補に
            name = TICKER_NAMES.get(ticker, ticker)
            rsi = latest.get("RSI_14", 50)
            macd = latest.get("MACD_hist", 0)
            close = latest["Close"]

            # 緊急度判定
            if prob >= 0.6:
                urgency = "高"
            elif prob >= 0.5:
                urgency = "中"
            else:
                urgency = "低"

            alerts.append({
                "銘柄": name,
                "ティッカー": ticker,
                "終値": close,
                "シグナル確率": prob,
                "RSI": rsi,
                "MACD": macd,
                "緊急度": urgency,
            })

    # 確率順にソート
    alerts.sort(key=lambda x: x["シグナル確率"], reverse=True)
    alerts = alerts[:top_n]

    logger.info(f"検出: {len(alerts)}銘柄")
    return alerts


def send_alert(alerts: list[dict]):
    """アラートをLINE/Discordに送信"""
    if not alerts:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n[デイトレAIアラート] {now}"]
    lines.append(f"翌日+1%以上が期待される銘柄 TOP{len(alerts)}")
    lines.append("-" * 35)

    for a in alerts:
        mark = "★" if a["緊急度"] == "高" else "◎" if a["緊急度"] == "中" else "○"
        lines.append(
            f"{mark} {a['銘柄']} ({a['ティッカー']})"
            f"\n   終値:{a['終値']:,.0f}円 確率:{a['シグナル確率']:.1%}"
            f" RSI:{a['RSI']:.0f}"
        )

    lines.append(f"\n★=確率60%+ ◎=50%+ ○=45%+")
    message = "\n".join(lines)

    logger.info(message)
    send_line(message)
    send_discord(message)

    # ログ保存
    df = pd.DataFrame(alerts)
    df["Timestamp"] = now
    log_path = LOG_DIR / "daily_alerts.csv"
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


def run_daily_monitor():
    """毎日の監視ループ（15:30以降にスキャン→翌朝通知）"""
    logger.info("=" * 50)
    logger.info("デイトレアラート監視 起動")
    logger.info("  15:30以降にスキャン → アラート送信")
    logger.info("  Ctrl+C で停止")
    logger.info("=" * 50)

    last_scan_date = ""

    while True:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # 15:30以降 & 今日まだスキャンしていない
        if now.hour >= 15 and now.minute >= 30 and last_scan_date != today:
            try:
                alerts = scan_daily_signals(top_n=10)
                send_alert(alerts)
                last_scan_date = today
                logger.info("スキャン完了。次回は翌営業日15:30以降")
            except Exception as e:
                logger.error(f"スキャンエラー: {e}")

        time.sleep(300)  # 5分おきにチェック


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        # 1回だけスキャン
        alerts = scan_daily_signals()
        send_alert(alerts)
    else:
        run_daily_monitor()
