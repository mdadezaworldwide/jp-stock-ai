"""LINE / Discord 通知モジュール"""

import json
import urllib.request
import urllib.parse
from datetime import datetime

import pandas as pd

from config import LINE_NOTIFY_TOKEN, DISCORD_WEBHOOK_URL, TICKER_NAMES


def send_line(message: str) -> bool:
    """LINE Notifyで通知"""
    if not LINE_NOTIFY_TOKEN:
        return False
    try:
        data = urllib.parse.urlencode({"message": message}).encode()
        req = urllib.request.Request(
            "https://notify-api.line.me/api/notify",
            data=data,
            headers={"Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  [WARN] LINE通知失敗: {e}")
        return False


def send_discord(content: str, embeds: list[dict] = None) -> bool:
    """Discord Webhookで通知"""
    if not DISCORD_WEBHOOK_URL:
        return False
    try:
        payload = {"content": content}
        if embeds:
            payload["embeds"] = embeds
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as e:
        print(f"  [WARN] Discord通知失敗: {e}")
        return False


def notify_signals(signals_df: pd.DataFrame):
    """シグナルを全チャンネルに通知"""
    buy_signals = signals_df[signals_df["Signal"] == "BUY"]
    if buy_signals.empty:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # メッセージ構築
    lines = [f"\n[株AI シグナル] {now}"]
    lines.append("-" * 30)

    for _, row in buy_signals.iterrows():
        name = TICKER_NAMES.get(row["Ticker"], row["Ticker"])
        lines.append(
            f"BUY: {name} ({row['Ticker']})"
            f"\n  終値: {row['Close']:,.0f}円"
            f"  確率: {row['Signal_prob']:.1%}"
        )
        if "stop_loss" in row and pd.notna(row.get("stop_loss")):
            lines.append(
                f"  損切: {row['stop_loss']:,.0f}円"
                f"  利確: {row['take_profit']:,.0f}円"
            )
    lines.append(f"\n買いシグナル: {len(buy_signals)}銘柄")

    message = "\n".join(lines)

    # LINE
    if LINE_NOTIFY_TOKEN:
        if send_line(message):
            print("  LINE通知: 送信完了")

    # Discord
    if DISCORD_WEBHOOK_URL:
        embeds = [{
            "title": f"株AI シグナル ({now})",
            "color": 0x00ff00,
            "fields": [
                {
                    "name": TICKER_NAMES.get(row["Ticker"], row["Ticker"]),
                    "value": f"終値: {row['Close']:,.0f}円 / 確率: {row['Signal_prob']:.1%}",
                    "inline": True,
                }
                for _, row in buy_signals.iterrows()
            ],
        }]
        if send_discord("", embeds=embeds):
            print("  Discord通知: 送信完了")

    if not LINE_NOTIFY_TOKEN and not DISCORD_WEBHOOK_URL:
        print("  [INFO] 通知先未設定（LINE_NOTIFY_TOKEN or DISCORD_WEBHOOK_URL を設定してください）")


def notify_risk_alert(message: str):
    """リスクアラートを通知"""
    alert = f"\n[株AI リスクアラート]\n{message}"
    send_line(alert)
    send_discord(alert)


def notify_trade(action: str, ticker: str, price: float, shares: int, reason: str = ""):
    """売買実行通知"""
    name = TICKER_NAMES.get(ticker, ticker)
    msg = f"\n[株AI 取引] {action}: {name}\n  価格: {price:,.0f}円 x {shares}株"
    if reason:
        msg += f"\n  理由: {reason}"
    send_line(msg)
    send_discord(msg)
