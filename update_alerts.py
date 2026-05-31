"""デイトレアラート事前計算 (GitHub Actions向け)"""

from pathlib import Path

import pandas as pd

from alert_monitor import scan_daily_signals

DATA_DIR = Path(__file__).parent / "signal_data"
DATA_DIR.mkdir(exist_ok=True)


def update_alerts(top_n: int = 20):
    print("=== デイトレアラート更新 ===")
    alerts = scan_daily_signals(top_n=top_n)
    if not alerts:
        print("アラート0件、既存CSV維持")
        return
    df = pd.DataFrame(alerts)
    out = DATA_DIR / "daily_alerts.csv"
    df.to_csv(out, index=False)
    print(f"保存: {out} ({len(df)}銘柄)")


if __name__ == "__main__":
    update_alerts()
