"""イベント分析事前計算 (GitHub Actions向け)"""

from pathlib import Path

from event_driven import analyze_all_events

DATA_DIR = Path(__file__).parent / "signal_data"
DATA_DIR.mkdir(exist_ok=True)


def update_events():
    print("=== イベント分析更新 ===")
    df = analyze_all_events()
    if df.empty:
        print("データなし、既存CSV維持")
        return
    out = DATA_DIR / "events.csv"
    df.to_csv(out, index=False)
    print(f"保存: {out} ({len(df)}銘柄)")


if __name__ == "__main__":
    update_events()
