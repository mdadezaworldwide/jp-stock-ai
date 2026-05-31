"""保有ポートフォリオ売り判定事前計算 (GitHub Actions向け)"""

from pathlib import Path

import pandas as pd

from portfolio_tracker import check_sell_signals

DATA_DIR = Path(__file__).parent / "signal_data"
DATA_DIR.mkdir(exist_ok=True)


def update_portfolio():
    print("=== ポートフォリオ売り判定更新 ===")
    results = check_sell_signals()
    out = DATA_DIR / "portfolio_signals.csv"
    if not results:
        # 保有0件のときも空CSVを書き出して「分析済み」を示す
        pd.DataFrame(columns=[
            "銘柄", "ティッカー", "買値", "現在値", "損益", "保有日数",
            "RSI", "損切ライン", "利確ライン", "判定", "理由", "緊急度",
        ]).to_csv(out, index=False)
        print(f"保存: {out} (保有0件)")
        return
    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    print(f"保存: {out} ({len(df)}銘柄)")


if __name__ == "__main__":
    update_portfolio()
