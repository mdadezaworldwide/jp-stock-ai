"""セクター分析事前計算 (GitHub Actions向け)"""

from pathlib import Path

from sector_analysis import analyze_sector_rotation

DATA_DIR = Path(__file__).parent / "signal_data"
DATA_DIR.mkdir(exist_ok=True)


def update_sector():
    print("=== セクター分析更新 ===")
    df = analyze_sector_rotation()
    if df.empty:
        print("データなし、既存CSV維持")
        return
    out = DATA_DIR / "sector_rotation.csv"
    df.to_csv(out, index=False)
    print(f"保存: {out} ({len(df)}セクター)")


if __name__ == "__main__":
    update_sector()
