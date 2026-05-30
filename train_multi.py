"""保有期間別モデルを一括訓練"""

import config
from data_fetcher import fetch_all_data
from features import prepare_features, get_feature_columns
from ensemble import EnsembleModel
from pathlib import Path

# 保有期間 → 目標リターン
PERIODS = {
    1: 0.01,    # デイトレ: +1%
    5: 0.02,    # 1週間: +2%
    10: 0.03,   # 2週間: +3%
    20: 0.04,   # 1か月: +4%
    60: 0.06,   # 3か月: +6%
}
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def train_all():
    print("=" * 60)
    print("  保有期間別モデル一括訓練")
    print("=" * 60)

    print("\nデータ取得中...")
    raw_data = fetch_all_data()

    for days, target_ret in PERIODS.items():
        print(f"\n{'=' * 60}")
        print(f"  保有期間: {days}日 / 目標リターン: {target_ret:.0%} モデル訓練")
        print(f"{'=' * 60}")

        config.HOLD_DAYS = days
        config.TARGET_RETURN = target_ret

        print(f"  特徴量生成中 (HOLD_DAYS={days}, TARGET={target_ret:.0%})...")
        df = prepare_features(raw_data, include_fundamentals=False,
                              include_sentiment=False, include_market=False,
                              include_news=False, include_jquants=False)
        print(f"  特徴量数: {len(get_feature_columns(df))}, データ行数: {len(df)}")

        print(f"  アンサンブル訓練中...")
        model = EnsembleModel()
        model.train(df)

        save_path = MODEL_DIR / f"ensemble_{days}d.pkl"
        model.save(save_path)
        print(f"  保存: {save_path}")

    # デフォルトに戻す
    config.HOLD_DAYS = 5
    config.TARGET_RETURN = 0.02
    print(f"\n{'=' * 60}")
    print(f"  全{len(PERIODS)}モデルの訓練完了")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_all()
