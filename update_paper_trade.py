"""ペーパートレード1ショット実行 (常駐ループの代わり、CI/GitHub Actions向け)"""

from realtime import RealtimeTrader


def update_paper_trade():
    trader = RealtimeTrader()
    if trader.model is None:
        print("モデル未訓練のためスキップ (models/ensemble.pkl が必要)")
        return
    trader.generate_signals()
    print("ペーパートレード更新完了")


if __name__ == "__main__":
    update_paper_trade()
