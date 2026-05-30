"""
日本株AIトレーディングシステム（フル機能版）

使い方:
    python main.py train          - アンサンブル訓練 + バックテスト
    python main.py train-fast     - テクニカルのみ高速訓練
    python main.py optimize       - Optunaでパラメータ最適化 → 訓練
    python main.py signal         - 最新シグナル確認
    python main.py backtest       - バックテスト再実行
    python main.py realtime       - リアルタイム監視 + ペーパートレード
    python main.py sector         - セクターローテーション分析
    python main.py event          - イベントドリブン分析（決算パターン）
    python main.py paper          - ペーパートレード状況確認
    python main.py dashboard      - Webダッシュボード起動
"""

import sys

from data_fetcher import fetch_all_data
from features import prepare_features, get_feature_columns
from config import TICKERS, TICKER_NAMES, HOLD_DAYS, TARGET_RETURN


def cmd_train(fast: bool = False):
    """アンサンブル訓練"""
    from ensemble import EnsembleModel
    from backtest import run_backtest, print_backtest_report, plot_backtest

    mode = "高速（テクニカルのみ）" if fast else "フル（全データソース）"
    print("=" * 60)
    print(f"  日本株AI - アンサンブル訓練 [{mode}]")
    print(f"  保有日数: {HOLD_DAYS}日, 目標リターン: {TARGET_RETURN*100:.1f}%")
    print("=" * 60)

    print("\n[1/4] データ取得中...")
    raw_data = fetch_all_data()

    print("\n[2/4] 特徴量生成中...")
    df = prepare_features(
        raw_data,
        include_fundamentals=not fast,
        include_sentiment=not fast,
        include_market=not fast,
    )
    feature_cols = get_feature_columns(df)
    print(f"  特徴量数: {len(feature_cols)}, データ行数: {len(df)}")

    print("\n[3/4] アンサンブル訓練中...")
    model = EnsembleModel()
    model.train(df)
    model.feature_importance()
    model.save()

    print("\n[4/4] バックテスト実行中...")
    df_sig = model.predict_signals(df)
    result = run_backtest(df_sig, signal_threshold=0.5, use_risk_management=True)
    print_backtest_report(result)
    plot_backtest(result)


def cmd_optimize():
    """パラメータ最適化 → アンサンブル訓練"""
    from optimizer import optimize_lgbm, optimize_xgb, optimize_hold_days
    from ensemble import EnsembleModel
    from backtest import run_backtest, print_backtest_report, plot_backtest

    print("=" * 60)
    print("  日本株AI - パラメータ最適化")
    print("=" * 60)

    print("\n[1/5] データ取得中...")
    raw_data = fetch_all_data()

    print("\n[2/5] 最適保有日数の探索...")
    best_days = optimize_hold_days(raw_data)
    import config
    config.HOLD_DAYS = best_days

    print("\n[3/5] 特徴量生成中...")
    df = prepare_features(raw_data)
    print(f"  特徴量数: {len(get_feature_columns(df))}")

    print("\n[4/5] LightGBM 最適化...")
    lgbm_params = optimize_lgbm(df)

    print("\n[4/5] XGBoost 最適化...")
    xgb_params = optimize_xgb(df)

    print("\n[5/5] 最適パラメータでアンサンブル訓練...")
    model = EnsembleModel(lgbm_params=lgbm_params, xgb_params=xgb_params)
    model.train(df)
    model.feature_importance()
    model.save()

    print("\nバックテスト...")
    df_sig = model.predict_signals(df)
    result = run_backtest(df_sig, signal_threshold=0.5, use_risk_management=True)
    print_backtest_report(result)
    plot_backtest(result)


def cmd_signal():
    """最新シグナル"""
    from ensemble import EnsembleModel
    from risk_manager import RiskManager

    print("=" * 60)
    print("  日本株AI - シグナル生成")
    print("=" * 60)

    try:
        model = EnsembleModel.load()
    except FileNotFoundError:
        from model import load_model
        model = load_model()

    raw_data = fetch_all_data()
    df = prepare_features(raw_data)

    if hasattr(model, "predict_signals"):
        df_sig = model.predict_signals(df)
    else:
        from model import predict_signals
        df_sig = predict_signals(model, df)

    risk = RiskManager()

    print(f"\n{'銘柄':>10s} {'終値':>10s} {'確率':>8s} {'損切':>8s} {'利確':>8s} {'判定':>8s}")
    print("-" * 58)
    for ticker in TICKERS:
        td = df_sig[df_sig["Ticker"] == ticker]
        if td.empty:
            continue
        latest = td.iloc[-1]
        atr = latest.get("ATR_14", latest["Close"] * 0.02)
        sl = risk.calculate_stop_loss(latest["Close"], atr)
        tp = risk.calculate_take_profit(latest["Close"], atr)
        name = TICKER_NAMES.get(ticker, ticker)
        signal = ">>> BUY" if latest["Signal_prob"] >= 0.5 else "    ---"
        print(f"{name:>10s} {latest['Close']:>10,.0f} {latest['Signal_prob']:>8.3f} "
              f"{sl:>8,.0f} {tp:>8,.0f} {signal:>8s}")


def cmd_backtest():
    """バックテスト"""
    from ensemble import EnsembleModel
    from backtest import run_backtest, print_backtest_report, plot_backtest

    try:
        model = EnsembleModel.load()
    except FileNotFoundError:
        from model import load_model
        model = load_model()

    raw_data = fetch_all_data()
    df = prepare_features(raw_data)

    if hasattr(model, "predict_signals"):
        df_sig = model.predict_signals(df)
    else:
        from model import predict_signals
        df_sig = predict_signals(model, df)

    result = run_backtest(df_sig, signal_threshold=0.5, use_risk_management=True)
    print_backtest_report(result)
    plot_backtest(result)


def cmd_realtime():
    """リアルタイム監視"""
    from realtime import RealtimeTrader
    trader = RealtimeTrader()
    trader.run_scheduler()


def cmd_sector():
    """セクター分析"""
    from sector_analysis import print_sector_report
    print_sector_report()


def cmd_event():
    """イベント分析"""
    from event_driven import print_event_report
    print_event_report()


def cmd_paper():
    """ペーパートレード状況"""
    from paper_trade import PaperTrader
    paper = PaperTrader()
    paper.report()


def cmd_dashboard():
    """Streamlitダッシュボード起動"""
    import subprocess
    dashboard_path = str(__import__("pathlib").Path(__file__).parent / "dashboard.py")
    print("ダッシュボード起動中... ブラウザで http://localhost:8501 を開いてください")
    subprocess.run(["streamlit", "run", dashboard_path])


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    commands = {
        "train": lambda: cmd_train(fast=False),
        "train-fast": lambda: cmd_train(fast=True),
        "optimize": cmd_optimize,
        "signal": cmd_signal,
        "backtest": cmd_backtest,
        "realtime": cmd_realtime,
        "sector": cmd_sector,
        "event": cmd_event,
        "paper": cmd_paper,
        "dashboard": cmd_dashboard,
    }

    if cmd not in commands:
        print(f"不明なコマンド: {cmd}")
        print(f"使用可能: {', '.join(commands.keys())}")
        sys.exit(1)

    commands[cmd]()


if __name__ == "__main__":
    main()
