"""Streamlit ダッシュボード"""

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# Streamlit Secrets から環境変数にセット
for key in ["ANTHROPIC_API_KEY", "JQUANTS_API_KEY"]:
    if key not in os.environ and hasattr(st, "secrets") and key in st.secrets:
        os.environ[key] = st.secrets[key]

from config import TICKERS, TICKER_NAMES, TICKER_SECTORS, HOLD_DAYS, TARGET_RETURN, INITIAL_CAPITAL
from data_fetcher import fetch_stock_data
from features import prepare_features, get_feature_columns

st.set_page_config(page_title="JP Stock AI", layout="wide")

# --- サイドバー ---
st.sidebar.title("JP Stock AI")
page = st.sidebar.radio("ページ", [
    "シグナル",
    "バックテスト結果",
    "セクター分析",
    "イベント分析",
    "ペーパートレード",
    "銘柄詳細",
])


@st.cache_data(ttl=300)
def load_model():
    from ensemble import EnsembleModel
    try:
        return EnsembleModel.load()
    except FileNotFoundError:
        from model import load_model as load_lgbm
        return load_lgbm()


@st.cache_data(ttl=600)
def load_data():
    from data_fetcher import fetch_all_data
    return fetch_all_data()


# ========== シグナル ==========
if page == "シグナル":
    st.title("売買シグナル")
    st.caption(f"保有日数: {HOLD_DAYS}日 / 目標リターン: {TARGET_RETURN*100:.1f}%")

    with st.spinner("データ取得・分析中..."):
        try:
            model = load_model()
            raw_data = load_data()
            df = prepare_features(raw_data, include_fundamentals=False,
                                  include_sentiment=False, include_market=False)

            if hasattr(model, "predict_signals"):
                df_sig = model.predict_signals(df)
            else:
                from model import predict_signals
                df_sig = predict_signals(model, df)

            # 最新シグナル
            signals = []
            for ticker in TICKERS:
                td = df_sig[df_sig["Ticker"] == ticker]
                if td.empty:
                    continue
                latest = td.iloc[-1]
                signals.append({
                    "銘柄": TICKER_NAMES.get(ticker, ticker),
                    "ティッカー": ticker,
                    "セクター": TICKER_SECTORS.get(ticker, ""),
                    "終値": latest["Close"],
                    "シグナル確率": latest["Signal_prob"],
                    "判定": "BUY" if latest["Signal"] == 1 else "-",
                    "RSI": latest.get("RSI_14", np.nan),
                    "MACD": latest.get("MACD_hist", np.nan),
                })

            sig_df = pd.DataFrame(signals)

            # 買いシグナルをハイライト
            buy_count = (sig_df["判定"] == "BUY").sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("買いシグナル", f"{buy_count}銘柄")
            col2.metric("分析銘柄数", f"{len(sig_df)}銘柄")
            col3.metric("閾値", "50%")

            st.dataframe(
                sig_df.style.apply(
                    lambda row: ["background-color: #d4edda" if row["判定"] == "BUY" else "" for _ in row],
                    axis=1
                ).format({
                    "終値": "{:,.0f}",
                    "シグナル確率": "{:.1%}",
                    "RSI": "{:.1f}",
                    "MACD": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:
            st.error(f"モデル読み込みエラー: {e}")
            st.info("先に `python main.py train` を実行してください")


# ========== バックテスト結果 ==========
elif page == "バックテスト結果":
    st.title("バックテスト結果")

    # グラフ画像
    img_path = Path(__file__).parent / "backtest_result.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.warning("バックテスト未実行。`python main.py train` を実行してください")

    # 取引ログ
    trade_log = Path(__file__).parent / "paper_trades" / "trades.csv"
    if trade_log.exists():
        trades = pd.read_csv(trade_log)
        if not trades.empty:
            st.subheader("取引履歴")
            st.dataframe(trades, use_container_width=True)


# ========== セクター分析 ==========
elif page == "セクター分析":
    st.title("セクターローテーション分析")

    with st.spinner("セクターデータ取得中..."):
        try:
            from sector_analysis import analyze_sector_rotation
            sector_df = analyze_sector_rotation()

            if not sector_df.empty:
                # モメンタムチャート
                fig = go.Figure()
                colors = ["green" if m > 0 else "red" for m in sector_df["モメンタム"]]
                fig.add_trace(go.Bar(
                    x=sector_df["セクター"],
                    y=sector_df["モメンタム"],
                    marker_color=colors,
                ))
                fig.update_layout(title="セクター モメンタム", yaxis_title="モメンタム (短期-長期)")
                st.plotly_chart(fig, use_container_width=True)

                # テーブル
                st.dataframe(
                    sector_df.style.format({
                        "5日リターン": "{:+.1%}",
                        "20日リターン": "{:+.1%}",
                        "60日リターン": "{:+.1%}",
                        "モメンタム": "{:+.3f}",
                        "ボラティリティ": "{:.1%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("セクターデータを取得できませんでした")
        except Exception as e:
            st.error(f"エラー: {e}")


# ========== イベント分析 ==========
elif page == "イベント分析":
    st.title("イベントドリブン分析（決算パターン）")

    with st.spinner("決算データ分析中..."):
        try:
            from event_driven import analyze_all_events
            event_df = analyze_all_events()

            valid = event_df[event_df.get("has_data", False) == True]
            if not valid.empty:
                # 決算前パフォーマンス
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="決算前5日",
                    x=valid["name"],
                    y=valid["pre_mean"],
                    marker_color="steelblue",
                ))
                fig.add_trace(go.Bar(
                    name="決算後5日",
                    x=valid["name"],
                    y=valid["post_mean"],
                    marker_color="coral",
                ))
                fig.update_layout(title="決算前後の平均リターン", barmode="group", yaxis_tickformat=".1%")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    valid[["name", "pre_mean", "pre_win_rate", "post_mean",
                           "post_win_rate", "sample_count", "next_earnings"]].rename(columns={
                        "name": "銘柄", "pre_mean": "決算前リターン", "pre_win_rate": "決算前勝率",
                        "post_mean": "決算後リターン", "post_win_rate": "決算後勝率",
                        "sample_count": "サンプル数", "next_earnings": "次回決算",
                    }).style.format({
                        "決算前リターン": "{:+.2%}",
                        "決算前勝率": "{:.0%}",
                        "決算後リターン": "{:+.2%}",
                        "決算後勝率": "{:.0%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as e:
            st.error(f"エラー: {e}")


# ========== ペーパートレード ==========
elif page == "ペーパートレード":
    st.title("ペーパートレード状況")

    state_file = Path(__file__).parent / "paper_trades" / "state.json"
    trade_file = Path(__file__).parent / "paper_trades" / "trades.csv"

    if state_file.exists():
        import json
        with open(state_file) as f:
            state = json.load(f)

        col1, col2, col3 = st.columns(3)
        equity = state.get("capital", INITIAL_CAPITAL)
        col1.metric("現金", f"{equity:,.0f}円")
        col2.metric("ポジション数", f"{len(state.get('positions', []))}件")
        pnl = equity - INITIAL_CAPITAL
        col3.metric("損益", f"{pnl:+,.0f}円", delta=f"{pnl/INITIAL_CAPITAL:+.1%}")

        positions = state.get("positions", [])
        if positions:
            st.subheader("保有ポジション")
            pos_df = pd.DataFrame(positions)
            pos_df["銘柄名"] = pos_df["ticker"].map(TICKER_NAMES)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)

        if trade_file.exists():
            trades = pd.read_csv(trade_file)
            if not trades.empty:
                st.subheader("取引履歴")
                st.dataframe(trades, use_container_width=True, hide_index=True)

                # 累積損益グラフ
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=trades["PnL"].cumsum(),
                    mode="lines+markers",
                    name="累積損益",
                ))
                fig.update_layout(title="累積損益推移", yaxis_title="円")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ペーパートレードはまだ開始されていません。`python main.py realtime` で開始できます")


# ========== 銘柄詳細 ==========
elif page == "銘柄詳細":
    st.title("銘柄詳細分析")

    ticker = st.selectbox(
        "銘柄を選択",
        TICKERS,
        format_func=lambda t: f"{TICKER_NAMES.get(t, t)} ({t})",
    )

    if ticker:
        with st.spinner("データ取得中..."):
            df = fetch_stock_data(ticker, years=1)

            if not df.empty:
                # ローソク足チャート
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.7, 0.3],
                                    vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name="株価",
                ), row=1, col=1)

                # 移動平均
                for period, color in [(5, "orange"), (20, "blue"), (60, "red")]:
                    sma = df["Close"].rolling(period).mean()
                    fig.add_trace(go.Scatter(
                        x=df.index, y=sma, name=f"SMA{period}",
                        line=dict(width=1, color=color),
                    ), row=1, col=1)

                # 出来高
                fig.add_trace(go.Bar(
                    x=df.index, y=df["Volume"], name="出来高",
                    marker_color="lightblue",
                ), row=2, col=1)

                fig.update_layout(
                    title=f"{TICKER_NAMES.get(ticker, ticker)} ({ticker})",
                    xaxis_rangeslider_visible=False,
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)

                # 基本情報
                import yfinance as yf
                stock = yf.Ticker(ticker)
                info = stock.info or {}
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("PER", f"{info.get('trailingPE', 'N/A')}")
                col2.metric("PBR", f"{info.get('priceToBook', 'N/A')}")
                col3.metric("ROE", f"{info.get('returnOnEquity', 'N/A')}")
                col4.metric("配当利回り", f"{info.get('dividendYield', 'N/A')}")
