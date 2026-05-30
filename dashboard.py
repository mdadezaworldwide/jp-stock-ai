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
    if key not in os.environ:
        try:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
        except Exception:
            pass

from config import TICKERS, TICKER_NAMES, TICKER_SECTORS, HOLD_DAYS, TARGET_RETURN, INITIAL_CAPITAL
from data_fetcher import fetch_stock_data
from features import prepare_features, get_feature_columns

# 全てのポップアップ・通知を非表示
st.markdown("""<style>
    [data-testid="stNotification"],
    [data-testid="stToast"],
    div[data-baseweb="toast"],
    div[data-baseweb="notification"],
    div[role="alert"],
    .stToast,
    .stNotification,
    div[class*="toast"],
    div[class*="Toast"],
    div[class*="notification"],
    div[class*="Notification"],
    div[class*="snackbar"],
    div[class*="Snackbar"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
</style>""", unsafe_allow_html=True)

st.set_page_config(page_title="JP Stock AI", layout="wide")

# --- サイドバー ---
st.sidebar.title("JP Stock AI")
page = st.sidebar.radio("ページ", [
    "シグナル",
    "マイポートフォリオ",
    "カスタム銘柄分析",
    "バックテスト結果",
    "セクター分析",
    "イベント分析",
    "ペーパートレード",
    "銘柄詳細",
])


@st.cache_resource(ttl=3600)
def load_model():
    from ensemble import EnsembleModel
    try:
        return EnsembleModel.load()
    except FileNotFoundError:
        # クラウド上ではモデルがないので自動訓練
        st.info("初回起動: モデルを訓練中です（数分かかります）...")
        from data_fetcher import fetch_all_data
        from features import prepare_features
        raw = fetch_all_data()
        df = prepare_features(raw, include_fundamentals=False,
                              include_sentiment=False, include_market=False,
                              include_news=False, include_jquants=False)
        model = EnsembleModel()
        model.train(df)
        model.save()
        return model


@st.cache_data(ttl=600)
def load_data():
    from data_fetcher import fetch_all_data
    return fetch_all_data()


@st.cache_data(ttl=600, show_spinner=False)
def compute_signals():
    """シグナル計算をキャッシュ（10分間保持）"""
    model = load_model()
    raw_data = load_data()
    df = prepare_features(raw_data, include_fundamentals=False,
                          include_sentiment=False, include_market=False,
                          include_news=False, include_jquants=False)
    if hasattr(model, "predict_signals"):
        return model.predict_signals(df)
    else:
        from model import predict_signals
        return predict_signals(model, df)


@st.cache_data(ttl=600, show_spinner=False)
def get_signal_table():
    """番号付きシグナルテーブルを生成（全ページから参照可能）"""
    df_sig = compute_signals()
    from hold_advisor import advise_hold_period
    from custom_stocks import load_custom_stocks
    from data_fetcher import fetch_stock_data
    from features import add_technical_features

    signals = []
    model = load_model()

    # プライム銘柄
    for ticker in TICKERS:
        td = df_sig[df_sig["Ticker"] == ticker]
        if td.empty:
            continue
        latest = td.iloc[-1]
        rsi_val = latest.get("RSI_14", np.nan)
        macd_val = latest.get("MACD_hist", np.nan)

        if pd.notna(rsi_val):
            if rsi_val >= 70: rsi_label = "買われすぎ"
            elif rsi_val <= 30: rsi_label = "売られすぎ"
            elif rsi_val <= 40: rsi_label = "やや売られすぎ"
            elif rsi_val >= 60: rsi_label = "やや買われすぎ"
            else: rsi_label = "普通"
        else:
            rsi_label = "-"

        if pd.notna(macd_val):
            if macd_val > 0:
                macd_label = "上昇の勢い加速" if macd_val > 0.5 else "上昇の勢い鈍化"
            else:
                macd_label = "下落の勢い加速" if macd_val < -0.5 else "下落止まりつつある"
        else:
            macd_label = "-"

        hold_advice = advise_hold_period(td)
        signals.append({
            "銘柄": TICKER_NAMES.get(ticker, ticker),
            "ティッカー": ticker,
            "セクター": TICKER_SECTORS.get(ticker, ""),
            "終値": latest["Close"],
            "シグナル確率": latest["Signal_prob"],
            "判定": "BUY" if latest["Signal"] == 1 else "-",
            "RSI": rsi_val, "RSI判定": rsi_label,
            "MACD": macd_val, "MACD判定": macd_label,
            "推奨保有": hold_advice["label"], "保有理由": hold_advice["reason"],
        })

    # カスタム銘柄
    for cs in load_custom_stocks():
        ct = cs["ticker"]
        if ct in TICKERS:
            continue
        try:
            cdf = fetch_stock_data(ct, years=1)
            if cdf.empty or len(cdf) < 30:
                continue
            cdf = add_technical_features(cdf)
            cdf = cdf.dropna()
            if cdf.empty:
                continue
            feat_cols = get_feature_columns(cdf)
            if hasattr(model, "_align_features"):
                X_custom = cdf[feat_cols].tail(1)
                X_custom = model._align_features(X_custom)
                prob = float(model.predict_proba(X_custom)[0])
            else:
                prob = np.nan
            latest = cdf.iloc[-1]
            rsi_val = latest.get("RSI_14", np.nan)
            macd_val = latest.get("MACD_hist", np.nan)
            if pd.notna(rsi_val):
                if rsi_val >= 70: rsi_label = "買われすぎ"
                elif rsi_val <= 30: rsi_label = "売られすぎ"
                elif rsi_val <= 40: rsi_label = "やや売られすぎ"
                elif rsi_val >= 60: rsi_label = "やや買われすぎ"
                else: rsi_label = "普通"
            else:
                rsi_label = "-"
            if pd.notna(macd_val):
                if macd_val > 0:
                    macd_label = "上昇の勢い加速" if macd_val > 0.5 else "上昇の勢い鈍化"
                else:
                    macd_label = "下落の勢い加速" if macd_val < -0.5 else "下落止まりつつある"
            else:
                macd_label = "-"
            hold_advice = advise_hold_period(cdf)
            signals.append({
                "銘柄": cs["name"], "ティッカー": ct, "セクター": "カスタム",
                "終値": latest["Close"], "シグナル確率": prob,
                "判定": "BUY" if pd.notna(prob) and prob >= 0.5 else "-",
                "RSI": rsi_val, "RSI判定": rsi_label,
                "MACD": macd_val, "MACD判定": macd_label,
                "推奨保有": hold_advice["label"], "保有理由": hold_advice["reason"],
            })
        except Exception:
            continue

    sig_df = pd.DataFrame(signals)
    sort_order = {"BUY": 0, "強い買い": 1, "買い": 2, "やや買い": 3}
    sig_df["_sort"] = sig_df["判定"].map(sort_order).fillna(9)
    sig_df = sig_df.sort_values(["_sort", "シグナル確率"], ascending=[True, False]).drop(columns=["_sort"])
    sig_df = sig_df.reset_index(drop=True)
    sig_df.insert(0, "No.", range(1, len(sig_df) + 1))
    return sig_df


# ========== シグナル ==========
if page == "シグナル":
    st.title("売買シグナル")
    st.caption(f"保有日数: {HOLD_DAYS}日 / 目標リターン: {TARGET_RETURN*100:.1f}%")

    # カスタム銘柄の追加UI
    with st.expander("銘柄を追加"):
        st.markdown("東証: 銘柄コード + `.T`（例: `3776.T`） / 米国株: そのまま（例: `AAPL`）")
        add_col1, add_col2, add_col3 = st.columns([2, 2, 1])
        with add_col1:
            add_ticker = st.text_input("ティッカー", placeholder="3776.T", key="sig_add_ticker")
        with add_col2:
            add_name = st.text_input("銘柄名（任意）", placeholder="ブロードバンドタワー", key="sig_add_name")
        with add_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("追加", type="primary", key="sig_add_btn"):
                if add_ticker:
                    from custom_stocks import add_custom_stock
                    ok, msg = add_custom_stock(add_ticker.strip(), add_name.strip())
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.warning(msg)

    try:
        sig_df = get_signal_table()

        # 買いシグナルをハイライト
        buy_count = sig_df["判定"].str.contains("買い|BUY", na=False).sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("買いシグナル", f"{buy_count}銘柄")
        col2.metric("分析銘柄数", f"{len(sig_df)}銘柄")
        col3.metric("閾値", "50%")

        def highlight_signal(row):
            v = str(row["判定"])
            if v == "BUY" or "強い買い" in v:
                return ["background-color: #d4edda"] * len(row)
            elif "買い" in v:
                return ["background-color: #e8f5e9"] * len(row)
            elif "売り" in v:
                return ["background-color: #f8d7da"] * len(row)
            return [""] * len(row)

        st.dataframe(
            sig_df.style.apply(highlight_signal, axis=1).format({
                "終値": "{:,.0f}",
                "シグナル確率": "{:.1%}",
                "RSI": "{:.1f}",
                "MACD": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # 指標の説明
        with st.expander("指標の見方（シグナル確率 / RSI / MACD）"):
            st.markdown("#### シグナル確率（AIの総合判断）")
            st.markdown("""
この銘柄が保有日数以内に目標リターン以上 上がる確率をAIが90個の特徴量から予測した数値です。

| シグナル確率 | 意味 |
|---|---|
| **70%以上** | 強い買いシグナル（自信あり） |
| **60〜70%** | 買いシグナル |
| **50〜60%** | 弱い買いシグナル（他の指標も確認） |
| **50%未満** | 見送り |
""")
            st.markdown("---")
            col_r, col_m = st.columns(2)
            with col_r:
                st.markdown("#### RSI（買われすぎ / 売られすぎ）")
                st.markdown("""
| RSI | 意味 |
|---|---|
| **70以上** | 買われすぎ → そろそろ下がるかも |
| **30以下** | 売られすぎ → そろそろ上がるかも |
| 40〜60 | 普通 |
""")
            with col_m:
                st.markdown("#### MACD（トレンドの勢い）")
                st.markdown("""
| MACD | 意味 |
|---|---|
| **プラスで増加中** | 上昇の勢いが加速 |
| プラスで減少中 | 上昇の勢いが鈍化 |
| **マイナスで減少中** | 下落の勢いが加速 |
| マイナスで増加中 | 下落が止まりつつある |
""")
            st.markdown("---")
            st.markdown("#### 推奨保有期間")
            st.markdown("""
銘柄ごとのトレンド強度・ボラティリティ・移動平均の並びから、最適な保有期間をAIが判定します。

| 推奨 | 条件 |
|---|---|
| **短期（5〜7日）** | ボラティリティが高い / RSIが極端 / 短期リバウンド狙い |
| **中期（15〜20日）** | トレンド形成中 / 移動平均が上向き |
| **長期（30〜60日）** | パーフェクトオーダー成立 / 非常に強い上昇トレンド |
""")
            st.markdown("---")
            st.markdown("""
**シグナルとの組み合わせ:**
- BUY 70%以上 → そのまま買い
- BUY 50〜60% + RSI 30〜50 + MACD上昇中 → 有望（まだ安い＋勢いあり）
- BUY + RSI 70超え → 過熱気味、慎重に
- BUY + MACD大きくマイナス → まだ下落中、エントリーは待ちもあり
""")

        # === AIチャット（シグナルページ内） ===
        st.markdown("---")
        st.subheader("AIに質問する")
        st.caption("「2番はなんで買いシグナル？」のように番号で質問できます")

        import anthropic
        _DEFAULT_KEY = "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"
        ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", _DEFAULT_KEY)

        if ANTHROPIC_KEY:
            table_text = sig_df.to_string(index=False)
            chat_system = f"""あなたは日本株の専門アナリストAIです。
以下はAIシグナルシステムの現在の売買シグナル一覧です。ユーザーが「2番」「No.3」などと言ったら、この表のNo.列を参照してください。

--- 売買シグナル一覧 ---
{table_text}
--- ここまで ---

回答ルール:
- 上記テーブルのデータを使って具体的な数値で根拠を示す
- シグナル確率、RSI、MACD、RSI判定、MACD判定、推奨保有、保有理由の値を引用する
- 「買い」「売り」「保有」の判断を明確に述べる
- リスクも必ず言及する
- 日本語で簡潔に回答する"""

            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("例: 2番はなんで買いシグナル？ / トヨタは今買い時？"):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("分析中..."):
                        try:
                            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
                            messages = [{"role": m["role"], "content": m["content"]}
                                        for m in st.session_state.chat_messages[-8:]]
                            response = client.messages.create(
                                model="claude-haiku-4-5-20251001",
                                max_tokens=1500,
                                system=chat_system,
                                messages=messages,
                            )
                            reply = response.content[0].text
                        except Exception as e:
                            reply = f"エラー: {e}"

                        st.markdown(reply)
                        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

            if st.button("チャット履歴をクリア"):
                st.session_state.chat_messages = []
                st.rerun()

    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        st.info("先に `python main.py train` を実行してください")



# ========== マイポートフォリオ ==========
elif page == "マイポートフォリオ":
    st.title("マイポートフォリオ")
    st.caption("実際に買った銘柄を記録し、AIが売り時を判定します")

    from portfolio_tracker import (
        load_holdings, add_holding, remove_holding, check_sell_signals,
        TRADE_HISTORY_FILE,
    )

    # --- 銘柄追加フォーム ---
    with st.expander("買い記録を追加", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            buy_ticker = st.selectbox(
                "銘柄", TICKERS,
                format_func=lambda t: f"{TICKER_NAMES.get(t, t)} ({t})",
                key="buy_ticker",
            )
        with col2:
            buy_price = st.number_input("買値（円）", min_value=1.0, value=1000.0, step=1.0)
        with col3:
            buy_shares = st.number_input("株数", min_value=100, value=100, step=100)
        with col4:
            buy_date = st.date_input("購入日")

        if st.button("記録する", type="primary"):
            add_holding(buy_ticker, buy_price, buy_shares, buy_date.strftime("%Y-%m-%d"))
            st.success(f"{TICKER_NAMES.get(buy_ticker, buy_ticker)} を記録しました")
            st.rerun()

    # --- 売り判定 ---
    holdings = load_holdings()

    if holdings:
        st.subheader("保有銘柄 — AIの売り判定")

        with st.spinner("売り判定を分析中..."):
            sell_signals = check_sell_signals()

        if sell_signals:
            sell_df = pd.DataFrame(sell_signals)

            # 緊急度でソート
            urgency_order = {"高": 0, "中": 1, "低": 2}
            sell_df["_sort"] = sell_df["緊急度"].map(urgency_order)
            sell_df = sell_df.sort_values("_sort").drop(columns=["_sort"])

            def color_action(row):
                if "売り（" in row["判定"]:
                    return ["background-color: #f8d7da"] * len(row)
                elif "売り検討" in row["判定"]:
                    return ["background-color: #fff3cd"] * len(row)
                else:
                    return ["background-color: #d4edda"] * len(row)

            st.dataframe(
                sell_df.style.apply(color_action, axis=1).format({
                    "買値": "{:,.0f}",
                    "現在値": "{:,.0f}",
                    "損益": "{:+.1%}",
                    "RSI": "{:.1f}",
                    "損切ライン": "{:,.0f}",
                    "利確ライン": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # 判定の説明
            with st.expander("売り判定の見方"):
                st.markdown("""
| 判定 | 色 | 意味 |
|---|---|---|
| **売り（損切り/利確）** | 赤 | すぐに売却すべき（緊急度: 高） |
| **売り検討** | 黄 | 売却を検討すべき（緊急度: 中） |
| **保有継続** | 緑 | 今は売る必要なし |

**判定基準:**
- 損切りライン（買値 - ATR x 2）を下回った → 即売り
- 利確ライン（買値 + ATR x 3）に到達 → 即売り
- RSI 75以上 + 利益あり → 過熱、利確を検討
- 移動平均デッドクロス + 利益あり → トレンド転換、売り検討
- 含み損 -10%以上 → 損切り検討
- 90日以上保有で横ばい → 資金効率低下、乗り換え検討
""")

        # --- 売却フォーム ---
        with st.expander("売却を記録"):
            sell_tickers = [h["ticker"] for h in holdings]
            col1, col2 = st.columns(2)
            with col1:
                sell_ticker = st.selectbox(
                    "売却銘柄", sell_tickers,
                    format_func=lambda t: f"{TICKER_NAMES.get(t, t)} ({t})",
                    key="sell_ticker",
                )
            with col2:
                sell_price = st.number_input("売値（円）", min_value=1.0, value=1000.0, step=1.0, key="sell_price")

            if st.button("売却を記録"):
                remove_holding(sell_ticker, sell_price)
                st.success(f"{TICKER_NAMES.get(sell_ticker, sell_ticker)} を売却記録しました")
                st.rerun()

        # --- 取引履歴 ---
        if TRADE_HISTORY_FILE.exists():
            trades = pd.read_csv(TRADE_HISTORY_FILE)
            if not trades.empty:
                st.subheader("取引履歴")
                st.dataframe(trades, use_container_width=True, hide_index=True)

                total_pnl = trades["PnL"].sum()
                win_rate = (trades["PnL"] > 0).mean() * 100
                col1, col2, col3 = st.columns(3)
                col1.metric("累計損益", f"{total_pnl:+,.0f}円")
                col2.metric("勝率", f"{win_rate:.1f}%")
                col3.metric("取引数", f"{len(trades)}件")
    else:
        st.info("まだ銘柄が登録されていません。上の「買い記録を追加」から記録してください。")


# ========== カスタム銘柄分析 ==========
elif page == "カスタム銘柄分析":
    st.title("カスタム銘柄分析")
    st.caption("プライム株以外の銘柄も自由に追加して分析できます")

    from custom_stocks import load_custom_stocks, add_custom_stock, remove_custom_stock, analyze_custom_stock

    # --- 銘柄追加フォーム ---
    with st.expander("銘柄を追加", expanded=True):
        st.markdown("東証の銘柄コードに `.T` を付けて入力してください（例: `3776.T`）。米国株はそのまま（例: `AAPL`）。")
        col1, col2 = st.columns([2, 1])
        with col1:
            new_ticker = st.text_input("ティッカー", placeholder="例: 3776.T")
        with col2:
            new_name = st.text_input("銘柄名（任意）", placeholder="例: ブロードバンドタワー")

        if st.button("追加", type="primary"):
            if new_ticker:
                ok, msg = add_custom_stock(new_ticker.strip(), new_name.strip())
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.warning(msg)

    # --- 登録済み銘柄一覧 & 分析 ---
    custom_stocks = load_custom_stocks()

    if custom_stocks:
        st.subheader(f"登録銘柄 ({len(custom_stocks)}件)")

        # 削除ボタン
        cols = st.columns(min(len(custom_stocks), 6))
        for i, s in enumerate(custom_stocks):
            with cols[i % 6]:
                if st.button(f"x {s['name']}", key=f"del_{s['ticker']}"):
                    remove_custom_stock(s["ticker"])
                    st.rerun()

        st.markdown("---")

        # 全銘柄分析
        with st.spinner("分析中..."):
            results = []
            for s in custom_stocks:
                analysis = analyze_custom_stock(s["ticker"])
                if "error" not in analysis:
                    results.append({
                        "銘柄": s["name"],
                        "ティッカー": s["ticker"],
                        "現在値": analysis["current"],
                        "テクニカル判定": analysis["signal"],
                        "RSI": analysis["rsi"],
                        "RSI判定": analysis["rsi_label"],
                        "MACD": analysis["macd_hist"],
                        "MACD判定": analysis["macd_label"],
                        "トレンド": analysis["trend"],
                        "推奨保有": analysis["hold_label"],
                        "保有理由": analysis["hold_reason"],
                        "5日リターン": analysis["ret_5d"],
                        "20日リターン": analysis["ret_20d"],
                    })

            if results:
                res_df = pd.DataFrame(results)

                def color_signal(row):
                    s = row["テクニカル判定"]
                    if "強い買い" in s:
                        return ["background-color: #d4edda"] * len(row)
                    elif "買い" in s:
                        return ["background-color: #e8f5e9"] * len(row)
                    elif "売り" in s:
                        return ["background-color: #f8d7da"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    res_df.style.apply(color_signal, axis=1).format({
                        "現在値": "{:,.0f}",
                        "RSI": "{:.1f}",
                        "MACD": "{:.2f}",
                        "5日リターン": "{:+.1%}",
                        "20日リターン": "{:+.1%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

        # 個別チャート
        st.markdown("---")
        selected = st.selectbox(
            "詳細チャートを見る",
            [s["ticker"] for s in custom_stocks],
            format_func=lambda t: next((s["name"] for s in custom_stocks if s["ticker"] == t), t),
        )

        if selected:
            analysis = analyze_custom_stock(selected)
            if "error" not in analysis:
                name = next((s["name"] for s in custom_stocks if s["ticker"] == selected), selected)

                # チャート
                hist = analysis["hist"]
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(
                    x=hist.index, open=hist["Open"], high=hist["High"],
                    low=hist["Low"], close=hist["Close"], name="株価",
                ), row=1, col=1)

                for period, color in [(5, "orange"), (20, "blue"), (60, "red")]:
                    sma = hist["Close"].rolling(period).mean()
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=sma, name=f"SMA{period}",
                        line=dict(width=1, color=color),
                    ), row=1, col=1)

                fig.add_trace(go.Bar(
                    x=hist.index, y=hist["Volume"], name="出来高",
                    marker_color="lightblue",
                ), row=2, col=1)

                fig.update_layout(
                    title=f"{name} ({selected})",
                    xaxis_rangeslider_visible=False, height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

                # ファンダメンタルズ
                fund = analysis.get("fundamentals", {})
                if fund:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("PER", f"{fund.get('PER', 'N/A')}")
                    col2.metric("PBR", f"{fund.get('PBR', 'N/A')}")
                    col3.metric("ROE", f"{fund.get('ROE', 'N/A')}")
                    col4.metric("配当利回り", f"{fund.get('配当利回り', 'N/A')}")
    else:
        st.info("まだ銘柄が登録されていません。上のフォームからティッカーを入力して追加してください。")


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
