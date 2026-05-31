"""Streamlit ダッシュボード"""

import os
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SIGNAL_DATA_DIR = Path(__file__).parent / "signal_data"
LAST_UPDATED_FILE = SIGNAL_DATA_DIR / "last_updated.json"


def _show_last_updated(job_key: str | None = None):
    """signal_data/last_updated.json から最終更新時刻を表示"""
    if not LAST_UPDATED_FILE.exists():
        st.caption("最終更新: 未実行")
        return
    try:
        with open(LAST_UPDATED_FILE, encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        st.caption("最終更新: 不明")
        return
    if job_key and job_key in summary.get("jobs", {}):
        j = summary["jobs"][job_key]
        ts = j.get("finished_at", "?")
        status = "成功" if j.get("status") == "ok" else f"失敗 ({j.get('error', '?')[:40]})"
        st.caption(f"最終更新: {ts} ({status})")
    else:
        st.caption(f"最終更新: {summary.get('run_at', '?')}")

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
    "デイトレアラート",
    "マイポートフォリオ",
    "カスタム銘柄分析",
    "バックテスト結果",
    "セクター分析",
    "イベント分析",
    "ペーパートレード",
    "銘柄詳細",
])


@st.cache_resource(ttl=86400)
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


@st.cache_data(ttl=86400)
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
            from features import get_feature_columns
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


def _show_signal_table(sig_df):
    """シグナルテーブルを表示"""
    def highlight_signal(row):
        v = str(row["判定"])
        if v == "BUY":
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    fmt = {
        "終値": "{:,.0f}",
        "シグナル確率": "{:.1%}",
        "RSI": "{:.1f}",
    }

    st.dataframe(
        sig_df.style.apply(highlight_signal, axis=1).format(fmt),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("指標の見方（シグナル確率 / RSI / MACD）"):
        st.markdown("""
#### シグナル確率（AIの総合判断）
この銘柄が保有日数以内に目標リターン以上上がる確率をAIが90個の特徴量から予測した数値です。

| シグナル確率 | 意味 |
|---|---|
| **70%以上** | 強い買いシグナル（自信あり） |
| **60〜70%** | 買いシグナル |
| **50〜60%** | 弱い買いシグナル（他の指標も確認） |
| **50%未満** | 見送り |

---
#### RSI / MACD
| RSI | 意味 |  | MACD | 意味 |
|---|---|---|---|---|
| 70以上 | 買われすぎ | | プラス増加 | 上昇加速 |
| 30以下 | 売られすぎ | | プラス減少 | 上昇鈍化 |
| 40〜60 | 普通 | | マイナス減少 | 下落加速 |
| | | | マイナス増加 | 下落止まる |

---
#### 推奨保有期間
| 推奨 | 条件 |
|---|---|
| **短期（5〜7日）** | ボラティリティ高い / RSI極端 |
| **中期（15〜20日）** | トレンド形成中 / 移動平均上向き |
| **長期（30〜60日）** | パーフェクトオーダー成立 |
""")


def _show_ai_chat(sig_df):
    """AIチャットを表示"""
    import anthropic
    _DEFAULT_KEY = "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"
    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", _DEFAULT_KEY)
    if not ANTHROPIC_KEY:
        return

    st.markdown("---")
    st.subheader("AIに質問する")
    st.caption("「2番はなんで買いシグナル？」のように番号で質問できます")

    table_text = sig_df.to_string(index=False)
    chat_system = f"""あなたは日本株の専門アナリストAIです。
以下はAIシグナルシステムの現在の売買シグナル一覧です。ユーザーが「2番」「No.3」などと言ったら、この表のNo.列を参照してください。

--- 売買シグナル一覧 ---
{table_text}
--- ここまで ---

=== システム設計ロジック詳細 ===

【モデル構成】
- LightGBM + XGBoost のアンサンブル（重み自動最適化）
- Optunaで1500回試行のハイパーパラメータ最適化済み
- 保有期間別に4つのモデル（5日/10日/20日/60日）を個別訓練

【シグナル確率の算出方法】
- 90個の特徴量を入力 → LightGBMとXGBoostが各々確率を出力 → 重み付き平均でアンサンブル確率を算出
- 50%以上でBUY判定

【90個の特徴量（入力データ）】
テクニカル指標（37個）:
- トレンド系: SMA(5/10/20/60), EMA(5/10/20/60), SMA乖離率(5/20/60), MACD, MACDシグナル, MACDヒストグラム, ADX
- モメンタム系: RSI(14/9), ストキャスティクス(%K/%D), ROC(1/3/5/10/20日)
- ボラティリティ系: ボリンジャーバンド(上限/下限/幅/位置), ATR(14)
- 出来高系: 出来高SMA20, 出来高比率, OBV
- ローソク足: 実体比率, 上ヒゲ, 下ヒゲ, 高安位置(10/20日)

ファンダメンタルズ（22個）:
- バリュエーション: PER, 予想PER, PBR, PSR, EV/EBITDA, 配当利回り, 目標株価乖離率, アナリスト推奨
- 収益性: ROE, ROA, 純利益率, 営業利益率, 粗利率
- 成長性: 売上成長率, 利益成長率, 四半期売上成長, 四半期利益成長
- 財務: 負債/自己資本, 流動比率, 自己資本比率, FCF
- 規模: 時価総額

Xセンチメント（7個）:
- 平均感情スコア, 加重スコア, ポジティブ/ネガティブ比率, 標準偏差, 投稿数, エンゲージメント

市場全体（21個）:
- 日経225: 1日/5日/20日リターン, ボラティリティ, SMA20乖離, RSI, トレンド方向
- TOPIX: 1日/5日リターン
- NT倍率, NT倍率変化
- ドル円: 1日/5日リターン, 水準
- VIX: 水準, 変化率, 高ボラフラグ

AIニュース分析（3個）:
- Claude Haikuによるニュース感情スコア, 確信度, 短期見通し

【推奨保有期間の判定ロジック】
- ボラティリティ高い + RSI極端 → 短期(5〜7日)
- ADX高い + 移動平均上向き → 中期(15〜20日)
- パーフェクトオーダー(SMA5>SMA20>SMA60) → 長期(30〜60日)

【リスク管理】
- ATR×2の損切りライン
- ATR×3の利確ライン
- ドローダウン10%で全ポジション強制決済
- 同一セクター50%以上の集中防止
- 相関0.8以上の銘柄の同時保有防止

【目標リターン（保有期間別）】
- 1週間(5日): +2%以上
- 2週間(10日): +3%以上
- 1か月(20日): +4%以上
- 3か月(60日): +6%以上

=== 回答ルール ===
- 上記の設計ロジックに基づいて、なぜその銘柄のシグナル確率が高い/低いかを具体的に説明する
- テーブルの数値（RSI, MACD, シグナル確率等）を必ず引用する
- 90個の特徴量のうち、どの要素が判定に影響しているかを分析する
- 「買い」「売り」「保有」の判断を明確に述べる
- リスクも必ず言及する
- 日本語で回答する"""

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
            with st.spinner("深層分析中（IR・決算・財務を読み込み中）..."):
                # 質問から銘柄を特定して深層分析を実行
                import re
                deep_context = ""
                all_names = {**TICKER_NAMES}
                from custom_stocks import load_custom_stocks
                for cs in load_custom_stocks():
                    all_names[cs["ticker"]] = cs["name"]

                # No.番号で特定
                found_tickers = []
                try:
                    number_matches = re.findall(r'(?:No\.?|番号|シグナル|#)\s*(\d+)|(\d+)\s*(?:番|号)', prompt)
                    for match in number_matches:
                        num = int(match[0] or match[1])
                        if sig_table is not None and 1 <= num <= len(sig_table):
                            row = sig_table[sig_table["No."] == num].iloc[0]
                            found_tickers.append(row["ティッカー"])
                except Exception:
                    pass

                # 銘柄名で特定
                for ticker, name in all_names.items():
                    if name in prompt or ticker in prompt or ticker.replace(".T", "") in prompt:
                        if ticker not in found_tickers:
                            found_tickers.append(ticker)

                # 見つかった銘柄を深層分析
                if found_tickers:
                    from deep_analyzer import deep_analyze_stock
                    for ticker in found_tickers[:2]:
                        name = all_names.get(ticker, ticker)
                        analysis = deep_analyze_stock(ticker, name)
                        ai = analysis.get("ai_analysis", {})
                        fins = analysis.get("financials", {}).get("basic", {})
                        qi = analysis.get("quarterly_income", [])
                        bs = analysis.get("financials", {}).get("balance_sheet", {})
                        cf = analysis.get("financials", {}).get("cashflow", {})
                        ir = analysis.get("ir_news", [])

                        deep_context += f"""
=== {name} ({ticker}) 深層分析結果 ===
【AI総合スコア】{ai.get('total_score', 'N/A')}/10
【ファンダメンタルズ】{ai.get('fundamental_score', 'N/A')}/10
【成長性】{ai.get('growth_score', 'N/A')}/10
【割安度】{ai.get('value_score', 'N/A')}/10
【財務健全性】{ai.get('financial_health_score', 'N/A')}/10
【IR/ニュースセンチメント】{ai.get('ir_sentiment_score', 'N/A')}/10
【推奨】{ai.get('recommendation', 'N/A')}
【要約】{ai.get('summary', 'N/A')}
【ポジティブ要因】{ai.get('key_positives', [])}
【リスク要因】{ai.get('key_risks', [])}
【適正株価推定】{ai.get('fair_value_estimate', 'N/A')}

【主要財務指標】
PER: {fins.get('PER')} / PBR: {fins.get('PBR')} / ROE: {fins.get('ROE')} / ROA: {fins.get('ROA')}
営業利益率: {fins.get('営業利益率')} / 純利益率: {fins.get('純利益率')}
売上成長率: {fins.get('売上成長率')} / 利益成長率: {fins.get('利益成長率')}
配当利回り: {fins.get('配当利回り')} / 負債/自己資本: {fins.get('負債/自己資本')}
FCF: {fins.get('フリーCF')} / 営業CF: {fins.get('営業CF')}
アナリスト推奨: {fins.get('アナリスト推奨')} / 目標株価: {fins.get('目標株価_平均')}
52週高値: {fins.get('52週高値')} / 52週安値: {fins.get('52週安値')}

【バランスシート】{json.dumps(bs, ensure_ascii=False, default=str) if bs else 'N/A'}
【キャッシュフロー】{json.dumps(cf, ensure_ascii=False, default=str) if cf else 'N/A'}

【最新IR・ニュース】
{chr(10).join(f'- {n}' for n in ir[:5]) if ir else '取得なし'}
"""

                try:
                    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
                    # 深層分析結果をユーザーメッセージに追加
                    user_content = prompt
                    if deep_context:
                        user_content = f"{prompt}\n\n--- 以下は自動取得した深層分析データです ---\n{deep_context}"

                    messages = [{"role": m["role"], "content": m["content"]}
                                for m in st.session_state.chat_messages[:-1][-6:]]
                    messages.append({"role": "user", "content": user_content})

                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=2000,
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


# ========== シグナル ==========
if page == "シグナル":
    st.title("売買シグナル")
    _show_last_updated("signals")

    # 保有期間の切り替え
    hold_period = st.radio(
        "保有期間",
        ["デイトレ(1日)", "1週間(5日)", "2週間(10日)", "1か月(20日)", "3か月(60日)"],
        horizontal=True,
        index=0,
    )
    hold_days_map = {"デイトレ(1日)": 1, "1週間(5日)": 5, "2週間(10日)": 10, "1か月(20日)": 20, "3か月(60日)": 60}
    selected_hold_days = hold_days_map[hold_period]
    target_map = {1: "1%", 5: "2%", 10: "3%", 20: "4%", 60: "6%"}
    target_pct = target_map[selected_hold_days]
    st.caption(f"「{hold_period}保有して{target_pct}以上上がる確率」を表示")

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
                        st.success(f"{msg}（分析結果は次回データ更新時に反映されます）")
                    else:
                        st.warning(msg)

    try:
        def get_signal_table_period(days):
            """事前計算済みCSVからシグナルテーブルを読み込む"""
            csv_path = Path(__file__).parent / "signal_data" / f"signals_{days}d.csv"
            if csv_path.exists():
                return pd.read_csv(csv_path)
            else:
                st.warning(f"{days}日のシグナルデータがまだ生成されていません。")
                return pd.DataFrame()

        sig_df = get_signal_table_period(selected_hold_days)

        if sig_df.empty:
            st.warning("データが取得できませんでした。「データを最新に更新」ボタンを押してください。")

        buy_count = sig_df["判定"].str.contains("買い|BUY", na=False).sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("買いシグナル", f"{buy_count}銘柄")
        col2.metric("分析銘柄数", f"{len(sig_df)}銘柄")
        col3.metric("保有期間", hold_period)

        _show_signal_table(sig_df)
        _show_ai_chat(sig_df)

    except Exception as e:
        import traceback
        st.error(f"エラー: {e}")
        st.code(traceback.format_exc())
        st.info("先に `python main.py train` を実行してください")



# ========== デイトレアラート ==========
elif page == "デイトレアラート":
    st.title("デイトレアラート")
    st.caption("翌日+1%以上が期待される銘柄をAIが検出")
    _show_last_updated("alerts")

    alerts_csv = SIGNAL_DATA_DIR / "daily_alerts.csv"
    if alerts_csv.exists():
        alert_df = pd.read_csv(alerts_csv)
        if not alert_df.empty:
            def color_urgency(row):
                u = row.get("緊急度", "")
                if u == "高":
                    return ["background-color: #d4edda"] * len(row)
                elif u == "中":
                    return ["background-color: #e8f5e9"] * len(row)
                return [""] * len(row)

            st.subheader(f"検出: {len(alert_df)}銘柄")
            st.dataframe(
                alert_df.style.apply(color_urgency, axis=1).format({
                    "終値": "{:,.0f}",
                    "シグナル確率": "{:.1%}",
                    "RSI": "{:.1f}",
                    "MACD": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("""
**緊急度の見方:**
- **高（確率60%+）**: 強い買いシグナル、翌日寄付で買い推奨
- **中（確率50%+）**: 買い候補、他の指標も確認
- **低（確率45%+）**: 参考程度、慎重に判断
""")
        else:
            st.info("現在、強い短期シグナルは検出されませんでした")
    else:
        st.warning("アラートデータがまだ生成されていません。次回のデータ更新までお待ちください。")


# ========== マイポートフォリオ ==========
elif page == "マイポートフォリオ":
    st.title("マイポートフォリオ")
    st.caption("実際に買った銘柄を記録し、AIが売り時を判定します")

    from portfolio_tracker import (
        load_holdings, add_holding, remove_holding,
        TRADE_HISTORY_FILE,
    )
    _show_last_updated("portfolio")

    # --- 銘柄追加フォーム ---
    with st.expander("買い記録を追加", expanded=False):
        # プライム銘柄 + カスタム銘柄を統合
        from custom_stocks import load_custom_stocks as _load_cs
        _all_portfolio_tickers = list(TICKERS)
        _all_portfolio_names = dict(TICKER_NAMES)
        for _cs in _load_cs():
            if _cs["ticker"] not in _all_portfolio_tickers:
                _all_portfolio_tickers.append(_cs["ticker"])
                _all_portfolio_names[_cs["ticker"]] = _cs["name"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            buy_ticker = st.selectbox(
                "銘柄", _all_portfolio_tickers,
                format_func=lambda t: f"{_all_portfolio_names.get(t, t)} ({t})",
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
            st.success(f"{TICKER_NAMES.get(buy_ticker, buy_ticker)} を記録しました（売り判定は次回データ更新時に反映されます）")

    # --- 売り判定 ---
    holdings = load_holdings()

    if holdings:
        st.subheader("保有銘柄 — AIの売り判定")

        portfolio_csv = SIGNAL_DATA_DIR / "portfolio_signals.csv"
        if portfolio_csv.exists():
            sell_df = pd.read_csv(portfolio_csv)
            # holdings に無い銘柄は除外、holdings にあるが分析未済の銘柄は別途表示
            held_tickers = {h["ticker"] for h in holdings}
            if not sell_df.empty and "ティッカー" in sell_df.columns:
                sell_df = sell_df[sell_df["ティッカー"].isin(held_tickers)]
            unanalyzed = held_tickers - (set(sell_df["ティッカー"].tolist()) if not sell_df.empty else set())
            if unanalyzed:
                st.info(f"未分析の銘柄 ({len(unanalyzed)}件): 次回データ更新で反映されます — " + ", ".join(unanalyzed))
        else:
            sell_df = pd.DataFrame()
            st.warning("売り判定データがまだ生成されていません。")

        if not sell_df.empty:

            # 緊急度でソート
            urgency_order = {"高": 0, "中": 1, "低": 2}
            sell_df["_sort"] = sell_df["緊急度"].map(urgency_order)
            sell_df = sell_df.sort_values("_sort").drop(columns=["_sort"])

            def color_action(row):
                v = row["判定"]
                if "買い増しチャンス" in v:
                    return ["background-color: #cce5ff"] * len(row)
                elif "買い増し検討" in v:
                    return ["background-color: #d6eaf8"] * len(row)
                elif "売り（" in v:
                    return ["background-color: #f8d7da"] * len(row)
                elif "売り検討" in v:
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
            with st.expander("判定の見方"):
                st.markdown("""
| 判定 | 色 | 意味 |
|---|---|---|
| **買い増しチャンス** | 青 | ナンピン買い推奨（反発期待） |
| **買い増し検討** | 薄青 | 買い増しを検討してよい |
| **保有継続** | 緑 | 今は売る必要なし |
| **売り検討** | 黄 | 売却を検討すべき |
| **売り（損切り/利確）** | 赤 | すぐに売却すべき |

**買い増し条件:**
- RSI 30以下（売られすぎ）+ 業績健全（ROE/利益成長OK）→ 反発期待
- SMA20から8%以上乖離 + 出来高減少（売り枯れ）→ 平均回帰期待
- 含み損あり + RSI低め + トレンド弱い + ファンダ健全 → 一時的な下げ

**売り条件:**
- 損切りライン（買値 - ATR x 2）を下回った → 即売り
- 利確ライン（買値 + ATR x 3）に到達 → 即売り
- RSI 75以上 + 利益あり → 過熱、利確を検討
- デッドクロス + 強い下降トレンド → 損切り検討
- 含み損 -10%以上 + 業績悪化 → 損切り検討

**注意:** 買い増しはファンダメンタルズが健全な場合のみ推奨されます。業績悪化中の銘柄では表示されません。
""")

        # --- 売却 / 削除フォーム ---
        with st.expander("売却・削除"):
            sell_tickers = [h["ticker"] for h in holdings]
            col1, col2 = st.columns(2)
            with col1:
                sell_ticker = st.selectbox(
                    "銘柄を選択", sell_tickers,
                    format_func=lambda t: f"{_all_portfolio_names.get(t, t)} ({t})",
                    key="sell_ticker",
                )
            with col2:
                sell_price = st.number_input("売値（円）※売却時のみ", min_value=0.0, value=0.0, step=1.0, key="sell_price")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("売却を記録", type="primary"):
                    if sell_price > 0:
                        remove_holding(sell_ticker, sell_price)
                        st.success(f"{_all_portfolio_names.get(sell_ticker, sell_ticker)} を売却記録しました")
                        st.rerun()
                    else:
                        st.warning("売値を入力してください")
            with col_b:
                if st.button("記録を削除（売却せず取消）"):
                    remove_holding(sell_ticker)
                    st.success(f"{_all_portfolio_names.get(sell_ticker, sell_ticker)} を削除しました")
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
    _show_last_updated("custom")

    from custom_stocks import load_custom_stocks, add_custom_stock, remove_custom_stock

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
                    st.success(f"{msg}（分析結果は次回データ更新時に反映されます）")
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

        # 全銘柄分析 (事前計算CSV読み込み)
        custom_csv = SIGNAL_DATA_DIR / "custom_analysis.csv"
        if custom_csv.exists():
            res_df = pd.read_csv(custom_csv)
            if not res_df.empty:
                def color_signal(row):
                    s = str(row.get("判定", ""))
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
                        "5日リターン": "{:+.1%}",
                        "20日リターン": "{:+.1%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("カスタム銘柄分析データがまだ生成されていません。次回のデータ更新までお待ちください。")

        # 個別チャート
        st.markdown("---")
        selected = st.selectbox(
            "詳細チャートを見る",
            [s["ticker"] for s in custom_stocks],
            format_func=lambda t: next((s["name"] for s in custom_stocks if s["ticker"] == t), t),
        )

        if selected:
            name = next((s["name"] for s in custom_stocks if s["ticker"] == selected), selected)
            hist_csv = SIGNAL_DATA_DIR / "custom_hist" / f"{selected}.csv"
            if hist_csv.exists():
                hist = pd.read_csv(hist_csv, index_col=0, parse_dates=True)
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
            else:
                st.info(f"{name} のチャートデータは次回更新時に生成されます")

            # ファンダメンタルズ (事前計算JSON)
            info_json = SIGNAL_DATA_DIR / "custom_info" / f"{selected}.json"
            if info_json.exists():
                with open(info_json, encoding="utf-8") as f:
                    fund = json.load(f)
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
    _show_last_updated("sector")

    sector_csv = SIGNAL_DATA_DIR / "sector_rotation.csv"
    if sector_csv.exists():
        try:
            sector_df = pd.read_csv(sector_csv)

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
                st.warning("セクターデータが空です")
        except Exception as e:
            st.error(f"エラー: {e}")
    else:
        st.warning("セクターデータがまだ生成されていません。次回のデータ更新までお待ちください。")


# ========== イベント分析 ==========
elif page == "イベント分析":
    st.title("イベントドリブン分析（決算パターン）")
    _show_last_updated("events")

    events_csv = SIGNAL_DATA_DIR / "events.csv"
    if events_csv.exists():
        try:
            event_df = pd.read_csv(events_csv)

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
    else:
        st.warning("イベント分析データがまだ生成されていません。次回のデータ更新までお待ちください。")


# ========== ペーパートレード ==========
elif page == "ペーパートレード":
    st.title("ペーパートレード状況")
    _show_last_updated("paper_trade")

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
        st.info("ペーパートレードデータがまだ生成されていません。次回のデータ更新までお待ちください（GitHub Actions により毎営業日16:00 JSTに自動更新されます）。")


# ========== 銘柄詳細 ==========
elif page == "銘柄詳細":
    st.title("銘柄詳細分析")
    _show_last_updated("stock_detail")

    # カスタム銘柄も選択肢に含める
    from custom_stocks import load_custom_stocks as _load_cs_detail
    _detail_tickers = list(TICKERS)
    _detail_names = dict(TICKER_NAMES)
    for _cs in _load_cs_detail():
        if _cs["ticker"] not in _detail_tickers:
            _detail_tickers.append(_cs["ticker"])
            _detail_names[_cs["ticker"]] = _cs["name"]

    ticker = st.selectbox(
        "銘柄を選択",
        _detail_tickers,
        format_func=lambda t: f"{_detail_names.get(t, t)} ({t})",
    )

    if ticker:
        detail_csv = SIGNAL_DATA_DIR / "stock_detail" / f"{ticker}.csv"
        detail_info = SIGNAL_DATA_DIR / "stock_detail" / f"{ticker}_info.json"

        if detail_csv.exists():
            df = pd.read_csv(detail_csv, index_col=0, parse_dates=True)
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
                    title=f"{_detail_names.get(ticker, ticker)} ({ticker})",
                    xaxis_rangeslider_visible=False,
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)

                # 基本情報 (事前計算JSON)
                if detail_info.exists():
                    with open(detail_info, encoding="utf-8") as f:
                        info = json.load(f)
                else:
                    info = {}
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("PER", f"{info.get('trailingPE', 'N/A')}")
                col2.metric("PBR", f"{info.get('priceToBook', 'N/A')}")
                col3.metric("ROE", f"{info.get('returnOnEquity', 'N/A')}")
                col4.metric("配当利回り", f"{info.get('dividendYield', 'N/A')}")
        else:
            st.warning(f"{_detail_names.get(ticker, ticker)} の詳細データはまだ生成されていません。次回のデータ更新までお待ちください。")
