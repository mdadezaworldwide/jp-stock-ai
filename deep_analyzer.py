"""ディープファンダメンタルズ分析 — 決算短信・IR・適時開示・財務諸表を総合分析"""

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from safe_yf import download as _yf_download, get_info as _yf_info, get_ticker as _yf_ticker

ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY",
    "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"
)

CACHE_DIR = Path(__file__).parent / "analysis_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _claude_analyze(prompt: str, max_tokens: int = 2000) -> str:
    """Claude APIで分析"""
    if not ANTHROPIC_API_KEY:
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"[分析エラー: {e}]"


def fetch_ir_news(ticker: str, name: str) -> list[str]:
    """適時開示・IRニュースを取得"""
    code = ticker.replace(".T", "").replace("A", "")
    news = []

    # 1. 株探から最新ニュース
    try:
        url = f"https://kabutan.jp/stock/news?code={code}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            # ニュースタイトルを抽出
            for marker in ['class="news_firing"', 'class="news_firing "']:
                parts = html.split(marker)
                for part in parts[1:8]:
                    end = part.find("</a>")
                    if end > 0:
                        text = part[:end].split(">")[-1].strip()
                        if text and len(text) > 5:
                            news.append(text)
    except Exception:
        pass

    # 2. Google検索でIR・決算ニュース
    try:
        queries = [
            f"{name} 決算 2026",
            f"{name} IR 適時開示",
            f"{name} 業績 予想",
        ]
        for q in queries[:2]:
            encoded = urllib.parse.quote(q)
            url = f"https://www.google.com/search?q={encoded}&tbm=nws&num=5&hl=ja"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                for tag in ['<div class="BNeawe vvjwJb AP7Wnd">', '<h3']:
                    parts = html.split(tag)
                    for part in parts[1:4]:
                        end = part.find("<")
                        if end > 0:
                            title = part[:end].strip()
                            if len(title) > 10 and title not in news:
                                news.append(title)
    except Exception:
        pass

    return news[:10]


def get_detailed_financials(ticker: str) -> dict:
    """詳細な財務データを取得"""
    stock = _yf_ticker(ticker)
    if stock is None:
        return {"basic": {}}
    data = {}

    # 基本情報
    info = stock.info or {}
    data["basic"] = {
        "時価総額": info.get("marketCap"),
        "PER": info.get("trailingPE"),
        "予想PER": info.get("forwardPE"),
        "PBR": info.get("priceToBook"),
        "PSR": info.get("priceToSalesTrailing12Months"),
        "EV/EBITDA": info.get("enterpriseToEbitda"),
        "配当利回り": info.get("dividendYield"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "純利益率": info.get("profitMargins"),
        "営業利益率": info.get("operatingMargins"),
        "粗利率": info.get("grossMargins"),
        "売上成長率": info.get("revenueGrowth"),
        "利益成長率": info.get("earningsGrowth"),
        "負債/自己資本": info.get("debtToEquity"),
        "流動比率": info.get("currentRatio"),
        "速動比率": info.get("quickRatio"),
        "フリーCF": info.get("freeCashflow"),
        "営業CF": info.get("operatingCashflow"),
        "総収益": info.get("totalRevenue"),
        "EBITDA": info.get("ebitda"),
        "総負債": info.get("totalDebt"),
        "現金": info.get("totalCash"),
        "従業員数": info.get("fullTimeEmployees"),
        "アナリスト推奨": info.get("recommendationKey"),
        "目標株価_平均": info.get("targetMeanPrice"),
        "目標株価_高": info.get("targetHighPrice"),
        "目標株価_低": info.get("targetLowPrice"),
        "アナリスト数": info.get("numberOfAnalystOpinions"),
        "52週高値": info.get("fiftyTwoWeekHigh"),
        "52週安値": info.get("fiftyTwoWeekLow"),
        "業種": info.get("industry"),
        "セクター": info.get("sector"),
        "事業概要": info.get("longBusinessSummary", "")[:500],
    }

    # 損益計算書（四半期）
    try:
        income = stock.quarterly_income_stmt
        if income is not None and not income.empty:
            quarters = []
            for i in range(min(4, income.shape[1])):
                q = income.iloc[:, i]
                quarters.append({
                    "期": str(income.columns[i].date()) if hasattr(income.columns[i], 'date') else str(income.columns[i]),
                    "売上": q.get("Total Revenue"),
                    "営業利益": q.get("Operating Income"),
                    "純利益": q.get("Net Income"),
                    "EPS": q.get("Basic EPS"),
                })
            data["quarterly_income"] = quarters
    except Exception:
        pass

    # バランスシート
    try:
        bs = stock.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            latest = bs.iloc[:, 0]
            data["balance_sheet"] = {
                "総資産": latest.get("Total Assets"),
                "自己資本": latest.get("Stockholders Equity"),
                "流動資産": latest.get("Current Assets"),
                "流動負債": latest.get("Current Liabilities"),
                "長期負債": latest.get("Long Term Debt"),
                "棚卸資産": latest.get("Inventory"),
                "売掛金": latest.get("Accounts Receivable"),
            }
    except Exception:
        pass

    # キャッシュフロー
    try:
        cf = stock.quarterly_cashflow
        if cf is not None and not cf.empty:
            latest = cf.iloc[:, 0]
            data["cashflow"] = {
                "営業CF": latest.get("Operating Cash Flow"),
                "投資CF": latest.get("Investing Cash Flow"),
                "財務CF": latest.get("Financing Cash Flow"),
                "設備投資": latest.get("Capital Expenditure"),
                "FCF": (latest.get("Operating Cash Flow") or 0) + (latest.get("Capital Expenditure") or 0),
            }
    except Exception:
        pass

    return data


def deep_analyze_stock(ticker: str, name: str) -> dict:
    """1銘柄を徹底分析（Claude APIで総合評価）"""

    # キャッシュチェック（当日分）
    cache_file = CACHE_DIR / f"{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    print(f"  深層分析: {name} ({ticker})")

    # 1. 詳細財務データ取得
    financials = get_detailed_financials(ticker)

    # 2. IR・ニュース取得
    ir_news = fetch_ir_news(ticker, name)

    # 3. 株価データ
    try:
        hist = _yf_download(ticker, period="6mo", progress=False)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        price_info = {
            "現在値": float(hist["Close"].iloc[-1]) if not hist.empty else None,
            "6ヶ月リターン": float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) if len(hist) > 1 else None,
            "1ヶ月リターン": float(hist["Close"].iloc[-1] / hist["Close"].iloc[-21] - 1) if len(hist) >= 21 else None,
        }
    except Exception:
        price_info = {}

    # 4. Claude APIで総合分析
    analysis_prompt = f"""あなたは世界最高レベルの株式アナリストです。以下の{name}({ticker})の全データを分析し、投資判断を行ってください。

=== 財務指標 ===
{json.dumps(financials.get('basic', {}), ensure_ascii=False, indent=2, default=str)}

=== 四半期業績推移 ===
{json.dumps(financials.get('quarterly_income', []), ensure_ascii=False, indent=2, default=str)}

=== バランスシート ===
{json.dumps(financials.get('balance_sheet', {}), ensure_ascii=False, indent=2, default=str)}

=== キャッシュフロー ===
{json.dumps(financials.get('cashflow', {}), ensure_ascii=False, indent=2, default=str)}

=== 株価情報 ===
{json.dumps(price_info, ensure_ascii=False, default=str)}

=== 最新IR・ニュース ===
{chr(10).join(f'- {n}' for n in ir_news) if ir_news else '取得なし'}

以下のJSON形式で回答してください。他のテキストは不要です:
{{
    "total_score": -10から+10の整数（-10=強い売り, 0=中立, +10=強い買い）,
    "fundamental_score": -10から+10（ファンダメンタルズ評価）,
    "growth_score": -10から+10（成長性評価）,
    "value_score": -10から+10（割安度評価）,
    "financial_health_score": -10から+10（財務健全性）,
    "ir_sentiment_score": -10から+10（IR・ニュースのセンチメント）,
    "summary": "3文以内の総合評価コメント",
    "key_positives": ["ポジティブ要因を3つ以内"],
    "key_risks": ["リスク要因を3つ以内"],
    "fair_value_estimate": "適正株価の推定（円）またはnull",
    "recommendation": "強い買い/買い/中立/売り/強い売り"
}}"""

    ai_analysis = _claude_analyze(analysis_prompt)

    # JSONパース
    parsed = {}
    try:
        if "{" in ai_analysis:
            json_str = ai_analysis[ai_analysis.index("{"):ai_analysis.rindex("}") + 1]
            parsed = json.loads(json_str)
    except Exception:
        parsed = {"total_score": 0, "summary": ai_analysis[:200]}

    result = {
        "ticker": ticker,
        "name": name,
        "financials": financials,
        "ir_news": ir_news,
        "price_info": price_info,
        "ai_analysis": parsed,
        "analyzed_at": datetime.now().isoformat(),
    }

    # キャッシュ保存
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

    return result


def get_deep_features(ticker: str, name: str) -> dict:
    """深層分析の結果を特徴量として返す"""
    analysis = deep_analyze_stock(ticker, name)
    ai = analysis.get("ai_analysis", {})

    return {
        "Deep_total_score": ai.get("total_score", 0),
        "Deep_fundamental_score": ai.get("fundamental_score", 0),
        "Deep_growth_score": ai.get("growth_score", 0),
        "Deep_value_score": ai.get("value_score", 0),
        "Deep_financial_health": ai.get("financial_health_score", 0),
        "Deep_ir_sentiment": ai.get("ir_sentiment_score", 0),
    }


def batch_deep_analyze(tickers: list[str], names: dict[str, str]) -> pd.DataFrame:
    """複数銘柄を一括深層分析"""
    rows = []
    for ticker in tickers:
        name = names.get(ticker, ticker)
        try:
            features = get_deep_features(ticker, name)
            features["Ticker"] = ticker
            rows.append(features)
        except Exception as e:
            print(f"  [WARN] {name} 深層分析失敗: {e}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # テスト: トヨタを分析
    result = deep_analyze_stock("7203.T", "トヨタ")
    ai = result["ai_analysis"]
    print(f"\n=== トヨタ 深層分析結果 ===")
    print(f"総合スコア: {ai.get('total_score', 'N/A')}/10")
    print(f"推奨: {ai.get('recommendation', 'N/A')}")
    print(f"要約: {ai.get('summary', 'N/A')}")
    print(f"ポジティブ: {ai.get('key_positives', [])}")
    print(f"リスク: {ai.get('key_risks', [])}")
