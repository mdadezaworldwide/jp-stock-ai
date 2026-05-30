"""Claude APIによるニュース・決算短信分析"""

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

import pandas as pd
import anthropic

from config import TICKERS, TICKER_NAMES

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# キャッシュ（同日中は再分析しない）
_cache: dict[str, dict] = {}
_cache_date: str = ""


def analyze_news_with_claude(ticker: str, news_texts: list[str]) -> dict:
    """Claude Haikuでニュース・決算情報を分析"""
    if not ANTHROPIC_API_KEY:
        return _empty_result()

    name = TICKER_NAMES.get(ticker, ticker)

    if not news_texts:
        return _empty_result()

    # 最大5件に絞る
    texts = news_texts[:5]
    combined = "\n---\n".join(texts)

    prompt = f"""以下は{name}({ticker})に関する最新ニュース・適時開示情報です。
投資判断の観点から分析してください。

{combined}

以下のJSON形式で回答してください。他のテキストは不要です。
{{
  "sentiment_score": -1.0から1.0の数値（-1=非常にネガティブ, 0=中立, 1=非常にポジティブ）,
  "confidence": 0.0から1.0の数値（分析の確信度）,
  "key_factors": ["株価に影響する要因を3つ以内で"],
  "outlook": "short_positive" or "short_negative" or "neutral"（短期見通し）
}}"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # JSON部分を抽出
        if "{" in text:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            result = json.loads(json_str)
            return {
                "AI_sentiment": float(result.get("sentiment_score", 0)),
                "AI_confidence": float(result.get("confidence", 0)),
                "AI_outlook": 1.0 if result.get("outlook") == "short_positive"
                              else -1.0 if result.get("outlook") == "short_negative"
                              else 0.0,
                "AI_factors": result.get("key_factors", []),
            }
    except Exception as e:
        print(f"  [WARN] Claude API エラー ({name}): {e}")

    return _empty_result()


def fetch_news_from_google(query: str, num: int = 5) -> list[str]:
    """Google検索からニュースタイトルを取得（無料）"""
    try:
        encoded = urllib.parse.quote(f"{query} 株 ニュース site:nikkei.com OR site:reuters.com OR site:kabutan.jp")
        url = f"https://www.google.com/search?q={encoded}&tbm=nws&num={num}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            # 簡易的にタイトルを抽出
            titles = []
            for marker in ['<div class="BNeawe vvjwJb AP7Wnd">', '<div class="n0jPhd ynAwRc tNxQIb nDgy9d">', '<h3']:
                parts = html.split(marker)
                for part in parts[1:]:
                    end = part.find("<")
                    if end > 0:
                        title = part[:end].strip()
                        if len(title) > 10:
                            titles.append(title)
            return titles[:num]
    except Exception:
        return []


def fetch_tdnet_disclosures(ticker_code: str) -> list[str]:
    """TDnet（適時開示）から最新情報を取得"""
    # TDnetは直接APIがないため、株探から取得を試みる
    try:
        url = f"https://kabutan.jp/stock/news?code={ticker_code}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            titles = []
            parts = html.split('<td class="news_firing">')
            for part in parts[1:6]:
                end = part.find("</a>")
                if end > 0:
                    # タグを除去
                    text = part[:end]
                    text = text.split(">")[-1].strip()
                    if text:
                        titles.append(text)
            return titles
    except Exception:
        return []


def analyze_all_stocks() -> pd.DataFrame:
    """全銘柄のニュース分析"""
    global _cache, _cache_date
    today = datetime.now().strftime("%Y-%m-%d")

    # 同日中はキャッシュを使う
    if _cache_date == today and _cache:
        print("  ニュース分析: キャッシュ使用")
        return pd.DataFrame(_cache.values())

    if not ANTHROPIC_API_KEY:
        print("  [INFO] ANTHROPIC_API_KEY 未設定 — ニュース分析をスキップ")
        return pd.DataFrame()

    results = []
    _cache = {}

    for ticker in TICKERS:
        name = TICKER_NAMES.get(ticker, ticker)
        code = ticker.replace(".T", "")
        print(f"  ニュース分析: {name}")

        # ニュースを収集
        news = []
        disclosures = fetch_tdnet_disclosures(code)
        news.extend(disclosures)

        google_news = fetch_news_from_google(name)
        news.extend(google_news)

        if not news:
            result = _empty_result()
        else:
            result = analyze_news_with_claude(ticker, news)

        result["Ticker"] = ticker
        results.append(result)
        _cache[ticker] = result

    _cache_date = today
    return pd.DataFrame(results)


def get_news_features(tickers: list[str] = None) -> pd.DataFrame:
    """特徴量として使えるニュース分析結果を返す"""
    df = analyze_all_stocks()
    if df.empty:
        return df
    # AI_factorsは特徴量に使えないので除外
    feature_cols = ["Ticker", "AI_sentiment", "AI_confidence", "AI_outlook"]
    return df[[c for c in feature_cols if c in df.columns]]


def _empty_result() -> dict:
    return {
        "AI_sentiment": 0.0,
        "AI_confidence": 0.0,
        "AI_outlook": 0.0,
        "AI_factors": [],
    }


if __name__ == "__main__":
    df = analyze_all_stocks()
    if not df.empty:
        print(df[["Ticker", "AI_sentiment", "AI_confidence", "AI_outlook"]].to_string())
    else:
        print("ANTHROPIC_API_KEY を設定してください")
