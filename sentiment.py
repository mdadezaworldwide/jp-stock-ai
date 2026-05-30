"""X (Twitter) センチメント分析"""

import json
import re
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

from config import X_BEARER_TOKEN, TICKER_NAMES, SENTIMENT_LOOKBACK_HOURS


# === 日本語センチメント辞書（株式向け） ===
POSITIVE_WORDS = {
    "上昇", "急騰", "高騰", "買い", "強気", "好決算", "増収", "増益",
    "最高益", "上方修正", "好調", "成長", "期待", "反発", "底打ち",
    "ブレイク", "出来高増", "ゴールデンクロス", "サプライズ", "好材料",
    "爆益", "テンバガー", "ストップ高", "年初来高値", "割安",
    "自社株買い", "増配", "復配", "黒字転換", "受注増",
}

NEGATIVE_WORDS = {
    "下落", "急落", "暴落", "売り", "弱気", "悪決算", "減収", "減益",
    "下方修正", "不調", "懸念", "リスク", "続落", "天井", "崩壊",
    "デッドクロス", "悪材料", "損切り", "含み損", "ストップ安",
    "年初来安値", "割高", "赤字", "債務超過", "減配", "無配",
    "リストラ", "不正", "訴訟", "行政処分",
}


def analyze_sentiment_local(text: str) -> float:
    """
    ローカル辞書ベースのセンチメント分析
    -1.0（ネガティブ）〜 +1.0（ポジティブ）
    """
    pos_count = sum(1 for w in POSITIVE_WORDS if w in text)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def fetch_x_posts(query: str, max_results: int = 50) -> list[dict]:
    """
    X API v2 で検索
    X_BEARER_TOKEN が未設定の場合は空リストを返す
    """
    if not X_BEARER_TOKEN:
        return []

    since = (datetime.now(timezone.utc) - timedelta(hours=SENTIMENT_LOOKBACK_HOURS))
    params = {
        "query": f"{query} lang:ja -is:retweet",
        "max_results": min(max_results, 100),
        "start_time": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tweet.fields": "created_at,public_metrics",
    }
    url = f"https://api.twitter.com/2/tweets/search/recent?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {X_BEARER_TOKEN}"})

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("data", [])
    except Exception as e:
        print(f"  [WARN] X API エラー ({query}): {e}")
        return []


def get_ticker_sentiment(ticker: str) -> dict:
    """1銘柄のセンチメント指標を算出"""
    name = TICKER_NAMES.get(ticker, "")
    if not name:
        return _empty_sentiment()

    # 銘柄名 + 「株」で検索
    posts = fetch_x_posts(f"{name} 株")

    if not posts:
        # APIなしの場合はダミー値
        return _empty_sentiment()

    sentiments = []
    engagement_scores = []

    for post in posts:
        text = post.get("text", "")
        score = analyze_sentiment_local(text)
        sentiments.append(score)

        metrics = post.get("public_metrics", {})
        engagement = (
            metrics.get("like_count", 0)
            + metrics.get("retweet_count", 0) * 2
            + metrics.get("reply_count", 0)
        )
        engagement_scores.append(engagement)

    sentiments = np.array(sentiments)
    engagements = np.array(engagement_scores)

    # エンゲージメント加重センチメント
    if engagements.sum() > 0:
        weighted_sentiment = np.average(sentiments, weights=engagements + 1)
    else:
        weighted_sentiment = sentiments.mean()

    return {
        "X_sentiment_mean": sentiments.mean(),
        "X_sentiment_std": sentiments.std(),
        "X_sentiment_weighted": weighted_sentiment,
        "X_sentiment_positive_ratio": (sentiments > 0).mean(),
        "X_sentiment_negative_ratio": (sentiments < 0).mean(),
        "X_post_count": len(posts),
        "X_total_engagement": engagements.sum(),
    }


def get_all_sentiments(tickers: list[str]) -> pd.DataFrame:
    """全銘柄のセンチメントを取得"""
    rows = []
    for ticker in tickers:
        name = TICKER_NAMES.get(ticker, ticker)
        print(f"  センチメント分析: {name}")
        data = get_ticker_sentiment(ticker)
        data["Ticker"] = ticker
        rows.append(data)
    return pd.DataFrame(rows)


def _empty_sentiment() -> dict:
    return {
        "X_sentiment_mean": 0.0,
        "X_sentiment_std": 0.0,
        "X_sentiment_weighted": 0.0,
        "X_sentiment_positive_ratio": 0.0,
        "X_sentiment_negative_ratio": 0.0,
        "X_post_count": 0,
        "X_total_engagement": 0,
    }


if __name__ == "__main__":
    from config import TICKERS
    df = get_all_sentiments(TICKERS)
    print(df.to_string())
