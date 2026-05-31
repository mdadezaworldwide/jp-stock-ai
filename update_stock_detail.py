"""銘柄詳細用OHLCV事前計算 (GitHub Actions向け)

TICKERS全銘柄の1年分OHLCV + 主要 info を保存。
銘柄詳細ページはこれを読むだけで表示できる。
"""

import json
from pathlib import Path

from safe_yf import get_ticker as _yf_ticker
from data_fetcher import fetch_stock_data

from config import TICKERS, TICKER_NAMES
from custom_stocks import load_custom_stocks

DATA_DIR = Path(__file__).parent / "signal_data"
DETAIL_DIR = DATA_DIR / "stock_detail"
DETAIL_DIR.mkdir(parents=True, exist_ok=True)

INFO_KEYS = [
    "trailingPE", "priceToBook", "returnOnEquity", "dividendYield",
    "marketCap", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "sector", "industry",
    "shortName", "longName", "averageVolume",
]


def _save_one(ticker: str):
    try:
        df = fetch_stock_data(ticker, years=1)
    except Exception as e:
        print(f"  価格取得失敗 {ticker}: {e}")
        return

    if df is None or df.empty:
        return

    df.to_csv(DETAIL_DIR / f"{ticker}.csv")

    info_path = DETAIL_DIR / f"{ticker}_info.json"
    try:
        stock = _yf_ticker(ticker)
        raw_info = stock.info or {}
        info = {k: raw_info.get(k) for k in INFO_KEYS}
    except Exception:
        info = {}

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, default=str)


def update_stock_detail():
    print("=== 銘柄詳細データ更新 ===")
    tickers = list(TICKERS)
    for cs in load_custom_stocks():
        if cs["ticker"] not in tickers:
            tickers.append(cs["ticker"])

    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        name = TICKER_NAMES.get(ticker, ticker)
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] {name}")
        _save_one(ticker)

    print(f"完了: {DETAIL_DIR} に {total}銘柄分保存")


if __name__ == "__main__":
    update_stock_detail()
