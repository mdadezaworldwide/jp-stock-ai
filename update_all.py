"""全データ事前計算オーケストレータ (GitHub Actions のエントリポイント)

各 update_*.py を順次実行し、個別失敗を許容して継続。
最後に signal_data/last_updated.json に各ジョブの結果を記録する。
"""

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "signal_data"
DATA_DIR.mkdir(exist_ok=True)
LAST_UPDATED_FILE = DATA_DIR / "last_updated.json"

JOBS = [
    ("signals", "update_signals", "update_all"),
    ("alerts", "update_alerts", "update_alerts"),
    ("sector", "update_sector", "update_sector"),
    ("events", "update_events", "update_events"),
    ("portfolio", "update_portfolio", "update_portfolio"),
    ("custom", "update_custom", "update_custom"),
    ("paper_trade", "update_paper_trade", "update_paper_trade"),
    ("stock_detail", "update_stock_detail", "update_stock_detail"),
]


def run_job(label: str, module_name: str, func_name: str) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    print(f"\n{'#' * 60}")
    print(f"# [{label}] 開始 {started}")
    print(f"{'#' * 60}")
    try:
        mod = __import__(module_name)
        func = getattr(mod, func_name)
        func()
        finished = datetime.now(timezone.utc).isoformat()
        print(f"# [{label}] 成功 {finished}")
        return {"status": "ok", "started_at": started, "finished_at": finished}
    except Exception as e:
        finished = datetime.now(timezone.utc).isoformat()
        tb = traceback.format_exc()
        print(f"# [{label}] 失敗 {finished}: {e}")
        print(tb)
        return {
            "status": "error",
            "started_at": started,
            "finished_at": finished,
            "error": str(e),
        }


def main():
    results = {}
    for label, module_name, func_name in JOBS:
        results[label] = run_job(label, module_name, func_name)

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "jobs": results,
    }
    with open(LAST_UPDATED_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("全ジョブ結果サマリ:")
    for label, r in results.items():
        mark = "OK " if r["status"] == "ok" else "FAIL"
        print(f"  [{mark}] {label}")
    print(f"記録: {LAST_UPDATED_FILE}")

    # 全失敗なら exit code 1 (CI で気付ける)
    if all(r["status"] == "error" for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
