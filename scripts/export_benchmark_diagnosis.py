from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.legal_case_rag.app import benchmark_service as benchmark
from src.legal_case_rag.runtime.env import load_project_env


CSV_FIELDS = [
    "method",
    "query_id",
    "query_text",
    "difficulty",
    "trap",
    "main_leaf",
    "positive_count",
    "expected_ndcg@20",
    "recall@20",
    "recall@100",
    "mrr",
    "first_positive_rank",
    "top20_has_positive",
    "top100_has_positive",
    "weak_top20_count",
    "failure_type",
    "diagnosis_label",
    "diagnosis_bucket",
    "diagnosis_reason",
    "positive_doc_ids",
    "missed_positive_doc_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CaseLaw-Bench bad-case diagnosis tables.")
    parser.add_argument("--limit", type=int, default=58)
    parser.add_argument("--method", choices=["hybrid", "hybrid_rerank"], default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--candidate-size", type=int, default=300)
    parser.add_argument("--chunk-top-k", type=int, default=2)
    parser.add_argument("--rerank-top-n", type=int, default=100)
    parser.add_argument("--rerank-model-weight", type=float, default=0.25)
    parser.add_argument("--rerank-min-interval-ms", type=int, default=1200)
    parser.add_argument("--rerank-max-retries", type=int, default=3)
    parser.add_argument("--no-rank-safe", dest="rerank_rank_safe", action="store_false")
    parser.add_argument("--rerank-max-rank-promotion", type=int, default=20)
    parser.add_argument("--display-top-n", type=int, default=20)
    parser.add_argument("--include-details", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.set_defaults(rerank_rank_safe=True)
    return parser.parse_args()


def make_output_dir(path: Path | None) -> Path:
    if path is not None:
        output_dir = path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("benchmark_runs") / f"diagnosis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    load_project_env()
    args = parse_args()
    output_dir = make_output_dir(args.output_dir)

    queries = benchmark.load_benchmark_queries(limit=args.limit)
    qrels = benchmark.load_benchmark_qrels()
    payload = benchmark.run_benchmark_method(
        args.method,
        queries=queries,
        qrels=qrels,
        top_k=args.top_k,
        candidate_size=args.candidate_size,
        chunk_top_k=args.chunk_top_k,
        rerank_top_n=args.rerank_top_n,
        rerank_model_weight=args.rerank_model_weight,
        rerank_min_interval_ms=args.rerank_min_interval_ms,
        rerank_max_retries=args.rerank_max_retries,
        rerank_rank_safe=args.rerank_rank_safe,
        rerank_max_rank_promotion=args.rerank_max_rank_promotion,
        route_weight_overrides={},
        display_top_n=args.display_top_n,
        include_details=args.include_details,
    )

    rows = benchmark.build_query_diagnosis_export_rows(args.method, payload)
    csv_path = output_dir / "query_diagnosis.csv"
    summary_path = output_dir / "diagnosis_summary.json"
    metrics_path = output_dir / "metrics.json"

    write_csv(csv_path, rows)
    write_json(summary_path, payload.get("diagnosis_summary", {}))
    write_json(
        metrics_path,
        {
            "settings": payload.get("settings", {}),
            "metrics": payload.get("metrics", {}),
            "errors": payload.get("errors", []),
        },
    )

    print(f"Diagnosis CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Metrics JSON: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
