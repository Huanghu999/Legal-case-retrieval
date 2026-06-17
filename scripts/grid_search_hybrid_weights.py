from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import legal_rag_web as benchmark


REQUIRED_ENV = ["OPENSEARCH_PASSWORD", "SILICONFLOW_API_KEY"]

ROUTE_NAMES = [
    "bm25_raw",
    "vector_raw",
    "bm25_focus",
    "vector_focus",
    "bm25_fine_issue",
    "bm25_focus_section",
    "bm25_reasoning",
    "bm25_facts",
    "bm25_negative",
    "bm25_legal",
]

BASE_WEIGHTS = {
    "bm25_raw": 1.0,
    "vector_raw": 0.80,
    "bm25_focus": 0.95,
    "vector_focus": 1.20,
    "bm25_fine_issue": 1.20,
    "bm25_focus_section": 1.60,
    "bm25_reasoning": 1.10,
    "bm25_facts": 0.60,
    "bm25_negative": 1.20,
    "bm25_legal": 0.80,
}

CSV_FIELDS = [
    "run_id",
    "preset_name",
    *[f"{name}_weight" for name in ROUTE_NAMES],
    "queries",
    "queries_with_positive",
    "error_count",
    "recall@20",
    "recall@50",
    "recall@100",
    "ndcg@10",
    "mrr",
    "map",
    "hit@5",
    "hit@10",
    "weak_top20_avg",
    "first_positive_rank_avg",
    "optimization_score",
    "duration_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search Hybrid route weights on CaseLaw-Bench.")
    parser.add_argument("--limit", type=int, default=58)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--candidate-size", type=int, default=300)
    parser.add_argument("--chunk-top-k", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--score-ndcg-weight", type=float, default=0.40)
    parser.add_argument("--score-mrr-weight", type=float, default=0.35)
    parser.add_argument("--score-recall20-weight", type=float, default=0.25)
    parser.add_argument("--include-details", action="store_true")
    return parser.parse_args()


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def validate_environment() -> None:
    load_dotenv()
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if not missing:
        return
    names = ", ".join(missing)
    raise EnvironmentError(
        "Missing required environment variable(s): "
        f"{names}\n"
        "Hybrid grid search needs OpenSearch and embedding access.\n"
        "PowerShell example:\n"
        '$env:OPENSEARCH_PASSWORD="your OpenSearch password"\n'
        '$env:SILICONFLOW_API_KEY="your SiliconFlow API Key"\n'
        "Or create a .env file in the project root with those names."
    )


def make_output_dir(path: Path | None) -> Path:
    if path is not None:
        output_dir = path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("benchmark_runs") / f"hybrid_grid_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def merged_weights(name: str, updates: dict[str, float]) -> dict[str, Any]:
    weights = dict(BASE_WEIGHTS)
    weights.update(updates)
    return {"preset_name": name, "weights": weights}


def generate_weight_sets() -> list[dict[str, Any]]:
    presets = [
        merged_weights("baseline", {}),
        merged_weights("issue_strong", {"bm25_fine_issue": 1.60, "bm25_focus_section": 1.50, "bm25_facts": 0.70}),
        merged_weights("issue_extreme", {"bm25_fine_issue": 1.80, "bm25_focus_section": 1.70, "bm25_facts": 0.50}),
        merged_weights("facts_down", {"bm25_facts": 0.50, "bm25_reasoning": 1.20}),
        merged_weights("vector_up", {"vector_raw": 1.20, "vector_focus": 1.30}),
        merged_weights("bm25_up", {"bm25_raw": 1.20, "bm25_focus": 1.10, "bm25_fine_issue": 1.60}),
        merged_weights("negative_up", {"bm25_negative": 1.40, "bm25_facts": 0.70}),
        merged_weights("reasoning_up", {"bm25_reasoning": 1.30, "bm25_focus_section": 1.50}),
    ]

    fine_issue_values = [1.2, 1.4, 1.6, 1.8]
    focus_section_values = [1.2, 1.4, 1.6, 1.8]
    facts_values = [0.4, 0.6, 0.8, 1.0]
    vector_raw_values = [0.8, 1.0, 1.2]
    vector_focus_values = [0.9, 1.2]

    grid: list[dict[str, Any]] = []
    for fine_issue, focus_section, facts, vector_raw, vector_focus in product(
        fine_issue_values,
        focus_section_values,
        facts_values,
        vector_raw_values,
        vector_focus_values,
    ):
        grid.append(
            merged_weights(
                "grid",
                {
                    "bm25_fine_issue": fine_issue,
                    "bm25_focus_section": focus_section,
                    "bm25_facts": facts,
                    "vector_raw": vector_raw,
                    "vector_focus": vector_focus,
                    "bm25_reasoning": 1.10,
                    "bm25_negative": 1.20,
                    "bm25_legal": 0.80,
                },
            )
        )

    unique: dict[str, dict[str, Any]] = {}
    for item in presets + grid:
        key = json.dumps(item["weights"], sort_keys=True)
        unique.setdefault(key, item)
    return list(unique.values())


def value_or_zero(value: Any) -> float:
    return float(value or 0.0)


def average_defined(values: list[Any]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def optimization_score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    return (
        args.score_ndcg_weight * value_or_zero(metrics.get("ndcg@10"))
        + args.score_mrr_weight * value_or_zero(metrics.get("mrr"))
        + args.score_recall20_weight * value_or_zero(metrics.get("recall@20"))
    )


def run_one(
    run_id: int,
    preset_name: str,
    weights: dict[str, float],
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    method_payload = benchmark.run_benchmark_method(
        "hybrid",
        queries=queries,
        qrels=qrels,
        top_k=args.top_k,
        candidate_size=args.candidate_size,
        rerank_top_n=0,
        rerank_model_weight=0.0,
        rerank_min_interval_ms=0,
        rerank_max_retries=0,
        rerank_rank_safe=False,
        rerank_max_rank_promotion=0,
        route_weight_overrides=weights,
        display_top_n=20,
        include_details=args.include_details,
        chunk_top_k=args.chunk_top_k,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    metrics = method_payload["metrics"]["overall"]
    query_rows = method_payload.get("queries", [])
    weak_top20_avg = average_defined([row.get("weak_top20_count") for row in query_rows])
    first_positive_rank_avg = average_defined(
        [
            row.get("first_positive_rank")
            for row in query_rows
            if row.get("first_positive_rank") is not None
        ]
    )
    row: dict[str, Any] = {
        "run_id": run_id,
        "preset_name": preset_name,
        "queries": metrics.get("queries"),
        "queries_with_positive": metrics.get("queries_with_positive"),
        "error_count": len(method_payload.get("errors", [])),
        "weak_top20_avg": weak_top20_avg,
        "first_positive_rank_avg": first_positive_rank_avg,
        "optimization_score": optimization_score(metrics, args),
        "duration_ms": duration_ms,
    }
    for route_name in ROUTE_NAMES:
        row[f"{route_name}_weight"] = weights.get(route_name, BASE_WEIGHTS[route_name])
    for metric_name in [
        "recall@20",
        "recall@50",
        "recall@100",
        "ndcg@10",
        "mrr",
        "map",
        "hit@5",
        "hit@10",
    ]:
        row[metric_name] = metrics.get(metric_name)
    return row, method_payload.get("errors", [])


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    try:
        validate_environment()
    except EnvironmentError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_dir = make_output_dir(args.output_dir)
    results_path = output_dir / "results.csv"
    best_path = output_dir / "best.json"
    errors_path = output_dir / "errors.jsonl"

    queries = benchmark.load_benchmark_queries(limit=args.limit)
    qrels = benchmark.load_benchmark_qrels()
    weight_sets = generate_weight_sets()
    if args.max_runs is not None:
        weight_sets = weight_sets[: args.max_runs]

    rows: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, item in enumerate(weight_sets, start=1):
            row, errors = run_one(
                index,
                item["preset_name"],
                item["weights"],
                queries,
                qrels,
                args,
            )
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            for error in errors:
                all_errors.append({"run_id": index, "preset_name": item["preset_name"], **error})
            print(
                f"[{index}/{len(weight_sets)}] {item['preset_name']} "
                f"score={row['optimization_score']:.4f} "
                f"ndcg@10={value_or_zero(row.get('ndcg@10')):.4f} "
                f"mrr={value_or_zero(row.get('mrr')):.4f} "
                f"recall@20={value_or_zero(row.get('recall@20')):.4f}",
                flush=True,
            )

    valid_rows = [
        row
        for row in rows
        if int(row.get("queries_with_positive") or 0) >= 45 and int(row.get("error_count") or 0) == 0
    ]
    ranked_rows = sorted(
        valid_rows or rows,
        key=lambda row: value_or_zero(row.get("optimization_score")),
        reverse=True,
    )
    best = ranked_rows[0] if ranked_rows else {}
    write_json(
        best_path,
        {
            "best": best,
            "top10": ranked_rows[:10],
            "settings": {
                "limit": args.limit,
                "top_k": args.top_k,
                "candidate_size": args.candidate_size,
                "chunk_top_k": args.chunk_top_k,
                "score_weights": {
                    "ndcg@10": args.score_ndcg_weight,
                    "mrr": args.score_mrr_weight,
                    "recall@20": args.score_recall20_weight,
                },
                "runs": len(weight_sets),
            },
        },
    )
    with errors_path.open("w", encoding="utf-8") as file:
        for error in all_errors:
            file.write(json.dumps(error, ensure_ascii=False) + "\n")

    print(f"\nResults: {results_path}")
    print(f"Best: {best_path}")
    print(f"Errors: {errors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
