from __future__ import annotations

import json
import os
import time
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from flask import Flask, jsonify, render_template, request

from src.legal_case_rag.retrieval import search as retrieval
from src.legal_case_rag.app.search_args import (
    build_search_args as build_shared_search_args,
    normalize_sequence,
)


app = Flask(__name__)
BENCHMARK_DATA_DIR = Path("data") / "caselaw-benchmark release" / "data"
POSITIVE_GRADE = 2
BENCHMARK_METHODS = {
    "hybrid": {"label": "Hybrid 召回", "mode": "hybrid", "rerank": False},
    "hybrid_rerank": {"label": "Hybrid + Rerank", "mode": "hybrid", "rerank": True},
}
BENCHMARK_METRICS = [
    "ndcg@10",
    "hit@5",
    "hit@10",
    "recall@20",
    "recall@50",
    "recall@100",
    "mrr",
    "map",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_benchmark_queries(limit: int | None = None) -> list[dict[str, Any]]:
    queries = read_jsonl(BENCHMARK_DATA_DIR / "queries.jsonl")
    return queries[:limit] if limit else queries


def load_benchmark_qrels() -> dict[str, dict[str, dict[str, Any]]]:
    qrels: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(BENCHMARK_DATA_DIR / "qrels.jsonl"):
        qrels[row["query_id"]] = {
            item["doc_id"]: item
            for item in row.get("candidates", [])
        }
    return qrels


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def qrel_grade(item: dict[str, Any]) -> int:
    return int(item.get("grade_主档", 0) or 0)


def qrel_gain(item: dict[str, Any]) -> float:
    return float(item.get("grade_期望", item.get("grade_主档", 0)) or 0.0)


def dcg(gains: list[float]) -> float:
    return sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))


def average_precision(ranking: list[str], positives: set[str]) -> float | None:
    if not positives:
        return None
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranking, 1):
        if doc_id in positives:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(positives)


def reciprocal_rank(ranking: list[str], positives: set[str]) -> float | None:
    if not positives:
        return None
    for rank, doc_id in enumerate(ranking, 1):
        if doc_id in positives:
            return 1.0 / rank
    return 0.0


def first_positive_rank(ranking: list[str], positives: set[str]) -> int | None:
    if not positives:
        return None
    for rank, doc_id in enumerate(ranking, 1):
        if doc_id in positives:
            return rank
    return None


def evaluate_single_ranking(
    query: dict[str, Any],
    ranking: list[str],
    qrels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    seen: set[str] = set()
    filtered_ranking: list[str] = []
    for doc_id in ranking:
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        filtered_ranking.append(doc_id)

    positives = {
        doc_id
        for doc_id, item in qrels.items()
        if qrel_grade(item) >= POSITIVE_GRADE
    }
    gains = {
        doc_id: qrel_gain(item)
        for doc_id, item in qrels.items()
    }

    def recall_at(k: int) -> float | None:
        if not positives:
            return None
        return len(set(filtered_ranking[:k]) & positives) / len(positives)

    top10_gains = [gains.get(doc_id, 0.0) for doc_id in filtered_ranking[:10]]
    ideal10 = sorted(gains.values(), reverse=True)[:10]
    ideal_dcg = dcg(ideal10)
    first_rank = first_positive_rank(filtered_ranking, positives)
    return {
        "query_id": query.get("query_id"),
        "difficulty": query.get("难度"),
        "trap": bool(query.get("陷阱")),
        "main_leaf": query.get("主叶子"),
        "positive_count": len(positives),
        "first_positive_rank": first_rank,
        "top20_has_positive": bool(positives and (set(filtered_ranking[:20]) & positives)),
        "top100_has_positive": bool(positives and (set(filtered_ranking[:100]) & positives)),
        "hit@5": 1.0 if positives and (set(filtered_ranking[:5]) & positives) else (0.0 if positives else None),
        "hit@10": 1.0 if positives and (set(filtered_ranking[:10]) & positives) else (0.0 if positives else None),
        "recall@20": recall_at(20),
        "recall@50": recall_at(50),
        "recall@100": recall_at(100),
        "mrr": reciprocal_rank(filtered_ranking, positives),
        "map": average_precision(filtered_ranking, positives),
        "ndcg@10": dcg(top10_gains) / ideal_dcg if ideal_dcg > 0 else None,
        "returned_count": len(filtered_ranking),
    }


def mean_defined(values: list[float | None] | Any) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "queries": len(rows),
        "queries_with_positive": sum(1 for row in rows if row.get("positive_count", 0) > 0),
        "avg_positive_count": mean_defined([float(row.get("positive_count", 0)) for row in rows]),
    }
    metrics.update({name: mean_defined([row.get(name) for row in rows]) for name in BENCHMARK_METRICS})
    return metrics


def public_retrieval_result(
    result: dict[str, Any],
    rels: dict[str, dict[str, Any]],
    rank: int,
    anchor: str | None,
) -> dict[str, Any]:
    doc_id = result.get("doc_id") or ""
    rel = rels.get(doc_id, {})
    case_doc = result.get("case_doc") or {}
    grade = qrel_grade(rel)
    return {
        "rank": rank,
        "doc_id": doc_id,
        "case_name": case_doc.get("case_name") or result.get("case_name") or doc_id,
        "case_score": result.get("case_score"),
        "hybrid_case_score": result.get("hybrid_case_score"),
        "hybrid_rank": result.get("hybrid_rank"),
        "rerank_score": result.get("rerank_score"),
        "rerank_fused_score": result.get("rerank_fused_score"),
        "rerank_structure_adjustment": result.get("rerank_structure_adjustment"),
        "rerank_guardrail_adjustment": result.get("rerank_guardrail_adjustment"),
        "rerank_guardrail_penalty": result.get("rerank_guardrail_penalty"),
        "rerank_guardrail_bonus": result.get("rerank_guardrail_bonus"),
        "rerank_guardrail_missing": result.get("rerank_guardrail_missing"),
        "rerank_guardrail_conflicts": result.get("rerank_guardrail_conflicts"),
        "rank_safe_penalty": result.get("rank_safe_penalty"),
        "rank_safe_allowed_rank": result.get("rank_safe_allowed_rank"),
        "hit_count": result.get("hit_count"),
        "grade": grade,
        "grade_expected": rel.get("grade_期望", 0),
        "grade_std": rel.get("grade_std", 0),
        "is_anchor": bool(anchor and doc_id == anchor),
        "is_judged": bool(rel),
        "is_positive": grade >= POSITIVE_GRADE,
        "is_weak": grade == 1,
        "matched_chunks": [
            {
                "section_type": chunk.get("section_type"),
                "section_title": chunk.get("section_title"),
                "score": chunk.get("score"),
                "chunk_text": (chunk.get("chunk_text") or "")[:260],
            }
            for chunk in result.get("matched_chunks", [])[:2]
        ],
    }


def run_retrieval(args: SimpleNamespace) -> dict[str, Any]:
    try:
        return retrieval.run_search(args)
    except SystemExit as exc:
        message = str(exc.code) if exc.code else "检索配置错误"
        raise RuntimeError(message) from exc


def normalize_benchmark_methods(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_methods = [value]
    elif isinstance(value, list):
        raw_methods = [str(item) for item in value]
    else:
        raw_methods = ["hybrid", "hybrid_rerank"]

    methods: list[str] = []
    for method in raw_methods:
        if method in BENCHMARK_METHODS and method not in methods:
            methods.append(method)
    return methods or ["hybrid_rerank"]


def group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = str(row.get(key, "") or "未知")
        grouped.setdefault(group, []).append(row)
    return {
        group: aggregate_metrics(items)
        for group, items in sorted(grouped.items(), key=lambda item: item[0])
    }


def run_benchmark_method(
    method_name: str,
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, dict[str, Any]]],
    top_k: int,
    candidate_size: int,
    chunk_top_k: int,
    rerank_top_n: int,
    rerank_model_weight: float,
    rerank_min_interval_ms: int,
    rerank_max_retries: int,
    rerank_rank_safe: bool,
    rerank_max_rank_promotion: int,
    route_weight_overrides: dict[str, float] | None,
    display_top_n: int,
    include_details: bool,
) -> dict[str, Any]:
    method_config = BENCHMARK_METHODS[method_name]
    metric_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for query in queries:
        search_payload = {
            "query": query.get("query_text", ""),
            "mode": method_config["mode"],
            "rerank": method_config["rerank"],
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "candidate_size": candidate_size,
            "rerank_top_n": rerank_top_n,
            "rerank_model_weight": rerank_model_weight,
            "rerank_min_interval_ms": rerank_min_interval_ms,
            "rerank_max_retries": rerank_max_retries,
            "rerank_rank_safe": rerank_rank_safe,
            "rerank_max_rank_promotion": rerank_max_rank_promotion,
            "show_context": False,
            "query_profile": True,
            "query_profile_boost": True,
            "route_weight_overrides": route_weight_overrides or {},
        }
        args = build_search_args(search_payload)
        try:
            result = run_retrieval(args)
        except Exception as exc:
            errors.append(
                {
                    "query_id": str(query.get("query_id", "")),
                    "query_text": str(query.get("query_text", "")),
                    "error": str(exc),
                }
            )
            continue

        results = result.get("results", [])
        ranking = [item.get("doc_id") for item in results if item.get("doc_id")]
        rels = qrels.get(query["query_id"], {})
        metric = evaluate_single_ranking(query, ranking, rels)
        metric_rows.append(metric)

        anchor = query.get("query_source_doc")
        positive_doc_ids = sorted(
            doc_id
            for doc_id, item in rels.items()
            if qrel_grade(item) >= POSITIVE_GRADE
        )
        missed_positive_doc_ids = [
            doc_id
            for doc_id in positive_doc_ids
            if doc_id not in set(ranking[:100])
        ]
        first_rank = metric.get("first_positive_rank")
        weak_top20_count = sum(
            1
            for doc_id in ranking[:20]
            if qrel_grade(rels.get(doc_id, {})) == 1
        )
        if metric.get("positive_count", 0) <= 0:
            failure_type = "no_positive"
        elif not metric.get("top100_has_positive"):
            failure_type = "recall_failure"
        elif not metric.get("top20_has_positive"):
            failure_type = "ranking_failure"
        else:
            failure_type = "hit_top20"
        top_results = [
            public_retrieval_result(item, rels, rank, anchor)
            for rank, item in enumerate(results[:display_top_n], 1)
        ]
        query_rows.append(
            {
                "query_id": query.get("query_id"),
                "query_text": query.get("query_text"),
                "difficulty": query.get("难度"),
                "trap": bool(query.get("陷阱")),
                "main_leaf": query.get("主叶子"),
                "query_source_doc": anchor,
                "positive_doc_ids": positive_doc_ids,
                "missed_positive_doc_ids": missed_positive_doc_ids,
                "first_positive_rank": first_rank,
                "top20_has_positive": metric.get("top20_has_positive"),
                "top100_has_positive": metric.get("top100_has_positive"),
                "weak_top20_count": weak_top20_count,
                "failure_type": failure_type,
                "metrics": metric,
                "top_results": top_results if include_details else [],
            }
        )

    return {
        "label": method_config["label"],
        "settings": {
            "method": method_name,
            "mode": method_config["mode"],
            "rerank": method_config["rerank"],
            "top_k": top_k,
            "candidate_size": candidate_size,
            "chunk_top_k": chunk_top_k,
            "route_weight_overrides": route_weight_overrides or {},
            "rerank_top_n": rerank_top_n if method_config["rerank"] else 0,
            "rerank_model_weight": rerank_model_weight if method_config["rerank"] else 0,
            "rerank_hybrid_weight": (1.0 - rerank_model_weight) if method_config["rerank"] else 0,
            "rerank_min_interval_ms": rerank_min_interval_ms if method_config["rerank"] else 0,
            "rerank_max_retries": rerank_max_retries if method_config["rerank"] else 0,
            "rerank_rank_safe": rerank_rank_safe if method_config["rerank"] else False,
            "rerank_max_rank_promotion": rerank_max_rank_promotion if method_config["rerank"] else 0,
            "display_top_n": display_top_n,
            "case_index": os.getenv("LEGAL_CASE_INDEX", retrieval.DEFAULT_CASE_INDEX),
            "chunk_index": os.getenv("LEGAL_CHUNK_INDEX", retrieval.DEFAULT_CHUNK_INDEX),
        },
        "metrics": {
            "overall": aggregate_metrics(metric_rows),
            "by_difficulty": group_metrics(metric_rows, "difficulty"),
            "by_trap": group_metrics(metric_rows, "trap"),
            "by_main_leaf": group_metrics(metric_rows, "main_leaf"),
        },
        "queries": query_rows,
        "errors": errors,
    }


def metric_delta(after: dict[str, Any], before: dict[str, Any], name: str) -> float | None:
    after_value = after.get(name)
    before_value = before.get(name)
    if after_value is None or before_value is None:
        return None
    return float(after_value) - float(before_value)


def build_method_comparison(method_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hybrid = method_results.get("hybrid")
    rerank = method_results.get("hybrid_rerank")
    if not hybrid or not rerank:
        return {}

    hybrid_by_id = {row["query_id"]: row for row in hybrid.get("queries", [])}
    rerank_by_id = {row["query_id"]: row for row in rerank.get("queries", [])}
    shared_ids = sorted(set(hybrid_by_id) & set(rerank_by_id))
    rows: list[dict[str, Any]] = []
    for query_id in shared_ids:
        base = hybrid_by_id[query_id]
        after = rerank_by_id[query_id]
        base_metrics = base.get("metrics", {})
        after_metrics = after.get("metrics", {})
        rows.append(
            {
                "query_id": query_id,
                "query_text": after.get("query_text") or base.get("query_text"),
                "main_leaf": after.get("main_leaf") or base.get("main_leaf"),
                "difficulty": after.get("difficulty") or base.get("difficulty"),
                "trap": after.get("trap"),
                "hybrid_first_positive_rank": base.get("first_positive_rank"),
                "rerank_first_positive_rank": after.get("first_positive_rank"),
                "hybrid_weak_top20_count": base.get("weak_top20_count"),
                "rerank_weak_top20_count": after.get("weak_top20_count"),
                "delta_recall@20": metric_delta(after_metrics, base_metrics, "recall@20"),
                "delta_ndcg@10": metric_delta(after_metrics, base_metrics, "ndcg@10"),
                "delta_mrr": metric_delta(after_metrics, base_metrics, "mrr"),
                "delta_map": metric_delta(after_metrics, base_metrics, "map"),
            }
        )
    return {
        "shared_query_count": len(shared_ids),
        "queries": rows,
    }


def build_search_args(payload: dict[str, Any]) -> SimpleNamespace:
    return build_shared_search_args(
        payload,
        default_config=default_frontend_config(),
        retrieval_module=retrieval,
        verify_ssl_default=False,
    )


def public_case_payload(case_doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": case_doc.get("doc_id") or "",
        "case_name": case_doc.get("case_name") or "",
        "reason": case_doc.get("reason") or "",
        "trial_level": case_doc.get("trial_level") or "",
        "court_name": case_doc.get("court_name") or "",
        "judge_date": case_doc.get("judge_date") or "",
        "publish_date": case_doc.get("publish_date") or "",
        "litigants": normalize_sequence(case_doc.get("litigants")),
        "statutes": normalize_sequence(case_doc.get("statutes")),
        "full_text_hash": case_doc.get("full_text_hash") or "",
        "full_text": case_doc.get("full_text") or "",
    }


def default_frontend_config() -> dict[str, Any]:
    return {
        "mode": "hybrid",
        "rerank": True,
        "query_profile": True,
        "query_profile_boost": True,
        "top_k": 8,
        "chunk_top_k": 3,
        "candidate_size": 80,
        "show_context": True,
        "context_window": 180,
        "rerank_top_n": 30,
        "rerank_model_weight": retrieval.CASE_RERANK_MODEL_WEIGHT,
        "rerank_min_interval_ms": retrieval.DEFAULT_RERANK_MIN_INTERVAL_MS,
        "rerank_max_retries": retrieval.DEFAULT_RERANK_MAX_RETRIES,
        "rerank_rank_safe": retrieval.DEFAULT_RERANK_RANK_SAFE,
        "rerank_max_rank_promotion": retrieval.DEFAULT_RERANK_MAX_RANK_PROMOTION,
        "section_type": "",
        "reason": "",
        "trial_level": "",
        "court_name": "",
        "judge_date_from": "",
        "judge_date_to": "",
        "embedding_model": retrieval.DEFAULT_EMBEDDING_MODEL,
        "rerank_model": retrieval.DEFAULT_RERANK_MODEL,
        "embedding_url": retrieval.DEFAULT_EMBEDDING_URL,
        "rerank_url": retrieval.DEFAULT_RERANK_URL,
    }


@app.get("/")
def index() -> str:
    return render_template(
        "legal_rag_index.html",
        defaults=default_frontend_config(),
    )


@app.get("/api/health")
def health() -> Any:
    return jsonify(
        {
            "ok": True,
            "opensearch_url": os.getenv("OPENSEARCH_URL", retrieval.DEFAULT_OPENSEARCH_URL),
            "chunk_index": os.getenv("LEGAL_CHUNK_INDEX", retrieval.DEFAULT_CHUNK_INDEX),
            "case_index": os.getenv("LEGAL_CASE_INDEX", retrieval.DEFAULT_CASE_INDEX),
            "has_opensearch_password": bool(
                os.getenv(retrieval.DEFAULT_OPENSEARCH_PASSWORD_ENV)
            ),
            "has_siliconflow_key": bool(os.getenv(retrieval.DEFAULT_EMBEDDING_KEY_ENV)),
            "defaults": default_frontend_config(),
        }
    )


@app.post("/api/search")
def api_search() -> Any:
    payload = request.get_json(silent=True) or {}
    args = build_search_args(payload)
    if not args.query:
        return jsonify({"ok": False, "error": "query 不能为空。"}), 400

    started = time.perf_counter()
    try:
        result = run_retrieval(args)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    duration_ms = int((time.perf_counter() - started) * 1000)
    for item in result.get("results", []):
        case_doc = item.get("case_doc", {})
        case_doc["litigants"] = normalize_sequence(case_doc.get("litigants"))
        case_doc["statutes"] = normalize_sequence(case_doc.get("statutes"))

    result["ok"] = True
    result["duration_ms"] = duration_ms
    result["result_count"] = len(result.get("results", []))
    return jsonify(result)


@app.post("/api/benchmark/evaluate")
def api_benchmark_evaluate() -> Any:
    payload = request.get_json(silent=True) or {}
    limit = clamp_int(payload.get("limit"), 58, 1, 58)
    top_k = clamp_int(payload.get("top_k"), 100, 1, 100)
    candidate_size = clamp_int(payload.get("candidate_size"), 300, top_k, 300)
    rerank_top_n = clamp_int(payload.get("rerank_top_n"), max(top_k, min(candidate_size, 100)), 1, 150)
    rerank_model_weight = clamp_float(
        payload.get("rerank_model_weight"),
        retrieval.CASE_RERANK_MODEL_WEIGHT,
        0.0,
        1.0,
    )
    rerank_min_interval_ms = clamp_int(
        payload.get("rerank_min_interval_ms"),
        retrieval.DEFAULT_RERANK_MIN_INTERVAL_MS,
        0,
        10000,
    )
    rerank_max_retries = clamp_int(
        payload.get("rerank_max_retries"),
        retrieval.DEFAULT_RERANK_MAX_RETRIES,
        0,
        8,
    )
    rerank_rank_safe = bool_value(
        payload.get("rerank_rank_safe"),
        retrieval.DEFAULT_RERANK_RANK_SAFE,
    )
    rerank_max_rank_promotion = clamp_int(
        payload.get("rerank_max_rank_promotion"),
        retrieval.DEFAULT_RERANK_MAX_RANK_PROMOTION,
        0,
        100,
    )
    display_top_n = clamp_int(payload.get("display_top_n"), min(top_k, 20), 1, min(top_k, 50))
    include_details = bool_value(payload.get("include_details"), True)
    methods = normalize_benchmark_methods(payload.get("methods"))

    started = time.perf_counter()
    try:
        queries = load_benchmark_queries(limit=limit)
        qrels = load_benchmark_qrels()
    except Exception as exc:
        return jsonify({"ok": False, "error": f"加载 benchmark 数据失败：{exc}"}), 500

    method_results = {
        method: run_benchmark_method(
            method,
            queries=queries,
            qrels=qrels,
            top_k=top_k,
            candidate_size=candidate_size,
            chunk_top_k=2,
            rerank_top_n=rerank_top_n,
            rerank_model_weight=rerank_model_weight,
            rerank_min_interval_ms=rerank_min_interval_ms,
            rerank_max_retries=rerank_max_retries,
            rerank_rank_safe=rerank_rank_safe,
            rerank_max_rank_promotion=rerank_max_rank_promotion,
            route_weight_overrides={},
            display_top_n=display_top_n,
            include_details=include_details,
        )
        for method in methods
    }
    comparison = build_method_comparison(method_results)
    all_errors = [
        {"method": method, **error}
        for method, method_payload in method_results.items()
        for error in method_payload["errors"]
    ]
    successful_methods = {
        method: method_payload
        for method, method_payload in method_results.items()
        if method_payload["queries"]
    }
    if not successful_methods and all_errors:
        first_error = all_errors[0]
        return jsonify(
            {
                "ok": False,
                "error": f"{first_error.get('method')} / {first_error.get('query_id')} 检索失败：{first_error.get('error')}",
                "errors": all_errors,
            }
        ), 500

    duration_ms = int((time.perf_counter() - started) * 1000)
    if "hybrid_rerank" in successful_methods:
        primary_method = "hybrid_rerank"
    elif successful_methods:
        primary_method = next(iter(successful_methods))
    else:
        primary_method = methods[0]
    return jsonify(
        {
            "ok": True,
            "duration_ms": duration_ms,
            "settings": {
                "limit": len(queries),
                "top_k": top_k,
                "candidate_size": candidate_size,
                "rerank_top_n": rerank_top_n,
                "rerank_model_weight": rerank_model_weight,
                "rerank_hybrid_weight": 1.0 - rerank_model_weight,
                "rerank_min_interval_ms": rerank_min_interval_ms,
                "rerank_max_retries": rerank_max_retries,
                "rerank_rank_safe": rerank_rank_safe,
                "rerank_max_rank_promotion": rerank_max_rank_promotion,
                "display_top_n": display_top_n,
                "methods": methods,
                "primary_method": primary_method,
                "case_index": os.getenv("LEGAL_CASE_INDEX", retrieval.DEFAULT_CASE_INDEX),
                "chunk_index": os.getenv("LEGAL_CHUNK_INDEX", retrieval.DEFAULT_CHUNK_INDEX),
            },
            "methods": method_results,
            "metrics": method_results[primary_method]["metrics"],
            "queries": method_results[primary_method]["queries"],
            "comparison": comparison,
            "errors": all_errors,
        }
    )


@app.get("/api/cases/<path:doc_id>")
def api_case(doc_id: str) -> Any:
    try:
        case_doc = retrieval.fetch_single_case(
            doc_id=doc_id,
            opensearch_url=os.getenv("OPENSEARCH_URL", retrieval.DEFAULT_OPENSEARCH_URL),
            opensearch_username=os.getenv(
                "OPENSEARCH_USERNAME",
                retrieval.DEFAULT_OPENSEARCH_USERNAME,
            ),
            opensearch_password=os.getenv(retrieval.DEFAULT_OPENSEARCH_PASSWORD_ENV),
            case_index=os.getenv("LEGAL_CASE_INDEX", retrieval.DEFAULT_CASE_INDEX),
            verify_ssl=False,
            timeout=30,
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    if not case_doc:
        return jsonify({"ok": False, "error": "未找到对应文书。"}), 404

    return jsonify({"ok": True, "case": public_case_payload(case_doc)})


if __name__ == "__main__":
    host = os.getenv("LEGAL_RAG_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("LEGAL_RAG_WEB_PORT", "7860"))
    app.run(host=host, port=port, debug=False)
