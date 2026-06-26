from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .constants import (
    CASE_RERANK_MODEL_WEIGHT,
    DEFAULT_CASE_INDEX,
    DEFAULT_CHUNK_INDEX,
    DEFAULT_EMBEDDING_KEY_ENV,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_URL,
    DEFAULT_OPENSEARCH_PASSWORD_ENV,
    DEFAULT_OPENSEARCH_URL,
    DEFAULT_OPENSEARCH_USERNAME,
    DEFAULT_RERANK_MAX_RANK_PROMOTION,
    DEFAULT_RERANK_MAX_RETRIES,
    DEFAULT_RERANK_MIN_INTERVAL_MS,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_RANK_SAFE,
    DEFAULT_RERANK_URL,
)
from .models import ChunkHit
from .llm_query_rewriter import rewrite_query_with_llm
from .opensearch_client import OpenSearchClient
from .query_profile import build_query_profile, build_query_routes, build_rerank_query
from .search_fusion import (
    apply_query_profile_bonus,
    case_level_reciprocal_rank_fusion,
    clone_hit_with_source,
)
from .search_queries import (
    build_filter_clauses,
    fetch_case_docs,
    route_filters,
    search_bm25,
    search_vector,
)
from .search_rerank import attach_case_key_chunks, rerank_case_hits
from .search_results import build_context, build_result_entry, compact_text
from .utils import safe_float


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)

    client = OpenSearchClient(
        base_url=args.opensearch_url,
        username=args.opensearch_username,
        password=args.opensearch_password,
        verify_ssl=args.verify_ssl,
        timeout=args.timeout,
    )
    filters = build_filter_clauses(args)
    query_profile_enabled = getattr(args, "query_profile", True)
    query_profile_boost = getattr(args, "query_profile_boost", True)
    route_weight_overrides = getattr(args, "route_weight_overrides", {}) or {}
    profile = build_query_profile(args.query)
    llm_query_rewrite_enabled = bool(query_profile_enabled and getattr(args, "llm_query_rewrite", False))
    rewrite = rewrite_query_with_llm(args.query, enabled=llm_query_rewrite_enabled)
    routes = build_query_routes(profile, rewrite if rewrite.used else None) if query_profile_enabled else []
    if not routes:
        routes = [
            route
            for route in build_query_routes(profile)
            if route.name in {"bm25_raw", "vector_raw"}
        ]

    ranked_lists: dict[str, list[ChunkHit]] = {}
    route_weights: dict[str, float] = {}
    route_payloads: list[dict[str, Any]] = []
    for route in routes:
        if route.route_type == "bm25" and args.mode not in {"bm25", "hybrid"}:
            continue
        if route.route_type == "vector" and args.mode not in {"vector", "hybrid"}:
            continue

        hits: list[ChunkHit]
        filters_for_route = route_filters(filters, getattr(route, "section_type", ""))
        if route.route_type == "bm25":
            hits = search_bm25(
                client=client,
                index_name=args.chunk_index,
                query=route.query,
                filters=filters_for_route,
                size=args.candidate_size,
            )
        else:
            hits = search_vector(
                client=client,
                index_name=args.chunk_index,
                query=route.query,
                filters=filters_for_route,
                size=args.candidate_size,
                api_key=args.embedding_api_key,
                model=args.embedding_model,
                endpoint=args.embedding_url,
            )

        source_name = route.name
        route_weight = safe_float(route_weight_overrides.get(source_name), route.weight)
        if route_weight is None:
            route_weight = route.weight
        ranked_lists[source_name] = [clone_hit_with_source(hit, source_name) for hit in hits]
        route_weights[source_name] = route_weight
        route_payloads.append(
            {
                "name": route.name,
                "type": route.route_type,
                "weight": route_weight,
                "default_weight": route.weight,
                "section_type": getattr(route, "section_type", ""),
                "query": route.query,
                "hit_count": len(hits),
            }
        )

    if query_profile_enabled and query_profile_boost:
        ranked_lists = {
            name: apply_query_profile_bonus(hits, profile)
            for name, hits in ranked_lists.items()
        }

    case_hits = case_level_reciprocal_rank_fusion(
        ranked_lists,
        weights=route_weights,
        top_chunks_per_case=args.chunk_top_k,
        k=60,
    )
    if args.rerank:
        rerank_query = build_rerank_query(profile, rewrite if rewrite.used else None) if query_profile_enabled else args.query
        guardrail_query = build_rerank_query(profile) if query_profile_enabled else args.query
        attach_case_key_chunks(
            client=client,
            index_name=args.chunk_index,
            case_hits=case_hits,
            top_n=max(args.top_k, args.rerank_top_n),
        )
        case_hits = rerank_case_hits(
            query=rerank_query,
            case_hits=case_hits,
            model_name=args.rerank_model,
            api_key=args.rerank_api_key,
            endpoint=args.rerank_url,
            top_n=max(args.top_k, args.rerank_top_n),
            timeout=args.rerank_timeout,
            max_chunks_per_doc=args.rerank_max_chunks_per_doc,
            overlap_tokens=args.rerank_overlap_tokens,
            model_weight=args.rerank_model_weight,
            min_interval_ms=args.rerank_min_interval_ms,
            max_retries=args.rerank_max_retries,
            rank_safe=args.rerank_rank_safe,
            max_rank_promotion=args.rerank_max_rank_promotion,
            guardrail_query=guardrail_query,
        )
    top_doc_ids = [item["doc_id"] for item in case_hits[: args.top_k]]
    case_docs = fetch_case_docs(client, args.case_index, top_doc_ids)

    payload = {
        "query": args.query,
        "mode": args.mode,
        "rerank": {
            "enabled": args.rerank,
            "model": args.rerank_model if args.rerank else "",
            "top_n": args.rerank_top_n if args.rerank else 0,
            "url": args.rerank_url if args.rerank else "",
            "timeout": args.rerank_timeout if args.rerank else 0,
            "max_chunks_per_doc": args.rerank_max_chunks_per_doc if args.rerank else 0,
            "overlap_tokens": args.rerank_overlap_tokens if args.rerank else 0,
            "model_weight": args.rerank_model_weight if args.rerank else 0,
            "hybrid_weight": (1.0 - args.rerank_model_weight) if args.rerank else 0,
            "min_interval_ms": args.rerank_min_interval_ms if args.rerank else 0,
            "max_retries": args.rerank_max_retries if args.rerank else 0,
            "rank_safe": args.rerank_rank_safe if args.rerank else False,
            "max_rank_promotion": args.rerank_max_rank_promotion if args.rerank else 0,
        },
        "filters": {
            "reason": args.reason,
            "trial_level": args.trial_level,
            "court_name": args.court_name,
            "section_type": args.section_type,
            "judge_date_from": args.judge_date_from,
            "judge_date_to": args.judge_date_to,
        },
        "query_profile": profile.to_dict() if query_profile_enabled else {},
        "query_profile_boost": bool(query_profile_enabled and query_profile_boost),
        "llm_query_rewrite": {
            "enabled": llm_query_rewrite_enabled,
            **rewrite.to_dict(),
        },
        "query_routes": route_payloads,
        "results": [
            build_result_entry(
                case_hit,
                case_docs.get(case_hit["doc_id"], {}),
                show_context=args.show_context,
                context_window=args.context_window,
                include_full_text=getattr(args, "include_full_text", False),
            )
            for case_hit in case_hits[: args.top_k]
        ],
    }
    return payload


def fetch_single_case(
    *,
    doc_id: str,
    opensearch_url: str = DEFAULT_OPENSEARCH_URL,
    opensearch_username: str = DEFAULT_OPENSEARCH_USERNAME,
    opensearch_password: str | None = None,
    case_index: str = DEFAULT_CASE_INDEX,
    verify_ssl: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    if not opensearch_password:
        raise RuntimeError("Missing OpenSearch password.")

    client = OpenSearchClient(
        base_url=opensearch_url,
        username=opensearch_username,
        password=opensearch_password,
        verify_ssl=verify_ssl,
        timeout=timeout,
    )
    docs = fetch_case_docs(client, case_index, [doc_id])
    return docs.get(doc_id, {})


def print_results(
    results: list[dict[str, Any]],
    case_docs: dict[str, dict[str, Any]],
    top_cases: int,
    top_chunks: int,
    show_context: bool,
    context_window: int,
) -> None:
    if not results:
        print("未召回到结果。可以尝试放宽过滤条件，或切换到 bm25/hybrid。")
        return

    for rank, case_hit in enumerate(results[:top_cases], start=1):
        doc_id = case_hit["doc_id"]
        case_doc = case_docs.get(doc_id, {})
        case_name = (
            case_doc.get("case_name")
            or case_hit.get("case_name")
            or case_doc.get("source_case_name")
            or ""
        )
        reason = case_doc.get("reason") or case_hit.get("reason") or ""
        trial_level = case_doc.get("trial_level") or case_hit.get("trial_level") or ""
        court_name = case_doc.get("court_name") or case_hit.get("court_name") or ""
        judge_date = case_doc.get("judge_date") or case_hit.get("judge_date") or ""
        print(f"[{rank}] {case_name}")
        print(
            f"    doc_id={doc_id} | score={case_hit['case_score']:.4f} | "
            f"案由={reason} | 审级={trial_level} | 法院={court_name} | 裁判日期={judge_date}"
        )
        print(f"    命中 chunk 数={case_hit['hit_count']}")

        full_text = case_doc.get("full_text", "")
        for chunk_rank, chunk in enumerate(case_hit["matched_chunks"][:top_chunks], start=1):
            source_flags = "+".join(chunk.match_sources) if chunk.match_sources else "unknown"
            title = chunk.section_title or chunk.section_type or "unknown"
            print(
                f"    ({chunk_rank}) {title} | chunk_score={chunk.score:.4f} | 来源={source_flags}"
            )
            print(f"        {compact_text(chunk.chunk_text, limit=220)}")
            if show_context:
                context = build_context(full_text, chunk.char_start, chunk.char_end, context_window)
                if context:
                    print(f"        上下文: {compact_text(context, limit=420)}")
        print()


def dump_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test legal case recall from OpenSearch.")
    parser.add_argument("--query", required=True, help="检索查询文本。")
    parser.add_argument(
        "--mode",
        choices=["bm25", "vector", "hybrid"],
        default="hybrid",
        help="召回模式。",
    )
    parser.add_argument("--reason", help="按案由过滤。")
    parser.add_argument("--trial-level", help="按审级过滤。")
    parser.add_argument("--court-name", help="按法院过滤。")
    parser.add_argument("--section-type", help="按 chunk 章节过滤。")
    parser.add_argument("--judge-date-from", help="裁判日期起始，格式 YYYY-MM-DD。")
    parser.add_argument("--judge-date-to", help="裁判日期结束，格式 YYYY-MM-DD。")
    parser.add_argument("--top-k", type=int, default=8, help="输出前多少个案件。")
    parser.add_argument("--chunk-top-k", type=int, default=3, help="每个案件展示多少个命中 chunk。")
    parser.add_argument("--candidate-size", type=int, default=80, help="每路召回拿多少个 chunk 候选。")
    parser.add_argument("--show-context", action="store_true", help="展示原文上下文片段。")
    parser.add_argument("--context-window", type=int, default=160, help="上下文窗口字符数。")
    parser.add_argument("--json-output", help="把结果写入 JSON 文件。")
    parser.add_argument(
        "--no-query-profile",
        dest="query_profile",
        action="store_false",
        help="关闭规则型 query profile、多路召回和否定事实加权，便于做对比实验。",
    )
    parser.add_argument(
        "--no-query-profile-boost",
        dest="query_profile_boost",
        action="store_false",
        help="保留多路 query，但关闭 query profile bonus 和否定事实加权。",
    )
    parser.add_argument(
        "--llm-query-rewrite",
        dest="llm_query_rewrite",
        action="store_true",
        help="启用 LLM query 重写/扩写，用字段对齐要素增强现有 routes。",
    )
    parser.add_argument(
        "--no-llm-query-rewrite",
        dest="llm_query_rewrite",
        action="store_false",
        help="关闭 LLM query 重写/扩写。",
    )
    parser.set_defaults(query_profile=True)
    parser.set_defaults(query_profile_boost=True)
    parser.set_defaults(llm_query_rewrite=False)
    parser.add_argument("--opensearch-url", default=os.getenv("OPENSEARCH_URL", DEFAULT_OPENSEARCH_URL))
    parser.add_argument(
        "--opensearch-username",
        default=os.getenv("OPENSEARCH_USERNAME", DEFAULT_OPENSEARCH_USERNAME),
    )
    parser.add_argument(
        "--opensearch-password",
        default=os.getenv(DEFAULT_OPENSEARCH_PASSWORD_ENV),
        help=f"OpenSearch 密码，默认读取环境变量 {DEFAULT_OPENSEARCH_PASSWORD_ENV}。",
    )
    parser.add_argument("--chunk-index", default=DEFAULT_CHUNK_INDEX)
    parser.add_argument("--case-index", default=DEFAULT_CASE_INDEX)
    parser.add_argument("--verify-ssl", action="store_true", help="校验证书。")
    parser.add_argument("--timeout", type=int, default=30, help="OpenSearch 请求超时秒数。")
    parser.add_argument(
        "--embedding-api-key",
        default=os.getenv(DEFAULT_EMBEDDING_KEY_ENV),
        help=f"Embedding API Key，默认读取环境变量 {DEFAULT_EMBEDDING_KEY_ENV}。",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-url", default=DEFAULT_EMBEDDING_URL)
    parser.add_argument("--embedding-timeout", type=int, default=60)
    parser.add_argument("--rerank", action="store_true", help="启用 SiliconFlow BGE reranker 精排。")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL, help="reranker 模型名。")
    parser.add_argument("--rerank-top-n", type=int, default=50, help="对前多少个案件做 rerank。")
    parser.add_argument("--rerank-model-weight", type=float, default=CASE_RERANK_MODEL_WEIGHT, help="案件级融合中 rerank 分数权重，0 到 1。")
    parser.add_argument(
        "--rerank-api-key",
        default=os.getenv(DEFAULT_EMBEDDING_KEY_ENV),
        help=f"Rerank API Key，默认读取环境变量 {DEFAULT_EMBEDDING_KEY_ENV}。",
    )
    parser.add_argument("--rerank-url", default=DEFAULT_RERANK_URL, help="rerank API 地址。")
    parser.add_argument("--rerank-timeout", type=int, default=120, help="rerank 请求超时秒数。")
    parser.add_argument("--rerank-min-interval-ms", type=int, default=DEFAULT_RERANK_MIN_INTERVAL_MS, help="rerank 请求之间的最小间隔毫秒数。")
    parser.add_argument("--rerank-max-retries", type=int, default=DEFAULT_RERANK_MAX_RETRIES, help="rerank 遇到限流或临时错误时的最大重试次数。")
    parser.add_argument("--no-rerank-rank-safe", dest="rerank_rank_safe", action="store_false", help="关闭 rank-safe rerank 名次上升限制。")
    parser.add_argument("--rerank-max-rank-promotion", type=int, default=DEFAULT_RERANK_MAX_RANK_PROMOTION, help="rerank 后候选最大上升名次。")
    parser.add_argument("--rerank-max-chunks-per-doc", type=int, default=32, help="单文档内部切分最大块数。")
    parser.add_argument("--rerank-overlap-tokens", type=int, default=32, help="单文档内部切分重叠 token 数。")
    parser.set_defaults(rerank_rank_safe=DEFAULT_RERANK_RANK_SAFE)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.opensearch_password:
        raise SystemExit(
            f"缺少 OpenSearch 密码。请设置环境变量 {DEFAULT_OPENSEARCH_PASSWORD_ENV} "
            "或通过 --opensearch-password 传入。"
        )
    if args.mode in {"vector", "hybrid"} and not args.embedding_api_key:
        raise SystemExit(
            f"缺少 Embedding API Key。请设置环境变量 {DEFAULT_EMBEDDING_KEY_ENV} "
            "或通过 --embedding-api-key 传入。"
        )
    if args.rerank and not args.rerank_api_key:
        raise SystemExit(
            f"缺少 Rerank API Key。请设置环境变量 {DEFAULT_EMBEDDING_KEY_ENV} "
            "或通过 --rerank-api-key 传入。"
        )
    if args.rerank and args.rerank_top_n <= 0:
        raise SystemExit("--rerank-top-n 必须大于 0。")
    if args.rerank and not 0 <= args.rerank_model_weight <= 1:
        raise SystemExit("--rerank-model-weight 必须在 0 到 1 之间。")
    if args.rerank and args.rerank_timeout <= 0:
        raise SystemExit("--rerank-timeout 必须大于 0。")
    if args.rerank and args.rerank_min_interval_ms < 0:
        raise SystemExit("--rerank-min-interval-ms 不能小于 0。")
    if args.rerank and args.rerank_max_retries < 0:
        raise SystemExit("--rerank-max-retries 不能小于 0。")
    if args.rerank and args.rerank_max_rank_promotion < 0:
        raise SystemExit("--rerank-max-rank-promotion 不能小于 0。")
    if args.rerank and args.rerank_max_chunks_per_doc <= 0:
        raise SystemExit("--rerank-max-chunks-per-doc 必须大于 0。")
    if args.rerank and not 0 <= args.rerank_overlap_tokens <= 80:
        raise SystemExit("--rerank-overlap-tokens 必须在 0 到 80 之间。")


def main() -> None:
    args = parse_args()
    args.include_full_text = True
    payload = run_search(args)
    case_hits = payload["results"]
    case_docs = {
        item["doc_id"]: {
            **item["case_doc"],
            "full_text": item["case_doc"].get("full_text", ""),
        }
        for item in case_hits
    }
    printable_case_hits = []
    for item in case_hits:
        printable_case_hits.append(
            {
                **{
                    key: value
                    for key, value in item.items()
                    if key not in {"case_doc", "matched_chunks"}
                },
                "matched_chunks": [
                    ChunkHit(
                        chunk_id=chunk["chunk_id"],
                        doc_id=chunk["doc_id"],
                        score=chunk["score"],
                        chunk_text=chunk["chunk_text"],
                        section_type=chunk.get("section_type", ""),
                        section_title=chunk.get("section_title", ""),
                        case_name=chunk.get("case_name", ""),
                        reason=chunk.get("reason", ""),
                        trial_level=chunk.get("trial_level", ""),
                        court_name=chunk.get("court_name", ""),
                        judge_date=chunk.get("judge_date", ""),
                        char_start=chunk.get("char_start"),
                        char_end=chunk.get("char_end"),
                        line_start=chunk.get("line_start"),
                        line_end=chunk.get("line_end"),
                        statutes=chunk.get("statutes", []),
                        negative_tags=chunk.get("negative_tags", []),
                        outcome_tags=chunk.get("outcome_tags", []),
                        match_sources=chunk.get("match_sources", []),
                        raw_scores=chunk.get("raw_scores", {}),
                    )
                    for chunk in item["matched_chunks"]
                ],
            }
        )

    print_results(
        results=printable_case_hits,
        case_docs=case_docs,
        top_cases=args.top_k,
        top_chunks=args.chunk_top_k,
        show_context=args.show_context,
        context_window=args.context_window,
    )

    if args.json_output:
        dump_json(args.json_output, payload)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
