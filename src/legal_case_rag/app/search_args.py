from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any


def bool_param(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def int_param(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_param(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def str_param(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def normalize_sequence(value: Any) -> list[Any]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def dict_param(value: Any, default: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = dict(default or {})
    if value in (None, ""):
        return fallback
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return fallback
        return parsed if isinstance(parsed, dict) else fallback
    return fallback


def build_search_args(
    payload: dict[str, Any],
    *,
    default_config: dict[str, Any],
    retrieval_module: Any,
    verify_ssl_default: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        query=str_param(payload.get("query")),
        mode=str_param(payload.get("mode"), default_config.get("mode", "hybrid")) or default_config.get("mode", "hybrid"),
        reason=str_param(payload.get("reason"), default_config.get("reason", "")),
        trial_level=str_param(payload.get("trial_level"), default_config.get("trial_level", "")),
        court_name=str_param(payload.get("court_name"), default_config.get("court_name", "")),
        section_type=str_param(payload.get("section_type"), default_config.get("section_type", "")),
        judge_date_from=str_param(payload.get("judge_date_from"), default_config.get("judge_date_from", "")),
        judge_date_to=str_param(payload.get("judge_date_to"), default_config.get("judge_date_to", "")),
        top_k=int_param(payload.get("top_k"), default_config.get("top_k", 8)),
        chunk_top_k=int_param(payload.get("chunk_top_k"), default_config.get("chunk_top_k", 3)),
        candidate_size=int_param(payload.get("candidate_size"), default_config.get("candidate_size", 80)),
        show_context=bool_param(payload.get("show_context"), default_config.get("show_context", True)),
        context_window=int_param(payload.get("context_window"), default_config.get("context_window", 180)),
        query_profile=bool_param(payload.get("query_profile"), default_config.get("query_profile", True)),
        query_profile_boost=bool_param(payload.get("query_profile_boost"), default_config.get("query_profile_boost", True)),
        route_weight_overrides=dict_param(
            payload.get("route_weight_overrides"),
            default_config.get("route_weight_overrides", {}),
        ),
        json_output="",
        opensearch_url=os.getenv("OPENSEARCH_URL", retrieval_module.DEFAULT_OPENSEARCH_URL),
        opensearch_username=os.getenv(
            "OPENSEARCH_USERNAME",
            retrieval_module.DEFAULT_OPENSEARCH_USERNAME,
        ),
        opensearch_password=os.getenv(retrieval_module.DEFAULT_OPENSEARCH_PASSWORD_ENV),
        chunk_index=os.getenv("LEGAL_CHUNK_INDEX", retrieval_module.DEFAULT_CHUNK_INDEX),
        case_index=os.getenv("LEGAL_CASE_INDEX", retrieval_module.DEFAULT_CASE_INDEX),
        verify_ssl=bool_param(payload.get("verify_ssl"), verify_ssl_default),
        timeout=int_param(payload.get("timeout"), 30),
        embedding_api_key=os.getenv(retrieval_module.DEFAULT_EMBEDDING_KEY_ENV),
        embedding_model=str_param(
            payload.get("embedding_model"),
            retrieval_module.DEFAULT_EMBEDDING_MODEL,
        ),
        embedding_url=str_param(
            payload.get("embedding_url"),
            retrieval_module.DEFAULT_EMBEDDING_URL,
        ),
        embedding_timeout=int_param(payload.get("embedding_timeout"), 60),
        rerank=bool_param(payload.get("rerank"), default_config.get("rerank", True)),
        rerank_model=str_param(
            payload.get("rerank_model"),
            retrieval_module.DEFAULT_RERANK_MODEL,
        ),
        rerank_top_n=int_param(payload.get("rerank_top_n"), default_config.get("rerank_top_n", 20)),
        rerank_model_weight=float_param(
            payload.get("rerank_model_weight"),
            default_config.get("rerank_model_weight", 0.35),
        ),
        rerank_min_interval_ms=int_param(
            payload.get("rerank_min_interval_ms"),
            default_config.get("rerank_min_interval_ms", 1200),
        ),
        rerank_max_retries=int_param(
            payload.get("rerank_max_retries"),
            default_config.get("rerank_max_retries", 3),
        ),
        rerank_rank_safe=bool_param(
            payload.get("rerank_rank_safe"),
            default_config.get("rerank_rank_safe", True),
        ),
        rerank_max_rank_promotion=int_param(
            payload.get("rerank_max_rank_promotion"),
            default_config.get("rerank_max_rank_promotion", 30),
        ),
        rerank_api_key=os.getenv(retrieval_module.DEFAULT_EMBEDDING_KEY_ENV),
        rerank_url=str_param(
            payload.get("rerank_url"),
            retrieval_module.DEFAULT_RERANK_URL,
        ),
        rerank_timeout=int_param(payload.get("rerank_timeout"), 120),
        rerank_max_chunks_per_doc=int_param(payload.get("rerank_max_chunks_per_doc"), 32),
        rerank_overlap_tokens=int_param(payload.get("rerank_overlap_tokens"), 32),
        include_full_text=False,
    )
