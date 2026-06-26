from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any

from .constants import (
    CASE_KEY_SECTION_TYPES,
    CASE_RERANK_MAX_SELECTED_CHUNKS,
    CASE_RERANK_MODEL_WEIGHT,
    CASE_RERANK_SECTION_BUDGETS,
    CASE_RERANK_SECTION_GROUPS,
    CASE_RERANK_SECTION_ORDER,
    CASE_RERANK_TEXT_LIMIT,
    DEFAULT_RERANK_MAX_RANK_PROMOTION,
    DEFAULT_RERANK_MAX_RETRIES,
    DEFAULT_RERANK_MIN_INTERVAL_MS,
    DEFAULT_RERANK_RANK_SAFE,
    DEFAULT_SOURCE_FIELDS,
    KEY_SECTION_TYPES,
    LEGAL_RERANK_SECTION_TYPES,
    RERANK_CONFLICT_FACTORS,
    RERANK_GUARDRAIL_MAX_CHUNKS,
    RERANK_GUARDRAIL_TEXT_LIMIT,
    RERANK_NEGATION_CUES,
    RERANK_NEGATION_LOOKBACK,
    RERANK_REQUIRED_FACTORS,
    RERANK_SINGLE_CHAR_NEGATION_CUES,
)
from .models import ChunkHit
from .opensearch_client import OpenSearchClient
from .search_queries import source_to_chunk_hit
from .search_results import compact_text
from .utils import safe_int


_LAST_RERANK_REQUEST_AT = 0.0


def build_rerank_passage(hit: ChunkHit) -> str:
    parts: list[str] = []
    if hit.reason:
        parts.append(f"案由：{hit.reason}")
    if hit.trial_level:
        parts.append(f"审级：{hit.trial_level}")
    section_label = hit.section_title or hit.section_type
    if section_label:
        parts.append(f"章节：{section_label}")
    if hit.negative_tags:
        parts.append(f"否定事实标签：{'、'.join(hit.negative_tags)}")
    if hit.outcome_tags:
        parts.append(f"裁判结果标签：{'、'.join(hit.outcome_tags)}")
    parts.append(f"正文：{hit.chunk_text}")
    return "\n".join(parts)

def request_rerank_scores(
    query: str,
    documents: list[str],
    api_key: str,
    model: str,
    endpoint: str,
    timeout: int,
    max_chunks_per_doc: int,
    overlap_tokens: int,
    min_interval_ms: int = DEFAULT_RERANK_MIN_INTERVAL_MS,
    max_retries: int = DEFAULT_RERANK_MAX_RETRIES,
) -> list[dict[str, Any]]:
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
        "return_documents": False,
        "max_chunks_per_doc": max_chunks_per_doc,
        "overlap_tokens": overlap_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    attempts = max(0, max_retries) + 1
    last_error = ""
    for attempt in range(attempts):
        throttle_rerank_request(min_interval_ms)
        request = urllib.request.Request(
            endpoint,
            data=body_bytes,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body)
                results = data.get("results", [])
                if not isinstance(results, list):
                    raise RuntimeError("Rerank response missing results.")
                return results
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"Rerank HTTP {exc.code}: {body}"
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= attempts - 1:
                raise RuntimeError(last_error) from exc
            retry_after = parse_retry_after(exc.headers.get("Retry-After"))
            time.sleep(retry_after if retry_after is not None else retry_delay(attempt))
        except urllib.error.URLError as exc:
            last_error = f"Rerank request failed: {exc}"
            if attempt >= attempts - 1:
                raise RuntimeError(last_error) from exc
            time.sleep(retry_delay(attempt))
    raise RuntimeError(last_error or "Rerank request failed.")

def throttle_rerank_request(min_interval_ms: int) -> None:
    global _LAST_RERANK_REQUEST_AT
    min_interval = max(0, min_interval_ms) / 1000.0
    now = time.monotonic()
    wait_seconds = _LAST_RERANK_REQUEST_AT + min_interval - now
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    _LAST_RERANK_REQUEST_AT = time.monotonic()

def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None

def retry_delay(attempt: int) -> float:
    return min(12.0, 1.5 * (2 ** attempt))

def rerank_hits(
    query: str,
    hits: list[ChunkHit],
    model_name: str,
    api_key: str,
    endpoint: str,
    top_n: int,
    timeout: int,
    max_chunks_per_doc: int,
    overlap_tokens: int,
    min_interval_ms: int = DEFAULT_RERANK_MIN_INTERVAL_MS,
    max_retries: int = DEFAULT_RERANK_MAX_RETRIES,
) -> list[ChunkHit]:
    if not hits:
        return []

    rerank_candidates = hits[: max(1, top_n)]
    documents = [build_rerank_passage(hit) for hit in rerank_candidates]
    results = request_rerank_scores(
        query=query,
        documents=documents,
        api_key=api_key,
        model=model_name,
        endpoint=endpoint,
        timeout=timeout,
        max_chunks_per_doc=max_chunks_per_doc,
        overlap_tokens=overlap_tokens,
        min_interval_ms=min_interval_ms,
        max_retries=max_retries,
    )

    rescored_hits: list[ChunkHit] = []
    for result in results:
        index = safe_int(result.get("index"))
        if index is None or not 0 <= index < len(rerank_candidates):
            continue
        rerank_score = float(result.get("relevance_score") or 0.0)
        hit = rerank_candidates[index]
        rescored_hit = ChunkHit(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            score=rerank_score,
            chunk_text=hit.chunk_text,
            section_type=hit.section_type,
            section_title=hit.section_title,
            case_name=hit.case_name,
            reason=hit.reason,
            trial_level=hit.trial_level,
            court_name=hit.court_name,
            judge_date=hit.judge_date,
            char_start=hit.char_start,
            char_end=hit.char_end,
            line_start=hit.line_start,
            line_end=hit.line_end,
            section_weight=hit.section_weight,
            match_sources=list(hit.match_sources),
            raw_scores=dict(hit.raw_scores),
        )
        rescored_hit.raw_scores["pre_rerank"] = hit.score
        rescored_hit.raw_scores["rerank"] = rerank_score
        if "rerank" not in rescored_hit.match_sources:
            rescored_hit.match_sources.append("rerank")
        rescored_hits.append(rescored_hit)

    if not rescored_hits:
        raise RuntimeError("Rerank returned no valid scored results.")

    rescored_hits.sort(key=lambda item: item.score, reverse=True)
    return rescored_hits

def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value <= min_value:
        return [1.0 for _ in values]
    return [
        (value - min_value) / (max_value - min_value)
        for value in values
    ]

def case_rerank_section_rank(section_type: str) -> int:
    if section_type in CASE_RERANK_SECTION_ORDER:
        return CASE_RERANK_SECTION_ORDER.index(section_type)
    return len(CASE_RERANK_SECTION_ORDER)

def rerank_chunk_key(chunk: ChunkHit) -> str:
    if chunk.chunk_id:
        return chunk.chunk_id
    return f"{chunk.doc_id}:{chunk.section_type}:{chunk.chunk_text[:80]}"

def merge_rerank_chunks(case_hit: dict[str, Any]) -> list[ChunkHit]:
    chunks_by_key: dict[str, ChunkHit] = {}
    chunks = list(case_hit.get("_rerank_chunks") or []) + list(case_hit.get("matched_chunks") or [])
    for chunk in chunks:
        if not isinstance(chunk, ChunkHit):
            continue
        key = rerank_chunk_key(chunk)
        current = chunks_by_key.get(key)
        if current is None:
            chunks_by_key[key] = chunk
            continue
        current.score = max(current.score, chunk.score)
        for source in chunk.match_sources:
            if source not in current.match_sources:
                current.match_sources.append(source)
        current.raw_scores.update(chunk.raw_scores)
    return sorted(
        chunks_by_key.values(),
        key=lambda item: (
            case_rerank_section_rank(item.section_type),
            -item.score,
        ),
    )

def case_hit_sections(case_hit: dict[str, Any]) -> set[str]:
    sections = {
        str(section)
        for section in case_hit.get("matched_sections", [])
        if section
    }
    for chunk in list(case_hit.get("_rerank_chunks") or []) + list(case_hit.get("matched_chunks") or []):
        if isinstance(chunk, ChunkHit) and chunk.section_type:
            sections.add(chunk.section_type)
    return sections

def contains_any(text: str, terms: list[str]) -> bool:
    return any(term and term in text for term in terms)

def has_negation_before(text: str, start_index: int) -> bool:
    window = text[max(0, start_index - RERANK_NEGATION_LOOKBACK):start_index]
    compact_window = re.sub(r"\s+", "", window)
    if any(cue in compact_window for cue in RERANK_NEGATION_CUES):
        return True
    return any(compact_window.endswith(cue) for cue in RERANK_SINGLE_CHAR_NEGATION_CUES)

def positive_term_hits(text: str, terms: list[str]) -> list[str]:
    hits: list[str] = []
    for term in terms:
        if not term:
            continue
        for match in re.finditer(re.escape(term), text):
            if has_negation_before(text, match.start()):
                continue
            hits.append(term)
            break
    return hits

def contains_positive_any(text: str, terms: list[str]) -> bool:
    return bool(positive_term_hits(text, terms))

def case_guardrail_text(case_hit: dict[str, Any]) -> str:
    parts = [
        str(case_hit.get("case_name") or ""),
        str(case_hit.get("reason") or ""),
    ]
    chunks = sorted(
        merge_rerank_chunks(case_hit),
        key=lambda item: (
            0 if item.section_type in CASE_KEY_SECTION_TYPES else 1,
            case_rerank_section_rank(item.section_type),
            -item.score,
        ),
    )
    for chunk in chunks[:RERANK_GUARDRAIL_MAX_CHUNKS]:
        parts.append(chunk.section_title or chunk.section_type or "")
        parts.append(chunk.chunk_text)
    return compact_text(" ".join(parts), limit=RERANK_GUARDRAIL_TEXT_LIMIT)

def guardrail_factor_applies(query_text: str, factor: dict[str, Any]) -> bool:
    all_terms = factor.get("query_all", [])
    if all_terms and not all(term and term in query_text for term in all_terms):
        return False
    return contains_any(query_text, factor["query_any"])

def rerank_guardrail_adjustment(query: str, case_hit: dict[str, Any]) -> dict[str, Any]:
    query_text = compact_text(query, limit=RERANK_GUARDRAIL_TEXT_LIMIT)
    doc_text = case_guardrail_text(case_hit)
    missing: list[str] = []
    conflicts: list[str] = []
    penalty = 0.0
    matched_required = 0

    for factor in RERANK_REQUIRED_FACTORS:
        if not guardrail_factor_applies(query_text, factor):
            continue
        if contains_positive_any(doc_text, factor["doc_any"]):
            matched_required += 1
            continue
        missing.append(str(factor["name"]))
        penalty += float(factor["penalty"])

    for factor in RERANK_CONFLICT_FACTORS:
        if not guardrail_factor_applies(query_text, factor):
            continue
        if not contains_positive_any(doc_text, factor["doc_any"]):
            continue
        absent_terms = factor.get("doc_required_absent", [])
        if absent_terms and contains_positive_any(doc_text, absent_terms):
            continue
        conflicts.append(str(factor["name"]))
        penalty += float(factor["penalty"])

    if matched_required and not missing and not conflicts:
        bonus = min(0.02, 0.008 * matched_required)
    else:
        bonus = 0.0

    return {
        "penalty": min(0.35, penalty),
        "bonus": bonus,
        "missing": missing,
        "conflicts": conflicts,
    }

def rerank_structure_adjustment(case_hit: dict[str, Any]) -> float:
    sections = case_hit_sections(case_hit)
    legal_sections = sections & LEGAL_RERANK_SECTION_TYPES
    weak_only_sections = sections and not legal_sections and sections <= {"facts", "claims", "defense"}

    adjustment = min(0.06, 0.025 * len(legal_sections))
    if {"fine_issue", "focus"} <= sections:
        adjustment += 0.02
    if "reasoning" in sections and (sections & {"fine_issue", "focus"}):
        adjustment += 0.02
    if weak_only_sections:
        adjustment -= 0.08
    elif "facts" in sections and not legal_sections:
        adjustment -= 0.04
    return adjustment

def rerank_allowed_rank(original_rank: int, max_rank_promotion: int) -> int:
    if original_rank <= 20:
        return max(1, original_rank - max_rank_promotion)
    if original_rank <= 50:
        return max(11, original_rank - max_rank_promotion)
    return max(21, original_rank - max_rank_promotion)

def select_case_rerank_chunks(case_hit: dict[str, Any]) -> list[ChunkHit]:
    chunks = merge_rerank_chunks(case_hit)
    selected: list[ChunkHit] = []
    selected_keys: set[str] = set()

    for section_type in CASE_RERANK_SECTION_ORDER:
        candidates = [chunk for chunk in chunks if chunk.section_type == section_type]
        if not candidates:
            continue
        chunk = max(candidates, key=lambda item: item.score)
        selected.append(chunk)
        selected_keys.add(rerank_chunk_key(chunk))
        if len(selected) >= CASE_RERANK_MAX_SELECTED_CHUNKS:
            return selected

    scored_chunks = sorted(
        chunks,
        key=lambda item: (
            0 if item.section_type in LEGAL_RERANK_SECTION_TYPES else 1,
            -item.score,
            case_rerank_section_rank(item.section_type),
        ),
    )
    for chunk in scored_chunks:
        if len(selected) >= CASE_RERANK_MAX_SELECTED_CHUNKS:
            break
        key = rerank_chunk_key(chunk)
        if key in selected_keys:
            continue
        selected.append(chunk)
        selected_keys.add(key)
    return selected

def build_case_rerank_passage(case_hit: dict[str, Any]) -> str:
    parts: list[str] = []
    meta_parts: list[str] = []
    if case_hit.get("case_name"):
        meta_parts.append(f"案名：{case_hit['case_name']}")
    if case_hit.get("reason"):
        meta_parts.append(f"案由：{case_hit['reason']}")
    if case_hit.get("trial_level"):
        meta_parts.append(f"审级：{case_hit['trial_level']}")
    if meta_parts:
        parts.append("【案件类型】\n" + "\n".join(meta_parts))

    grouped_parts: dict[str, list[str]] = defaultdict(list)
    for chunk in select_case_rerank_chunks(case_hit):
        section_type = chunk.section_type or ""
        section_group = CASE_RERANK_SECTION_GROUPS.get(section_type, "【相关片段】")
        section_label = chunk.section_title or section_type or "片段"
        budget = CASE_RERANK_SECTION_BUDGETS.get(section_type, 360)
        text = compact_text(chunk.chunk_text, limit=budget)
        if text:
            grouped_parts[section_group].append(f"{section_label}：{text}")

    group_order = [
        "【案件画像】",
        "【核心争议】",
        "【裁判规则】",
        "【关键事实】",
        "【裁判结果】",
        "【诉请摘要】",
        "【抗辩摘要】",
        "【相关片段】",
    ]
    for group_name in group_order:
        entries = grouped_parts.get(group_name, [])
        if entries:
            parts.append(group_name + "\n" + "\n".join(entries))

    return compact_text("\n\n".join(parts), limit=CASE_RERANK_TEXT_LIMIT)

def fetch_case_key_chunks(
    client: OpenSearchClient,
    index_name: str,
    doc_ids: list[str],
    size_per_doc: int = 8,
) -> dict[str, list[ChunkHit]]:
    if not doc_ids:
        return {}

    body = {
        "size": max(1, len(doc_ids) * size_per_doc),
        "_source": DEFAULT_SOURCE_FIELDS,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {"doc_id": doc_ids}},
                    {"terms": {"section_type": list(CASE_KEY_SECTION_TYPES)}},
                ]
            }
        },
        "sort": [
            {"doc_id": {"order": "asc"}},
            {"section_index": {"order": "asc"}},
            {"chunk_index_in_section": {"order": "asc"}},
        ],
    }
    response = client.request("POST", f"/{urllib.parse.quote(index_name)}/_search", body)
    grouped: dict[str, list[ChunkHit]] = defaultdict(list)
    for hit in response.get("hits", {}).get("hits", []):
        chunk = source_to_chunk_hit(hit, "case_key")
        if chunk.doc_id:
            grouped[chunk.doc_id].append(chunk)
    for chunks in grouped.values():
        chunks.sort(
            key=lambda item: (
                CASE_RERANK_SECTION_ORDER.index(item.section_type)
                if item.section_type in CASE_RERANK_SECTION_ORDER
                else len(CASE_RERANK_SECTION_ORDER),
                -item.score,
            )
        )
    return grouped

def attach_case_key_chunks(
    client: OpenSearchClient,
    index_name: str,
    case_hits: list[dict[str, Any]],
    top_n: int,
) -> None:
    doc_ids = [item["doc_id"] for item in case_hits[:top_n] if item.get("doc_id")]
    try:
        key_chunks = fetch_case_key_chunks(client, index_name, doc_ids)
    except Exception:
        return
    for item in case_hits[:top_n]:
        chunks = key_chunks.get(item.get("doc_id"), [])
        if chunks:
            item["_rerank_chunks"] = chunks

def apply_rank_safe_rerank(
    rescored_candidates: list[dict[str, Any]],
    max_rank_promotion: int,
) -> list[dict[str, Any]]:
    if max_rank_promotion <= 0:
        return rescored_candidates

    for new_rank, item in enumerate(rescored_candidates, start=1):
        original_rank = int(item.get("hybrid_rank") or 10**9)
        allowed_rank = rerank_allowed_rank(original_rank, max_rank_promotion)
        over_promotion = max(0, allowed_rank - new_rank)
        sections = case_hit_sections(item)
        if new_rank <= 10 and not (sections & LEGAL_RERANK_SECTION_TYPES):
            over_promotion += 3
        if new_rank <= 10 and (
            item.get("rerank_guardrail_missing") or item.get("rerank_guardrail_conflicts")
        ):
            over_promotion += 4
        if over_promotion:
            penalty = min(0.80, 0.025 * over_promotion)
            item["rank_safe_penalty"] = penalty
            item["rank_safe_allowed_rank"] = allowed_rank
            item["case_score"] = float(item.get("case_score") or 0.0) - penalty

    return sorted(rescored_candidates, key=lambda item: item["case_score"], reverse=True)

def rerank_case_hits(
    query: str,
    case_hits: list[dict[str, Any]],
    model_name: str,
    api_key: str,
    endpoint: str,
    top_n: int,
    timeout: int,
    max_chunks_per_doc: int,
    overlap_tokens: int,
    model_weight: float = CASE_RERANK_MODEL_WEIGHT,
    min_interval_ms: int = DEFAULT_RERANK_MIN_INTERVAL_MS,
    max_retries: int = DEFAULT_RERANK_MAX_RETRIES,
    rank_safe: bool = DEFAULT_RERANK_RANK_SAFE,
    max_rank_promotion: int = DEFAULT_RERANK_MAX_RANK_PROMOTION,
    guardrail_query: str | None = None,
) -> list[dict[str, Any]]:
    if not case_hits:
        return []

    for rank, item in enumerate(case_hits, start=1):
        item.setdefault("hybrid_rank", rank)

    rerank_count = min(len(case_hits), max(1, top_n))
    rerank_candidates = case_hits[:rerank_count]
    tail = case_hits[rerank_count:]
    documents = [build_case_rerank_passage(item) for item in rerank_candidates]
    results = request_rerank_scores(
        query=query,
        documents=documents,
        api_key=api_key,
        model=model_name,
        endpoint=endpoint,
        timeout=timeout,
        max_chunks_per_doc=max_chunks_per_doc,
        overlap_tokens=overlap_tokens,
        min_interval_ms=min_interval_ms,
        max_retries=max_retries,
    )

    rerank_scores: dict[int, float] = {}
    for result in results:
        index = safe_int(result.get("index"))
        if index is None or not 0 <= index < len(rerank_candidates):
            continue
        rerank_scores[index] = float(result.get("relevance_score") or 0.0)

    if not rerank_scores:
        raise RuntimeError("Rerank returned no valid scored results.")

    hybrid_values = [float(item.get("case_score") or 0.0) for item in rerank_candidates]
    model_values = [
        rerank_scores.get(index, min(rerank_scores.values()))
        for index in range(len(rerank_candidates))
    ]
    hybrid_norm = normalize_scores(hybrid_values)
    model_norm = normalize_scores(model_values)
    bounded_model_weight = min(1.0, max(0.0, model_weight))
    hybrid_weight = 1.0 - bounded_model_weight

    rescored_candidates: list[dict[str, Any]] = []
    for index, item in enumerate(rerank_candidates):
        original_score = float(item.get("case_score") or 0.0)
        rerank_score = model_values[index]
        fused_score = (
            hybrid_weight * hybrid_norm[index]
            + bounded_model_weight * model_norm[index]
        )
        structure_adjustment = rerank_structure_adjustment(item)
        guardrail = rerank_guardrail_adjustment(guardrail_query or query, item)
        guardrail_adjustment = float(guardrail["bonus"]) - float(guardrail["penalty"])
        fused_score = max(0.0, fused_score + structure_adjustment + guardrail_adjustment)
        updated = dict(item)
        updated["case_score"] = fused_score
        updated["hybrid_case_score"] = original_score
        updated["rerank_score"] = rerank_score
        updated["rerank_fused_score"] = fused_score
        updated["rerank_structure_adjustment"] = structure_adjustment
        updated["rerank_guardrail_adjustment"] = guardrail_adjustment
        updated["rerank_guardrail_penalty"] = guardrail["penalty"]
        updated["rerank_guardrail_bonus"] = guardrail["bonus"]
        updated["rerank_guardrail_missing"] = guardrail["missing"]
        updated["rerank_guardrail_conflicts"] = guardrail["conflicts"]
        updated["rerank_model_weight"] = bounded_model_weight
        updated["rerank_hybrid_weight"] = hybrid_weight
        rescored_candidates.append(updated)

    rescored_candidates.sort(key=lambda item: item["case_score"], reverse=True)
    if rank_safe:
        rescored_candidates = apply_rank_safe_rerank(
            rescored_candidates,
            max_rank_promotion=max_rank_promotion,
        )
    return rescored_candidates + tail
