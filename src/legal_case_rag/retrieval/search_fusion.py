from __future__ import annotations

from collections import defaultdict
from typing import Any

from .constants import CASE_RERANK_SECTION_ORDER, KEY_SECTION_TYPES, SECTION_WEIGHTS
from .models import ChunkHit
from .query_profile import QueryProfile, profile_match_bonus


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[ChunkHit]],
    weights: dict[str, float] | None = None,
    k: int = 10,
) -> list[ChunkHit]:
    weights = weights or {}
    fused: dict[str, ChunkHit] = {}

    for source_name, hits in ranked_lists.items():
        weight = float(weights.get(source_name, 1.0))
        for rank, hit in enumerate(hits, start=1):
            rrf_score = weight / (k + rank)
            if hit.chunk_id not in fused:
                fused[hit.chunk_id] = ChunkHit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    score=0.0,
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
                    match_sources=[],
                    raw_scores={},
                )
            merged = fused[hit.chunk_id]
            merged.score += rrf_score
            merged.raw_scores[source_name] = hit.raw_scores.get(source_name, hit.score)
            if source_name not in merged.match_sources:
                merged.match_sources.append(source_name)

    return sorted(fused.values(), key=lambda item: item.score, reverse=True)

def clone_hit_with_source(hit: ChunkHit, source_name: str) -> ChunkHit:
    raw_score = hit.raw_scores.get(hit.match_sources[0], hit.score) if hit.match_sources else hit.score
    return ChunkHit(
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        score=hit.score,
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
        statutes=list(hit.statutes),
        section_weight=hit.section_weight,
        negative_tags=list(hit.negative_tags),
        outcome_tags=list(hit.outcome_tags),
        match_sources=[source_name],
        raw_scores={source_name: raw_score},
    )

def apply_query_profile_bonus(hits: list[ChunkHit], profile: QueryProfile) -> list[ChunkHit]:
    for hit in hits:
        bonus = profile_match_bonus(
            profile,
            chunk_text=hit.chunk_text,
            reason=hit.reason,
            section_type=hit.section_type,
            statutes=hit.statutes,
        )
        if bonus:
            hit.score += bonus
            hit.raw_scores["query_profile_bonus"] = bonus
            if "query_profile" not in hit.match_sources:
                hit.match_sources.append("query_profile")
    return sorted(hits, key=lambda item: item.score, reverse=True)

def route_case_ranking(
    hits: list[ChunkHit],
    source_name: str,
    top_chunks_per_case: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ChunkHit]] = defaultdict(list)
    for hit in hits:
        if hit.doc_id:
            grouped[hit.doc_id].append(hit)

    case_rows: list[dict[str, Any]] = []
    for doc_id, doc_hits in grouped.items():
        doc_hits.sort(key=lambda item: item.score, reverse=True)
        top_hits = doc_hits[:top_chunks_per_case]
        key_hits = [hit for hit in doc_hits if hit.section_type in KEY_SECTION_TYPES]
        top_key_hits = [hit for hit in top_hits if hit.section_type in {"fine_tags", "fine_rule", "focus_tags", "focus_analysis"}]
        weighted_scores = [hit.score * section_weight(hit) for hit in top_hits]
        best_score = max(weighted_scores) if weighted_scores else 0.0
        section_bonus = min(0.20, 0.05 * len({hit.section_type for hit in key_hits}))
        top_key_bonus = min(0.10, 0.05 * len({hit.section_type for hit in top_key_hits}))
        multi_chunk_bonus = min(0.035, 0.007 * max(0, len(doc_hits) - 1))
        route_score = best_score + section_bonus + top_key_bonus + multi_chunk_bonus
        case_rows.append(
            {
                "doc_id": doc_id,
                "route_score": route_score,
                "hits": doc_hits,
                "source_name": source_name,
            }
        )

    case_rows.sort(key=lambda item: item["route_score"], reverse=True)
    return case_rows

def merge_case_hit_chunks(existing: list[ChunkHit], incoming: list[ChunkHit]) -> list[ChunkHit]:
    by_id: dict[str, ChunkHit] = {hit.chunk_id: hit for hit in existing}
    for hit in incoming:
        current = by_id.get(hit.chunk_id)
        if current is None:
            by_id[hit.chunk_id] = hit
            continue
        current.score = max(current.score, hit.score)
        for source in hit.match_sources:
            if source not in current.match_sources:
                current.match_sources.append(source)
        current.raw_scores.update(hit.raw_scores)
    return sorted(by_id.values(), key=lambda item: item.score, reverse=True)

def case_level_reciprocal_rank_fusion(
    ranked_lists: dict[str, list[ChunkHit]],
    weights: dict[str, float] | None = None,
    top_chunks_per_case: int = 3,
    k: int = 10,
) -> list[dict[str, Any]]:
    weights = weights or {}
    fused: dict[str, dict[str, Any]] = {}

    for source_name, hits in ranked_lists.items():
        route_cases = route_case_ranking(hits, source_name, top_chunks_per_case)
        weight = float(weights.get(source_name, 1.0))
        for rank, route_case in enumerate(route_cases, start=1):
            doc_id = route_case["doc_id"]
            rrf_score = weight / (k + rank)
            doc_hits = route_case["hits"]
            if doc_id not in fused:
                top_hit = doc_hits[0]
                fused[doc_id] = {
                    "doc_id": doc_id,
                    "case_score": 0.0,
                    "reason": top_hit.reason,
                    "trial_level": top_hit.trial_level,
                    "court_name": top_hit.court_name,
                    "judge_date": top_hit.judge_date,
                    "case_name": top_hit.case_name,
                    "matched_chunks": [],
                    "_all_chunks": [],
                    "_route_names": set(),
                    "_route_types": set(),
                    "_route_scores": {},
                }
            entry = fused[doc_id]
            entry["case_score"] += rrf_score
            entry["_route_names"].add(source_name)
            entry["_route_types"].add(source_name.split("_", 1)[0])
            entry["_route_scores"][source_name] = route_case["route_score"]
            entry["_all_chunks"] = merge_case_hit_chunks(entry["_all_chunks"], doc_hits)

    case_hits: list[dict[str, Any]] = []
    for entry in fused.values():
        all_chunks: list[ChunkHit] = entry["_all_chunks"]
        all_chunks.sort(key=lambda item: item.score, reverse=True)
        hit_sections = {hit.section_type for hit in all_chunks if hit.section_type}
        key_sections = hit_sections & KEY_SECTION_TYPES
        key_section_bonus = min(0.18, 0.055 * len(key_sections))
        fine_focus_bonus = min(0.10, 0.05 * len(hit_sections & {"fine_tags", "fine_rule", "focus_tags", "focus_analysis"}))
        dual_channel_bonus = 0.05 if {"bm25", "vector"} <= entry["_route_types"] else 0.0
        route_coverage_bonus = min(0.10, 0.015 * max(0, len(entry["_route_names"]) - 1))
        multi_chunk_bonus = min(0.035, 0.006 * max(0, len(all_chunks) - 1))
        case_score = (
            float(entry["case_score"])
            + key_section_bonus
            + fine_focus_bonus
            + dual_channel_bonus
            + route_coverage_bonus
            + multi_chunk_bonus
        )
        rerank_chunks = sorted(
            all_chunks,
            key=lambda item: (
                0 if item.section_type in KEY_SECTION_TYPES else 1,
                CASE_RERANK_SECTION_ORDER.index(item.section_type)
                if item.section_type in CASE_RERANK_SECTION_ORDER
                else len(CASE_RERANK_SECTION_ORDER),
                -item.score,
            ),
        )[:8]
        case_hits.append(
            {
                "doc_id": entry["doc_id"],
                "case_score": case_score,
                "reason": entry["reason"],
                "trial_level": entry["trial_level"],
                "court_name": entry["court_name"],
                "judge_date": entry["judge_date"],
                "case_name": entry["case_name"],
                "matched_chunks": all_chunks[:top_chunks_per_case],
                "_rerank_chunks": rerank_chunks,
                "hit_count": len(all_chunks),
                "matched_sections": sorted(hit_sections),
                "route_count": len(entry["_route_names"]),
                "route_names": sorted(entry["_route_names"]),
            }
        )

    case_hits.sort(key=lambda item: item["case_score"], reverse=True)
    return case_hits

def aggregate_case_hits(
    chunk_hits: list[ChunkHit],
    top_chunks_per_case: int = 3,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ChunkHit]] = defaultdict(list)
    for hit in chunk_hits:
        if hit.doc_id:
            grouped[hit.doc_id].append(hit)

    case_hits: list[dict[str, Any]] = []
    for doc_id, hits in grouped.items():
        hits.sort(key=lambda item: item.score, reverse=True)
        top_hits = hits[:top_chunks_per_case]
        rerank_chunks = sorted(
            hits,
            key=lambda item: (
                0 if item.section_type in KEY_SECTION_TYPES else 1,
                CASE_RERANK_SECTION_ORDER.index(item.section_type)
                if item.section_type in CASE_RERANK_SECTION_ORDER
                else len(CASE_RERANK_SECTION_ORDER),
                -item.score,
            ),
        )[:8]
        all_sources = {
            source
            for hit in hits
            for source in hit.match_sources
        }
        hit_sections = {hit.section_type for hit in hits if hit.section_type}
        top_sections = {hit.section_type for hit in top_hits if hit.section_type}
        key_section_hits = hit_sections & KEY_SECTION_TYPES
        weighted_scores = [
            item.score * section_weight(item) for item in top_hits
        ]
        max_score = max(weighted_scores) if weighted_scores else 0.0
        sum_score = sum(weighted_scores[:3])
        key_section_bonus = min(0.18, 0.055 * len(key_section_hits))
        top_key_bonus = min(0.08, 0.04 * len(top_sections & {"fine_tags", "fine_rule", "focus_tags", "focus_analysis"}))
        multi_chunk_bonus = min(0.08, 0.018 * max(0, len(hits) - 1))
        route_coverage_bonus = min(0.12, 0.018 * max(0, len(all_sources) - 1))
        dual_source_bonus = min(
            0.08,
            0.02 * sum(1 for item in top_hits if len(item.match_sources) >= 2),
        )
        case_score = (
            max_score * 0.55
            + sum_score * 0.35
            + key_section_bonus
            + top_key_bonus
            + multi_chunk_bonus
            + route_coverage_bonus
            + dual_source_bonus
        )

        case_hits.append(
            {
                "doc_id": doc_id,
                "case_score": case_score,
                "reason": top_hits[0].reason if top_hits else "",
                "trial_level": top_hits[0].trial_level if top_hits else "",
                "court_name": top_hits[0].court_name if top_hits else "",
                "judge_date": top_hits[0].judge_date if top_hits else "",
                "case_name": top_hits[0].case_name if top_hits else "",
                "matched_chunks": top_hits,
                "_rerank_chunks": rerank_chunks,
                "hit_count": len(hits),
                "matched_sections": sorted(hit_sections),
                "route_count": len(all_sources),
            }
        )

    case_hits.sort(key=lambda item: item["case_score"], reverse=True)
    return case_hits

def section_weight(hit: ChunkHit) -> float:
    if hit.section_weight is not None:
        return hit.section_weight
    return SECTION_WEIGHTS.get(hit.section_type or "", 0.6)
