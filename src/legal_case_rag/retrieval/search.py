from __future__ import annotations

import argparse
import base64
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .query_profile import (
    QueryProfile,
    build_query_profile,
    build_query_routes,
    build_rerank_query,
    extract_negative_tags,
    extract_outcome_tags,
    profile_match_bonus,
)


DEFAULT_OPENSEARCH_URL = "https://localhost:9200"
DEFAULT_OPENSEARCH_USERNAME = "admin"
DEFAULT_OPENSEARCH_PASSWORD_ENV = "OPENSEARCH_PASSWORD"
DEFAULT_CHUNK_INDEX = "caselaw_benchmark_chunks_v1"
DEFAULT_CASE_INDEX = "caselaw_benchmark_cases_v1"
DEFAULT_EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_KEY_ENV = "SILICONFLOW_API_KEY"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"

SECTION_WEIGHTS = {
    "fine_issue": 1.55,
    "focus": 1.45,
    "case_profile": 1.20,
    "reasoning": 1.15,
    "facts": 1.00,
    "claims": 0.75,
    "defense": 0.75,
    "judgment": 0.70,
    "header": 0.40,
    "statutes": 0.45,
}

KEY_SECTION_TYPES = {"fine_issue", "focus", "reasoning", "facts"}
LEGAL_RERANK_SECTION_TYPES = {"fine_issue", "focus", "reasoning"}
CASE_KEY_SECTION_TYPES = {"case_profile", "fine_issue", "focus", "reasoning", "facts"}
CASE_RERANK_SECTION_ORDER = [
    "fine_issue",
    "focus",
    "reasoning",
    "facts",
    "case_profile",
    "claims",
    "judgment",
]
CASE_RERANK_SECTION_BUDGETS = {
    "fine_issue": 700,
    "focus": 700,
    "reasoning": 900,
    "case_profile": 420,
    "facts": 320,
    "judgment": 240,
    "claims": 220,
    "defense": 220,
}
CASE_RERANK_SECTION_GROUPS = {
    "case_profile": "【案件画像】",
    "fine_issue": "【核心争议】",
    "focus": "【核心争议】",
    "reasoning": "【裁判规则】",
    "facts": "【关键事实】",
    "judgment": "【裁判结果】",
    "claims": "【诉请摘要】",
    "defense": "【抗辩摘要】",
}
CASE_RERANK_MAX_SELECTED_CHUNKS = 6
RERANK_GUARDRAIL_TEXT_LIMIT = 5000
RERANK_GUARDRAIL_MAX_CHUNKS = 12
RERANK_NEGATION_LOOKBACK = 18
RERANK_NEGATION_CUES = [
    "完全未涉及",
    "并未涉及",
    "未涉及",
    "不涉及",
    "未提及",
    "未载明",
    "未显示",
    "未体现",
    "未说明",
    "未论及",
    "未主张",
    "未请求",
    "没有",
    "并无",
    "均无",
    "不存在",
    "不包含",
    "未见",
    "未发现",
]
RERANK_SINGLE_CHAR_NEGATION_CUES = ["无"]
RERANK_REQUIRED_FACTORS = [
    {
        "name": "ownership_retention",
        "query_any": ["所有权保留", "取回权", "留置所有权"],
        "doc_any": ["所有权保留", "取回权", "返还货物", "留置所有权"],
        "penalty": 0.18,
    },
    {
        "name": "deposit_penalty",
        "query_any": ["定金罚则", "成约定金", "违约定金", "返还定金"],
        "doc_any": ["定金罚则", "成约定金", "违约定金", "返还定金", "没收定金", "双倍返还定金"],
        "penalty": 0.12,
    },
    {
        "name": "invoice_dispute",
        "query_any": [
            "发票争议",
            "未开票",
            "未开发票",
            "未开具发票",
            "拒开发票",
            "开票义务",
            "开具发票",
            "补开发票",
            "发票问题",
        ],
        "doc_any": ["发票争议", "未开票", "开票", "发票", "增值税发票"],
        "penalty": 0.06,
    },
    {
        "name": "third_party_supply",
        "query_any": ["第三方供货", "第三方代为供货", "第三人供货", "代为供货", "指示第三方"],
        "doc_any": ["第三方供货", "第三方代为供货", "第三人供货", "代为供货", "指示第三方", "第三人履行"],
        "penalty": 0.10,
    },
    {
        "name": "reconciliation_silence",
        "query_all": ["对账单"],
        "query_any": ["未在合理期限", "未提出异议", "未及时提出异议", "视为认可", "结算依据"],
        "doc_any": ["对账单", "未提出异议", "未在合理期限", "结算依据", "对账", "结算单"],
        "penalty": 0.08,
    },
    {
        "name": "seal_dispute",
        "query_any": ["偷盖", "冒盖", "私盖", "收货确认单", "合同外供货"],
        "doc_any": ["偷盖", "冒盖", "私盖", "印章", "收货确认单", "合同外供货", "盖章", "公章"],
        "penalty": 0.10,
    },
    {
        "name": "oral_contract_evidence",
        "query_any": ["口头买卖", "微信聊天记录", "仅凭微信", "无书面合同"],
        "doc_any": ["口头买卖", "微信聊天记录", "无书面合同", "聊天记录", "微信"],
        "penalty": 0.10,
    },
    {
        "name": "termination_time",
        "query_any": ["解除时间", "解除时间如何认定", "起诉状副本送达", "送达时解除"],
        "doc_any": ["解除时间", "起诉状副本送达", "送达时解除", "解除通知", "合同解除时间"],
        "penalty": 0.10,
    },
    {
        "name": "agency_or_third_payment",
        "query_any": ["委托他人", "代付", "案外人", "以自己名义"],
        "doc_any": ["委托他人", "代付", "案外人", "以自己名义", "第三人付款", "第三人代付", "代为支付"],
        "penalty": 0.04,
    },
]
RERANK_CONFLICT_FACTORS = [
    {
        "name": "defective_delivery_vs_non_delivery",
        "query_any": ["瑕疵", "异物", "碎骨", "淤血", "淋巴", "质量"],
        "doc_any": ["未交货", "未发货", "未交付", "迟延交货", "逾期交货", "未履行交货"],
        "doc_required_absent": ["瑕疵", "异物", "碎骨", "淤血", "淋巴", "质量"],
        "penalty": 0.14,
    },
    {
        "name": "non_delivery_vs_defective_delivery",
        "query_any": ["未交货", "未发货", "未交付"],
        "doc_any": ["质量异议", "瑕疵", "异物", "质量问题"],
        "doc_required_absent": ["未交货", "未发货", "未交付"],
        "penalty": 0.10,
    },
    {
        "name": "buyer_default_vs_seller_default",
        "query_any": ["未按约提车", "拒收", "价格过高", "买方拒收", "买方未提货", "买方未按约"],
        "doc_any": ["卖方未交货", "出卖人未交货", "卖方未发货", "出卖人未发货", "卖方根本违约"],
        "penalty": 0.08,
    },
    {
        "name": "collateral_invoice_only",
        "query_any": ["定金罚则", "所有权保留", "解除时间", "第三方供货"],
        "doc_any": ["发票", "开票"],
        "doc_required_absent": ["定金罚则", "所有权保留", "解除时间", "第三方供货", "第三方代为供货"],
        "penalty": 0.04,
    },
]
CASE_RERANK_HYBRID_WEIGHT = 0.65
CASE_RERANK_MODEL_WEIGHT = 0.25
CASE_RERANK_TEXT_LIMIT = 3600
DEFAULT_RERANK_MIN_INTERVAL_MS = 1200
DEFAULT_RERANK_MAX_RETRIES = 3
DEFAULT_RERANK_RANK_SAFE = True
DEFAULT_RERANK_MAX_RANK_PROMOTION = 20
_LAST_RERANK_REQUEST_AT = 0.0

DEFAULT_SOURCE_FIELDS = [
    "chunk_id",
    "doc_id",
    "case_name",
    "reason",
    "trial_level",
    "court_name",
    "judge_date",
    "section_type",
    "section_title",
    "chunk_text",
    "embedding_text",
    "char_start",
    "char_end",
    "line_start",
    "line_end",
    "prev_chunk_id",
    "next_chunk_id",
    "chunk_index_in_case",
    "chunk_index_in_section",
    "statutes",
    "section_weight",
]

DEFAULT_CASE_FIELDS = [
    "doc_id",
    "case_name",
    "reason",
    "trial_level",
    "court_name",
    "judge_date",
    "publish_date",
    "full_text",
    "full_text_hash",
    "litigants",
    "statutes",
]


@dataclass
class ChunkHit:
    chunk_id: str
    doc_id: str
    score: float
    chunk_text: str
    section_type: str = ""
    section_title: str = ""
    case_name: str = ""
    reason: str = ""
    trial_level: str = ""
    court_name: str = ""
    judge_date: str = ""
    char_start: int | None = None
    char_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    statutes: list[str] = field(default_factory=list)
    section_weight: float | None = None
    negative_tags: list[str] = field(default_factory=list)
    outcome_tags: list[str] = field(default_factory=list)
    match_sources: list[str] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "chunk_text": self.chunk_text,
            "section_type": self.section_type,
            "section_title": self.section_title,
            "case_name": self.case_name,
            "reason": self.reason,
            "trial_level": self.trial_level,
            "court_name": self.court_name,
            "judge_date": self.judge_date,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "statutes": self.statutes,
            "section_weight": self.section_weight,
            "negative_tags": self.negative_tags,
            "outcome_tags": self.outcome_tags,
            "match_sources": self.match_sources,
            "raw_scores": self.raw_scores,
        }


class OpenSearchClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        verify_ssl: bool = False,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        auth = f"{username}:{password}".encode("utf-8")
        self.auth_header = "Basic " + base64.b64encode(auth).decode("ascii")
        self.context = ssl.create_default_context()
        if not verify_ssl:
            self.context = ssl._create_unverified_context()

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": self.auth_header,
            "Content-Type": "application/json",
        }
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
                context=self.context,
            ) as response:
                content = response.read().decode("utf-8")
                return json.loads(content) if content else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenSearch HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenSearch connection failed: {exc}") from exc


def build_filter_clauses(args: argparse.Namespace) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    if args.reason:
        filters.append(exact_or_phrase_filter("reason", args.reason))
    if args.trial_level:
        filters.append(exact_or_phrase_filter("trial_level", args.trial_level))
    if args.court_name:
        filters.append(exact_or_phrase_filter("court_name", args.court_name))
    if args.section_type:
        filters.append(exact_or_phrase_filter("section_type", args.section_type))

    if args.judge_date_from or args.judge_date_to:
        range_clause: dict[str, Any] = {"range": {"judge_date": {}}}
        if args.judge_date_from:
            range_clause["range"]["judge_date"]["gte"] = args.judge_date_from
        if args.judge_date_to:
            range_clause["range"]["judge_date"]["lte"] = args.judge_date_to
        filters.append(range_clause)
    return filters


def exact_or_phrase_filter(field_name: str, value: str) -> dict[str, Any]:
    return {
        "bool": {
            "should": [
                {"term": {f"{field_name}.keyword": value}},
                {"term": {field_name: value}},
                {"match_phrase": {field_name: value}},
            ],
            "minimum_should_match": 1,
        }
    }


def section_type_filter(section_type: str) -> dict[str, Any]:
    return {"term": {"section_type": section_type}}


def route_filters(base_filters: list[dict[str, Any]], section_type: str = "") -> list[dict[str, Any]]:
    filters = list(base_filters)
    if section_type:
        filters.append(section_type_filter(section_type))
    return filters


def bm25_query_body(query: str, filters: list[dict[str, Any]], size: int) -> dict[str, Any]:
    return {
        "size": size,
        "_source": DEFAULT_SOURCE_FIELDS,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "chunk_text^3.0",
                                "embedding_text^2.0",
                                "section_title^1.4",
                                "reason^1.8",
                                "case_name^1.2",
                            ],
                            "type": "best_fields",
                        }
                    }
                ],
                "filter": filters,
            }
        },
    }


def vector_query_body(
    vector: list[float],
    filters: list[dict[str, Any]],
    size: int,
) -> dict[str, Any]:
    knn_body: dict[str, Any] = {"vector": vector, "k": size}
    if filters:
        knn_body["filter"] = {"bool": {"filter": filters}}

    return {
        "size": size,
        "_source": DEFAULT_SOURCE_FIELDS,
        "query": {
            "knn": {
                "embedding": knn_body,
            }
        },
    }


def source_to_chunk_hit(hit: dict[str, Any], source_name: str) -> ChunkHit:
    source = hit.get("_source", {})
    chunk_text = source.get("chunk_text", "")
    section_type = source.get("section_type", "")
    return ChunkHit(
        chunk_id=source.get("chunk_id") or hit.get("_id", ""),
        doc_id=source.get("doc_id", ""),
        score=float(hit.get("_score") or 0.0),
        chunk_text=chunk_text,
        section_type=section_type,
        section_title=source.get("section_title", ""),
        case_name=source.get("case_name", ""),
        reason=source.get("reason", ""),
        trial_level=source.get("trial_level", ""),
        court_name=source.get("court_name", ""),
        judge_date=source.get("judge_date", ""),
        char_start=safe_int(source.get("char_start")),
        char_end=safe_int(source.get("char_end")),
        line_start=safe_int(source.get("line_start")),
        line_end=safe_int(source.get("line_end")),
        statutes=normalize_list(source.get("statutes")),
        section_weight=safe_float(source.get("section_weight"), SECTION_WEIGHTS.get(section_type or "", 0.6)),
        negative_tags=extract_negative_tags(chunk_text),
        outcome_tags=extract_outcome_tags(chunk_text),
        match_sources=[source_name],
        raw_scores={source_name: float(hit.get("_score") or 0.0)},
    )


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_list(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def request_query_embedding(
    query: str,
    api_key: str,
    model: str,
    endpoint: str,
    timeout: int = 60,
) -> list[float]:
    payload = {
        "model": model,
        "input": query,
        "encoding_format": "float",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
            embedding = data["data"][0]["embedding"]
            if not isinstance(embedding, list):
                raise RuntimeError("Embedding response missing vector payload.")
            return embedding
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embedding HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Embedding request failed: {exc}") from exc


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
        guardrail = rerank_guardrail_adjustment(query, item)
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


def search_bm25(
    client: OpenSearchClient,
    index_name: str,
    query: str,
    filters: list[dict[str, Any]],
    size: int,
) -> list[ChunkHit]:
    body = bm25_query_body(query, filters, size)
    response = client.request("POST", f"/{urllib.parse.quote(index_name)}/_search", body)
    return [source_to_chunk_hit(hit, "bm25") for hit in response.get("hits", {}).get("hits", [])]


def search_vector(
    client: OpenSearchClient,
    index_name: str,
    query: str,
    filters: list[dict[str, Any]],
    size: int,
    api_key: str,
    model: str,
    endpoint: str,
) -> list[ChunkHit]:
    vector = request_query_embedding(query, api_key=api_key, model=model, endpoint=endpoint)
    body = vector_query_body(vector, filters, size)
    response = client.request("POST", f"/{urllib.parse.quote(index_name)}/_search", body)
    return [source_to_chunk_hit(hit, "vector") for hit in response.get("hits", {}).get("hits", [])]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[ChunkHit]],
    weights: dict[str, float] | None = None,
    k: int = 60,
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
        top_key_hits = [hit for hit in top_hits if hit.section_type in {"fine_issue", "focus"}]
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
    k: int = 60,
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
        fine_focus_bonus = min(0.10, 0.05 * len(hit_sections & {"fine_issue", "focus"}))
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
        top_key_bonus = min(0.08, 0.04 * len(top_sections & {"fine_issue", "focus"}))
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


def fetch_case_docs(
    client: OpenSearchClient,
    index_name: str,
    doc_ids: list[str],
) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for doc_id in doc_ids:
        path = (
            f"/{urllib.parse.quote(index_name)}/_doc/"
            f"{urllib.parse.quote(doc_id, safe='')}"
        )
        try:
            response = client.request("GET", path)
        except RuntimeError:
            missing.append(doc_id)
            continue
        if response.get("found"):
            source = response.get("_source", {})
            docs[doc_id] = source
        else:
            missing.append(doc_id)

    if not missing:
        return docs

    search_body = {
        "size": len(missing),
        "_source": DEFAULT_CASE_FIELDS,
        "query": {
            "bool": {
                "should": [
                    {"ids": {"values": missing}},
                    {"terms": {"doc_id.keyword": missing}},
                    {"terms": {"doc_id": missing}},
                ],
                "minimum_should_match": 1,
            }
        },
    }
    response = client.request(
        "POST",
        f"/{urllib.parse.quote(index_name)}/_search",
        search_body,
    )
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        doc_id = source.get("doc_id") or hit.get("_id")
        if doc_id:
            docs[doc_id] = source
    return docs


def build_context(full_text: str, char_start: int | None, char_end: int | None, window: int) -> str:
    if not full_text:
        return ""
    start = 0 if char_start is None else max(0, char_start - window)
    end = len(full_text) if char_end is None else min(len(full_text), char_end + window)
    prefix = full_text[start : char_start or start]
    hit = full_text[char_start or start : char_end or end]
    suffix = full_text[char_end or end : end]
    if not hit:
        hit = full_text[start:end]
        prefix = ""
        suffix = ""
    return compact_text(prefix) + "【命中】" + compact_text(hit) + "【/命中】" + compact_text(suffix)


def compact_text(text: str, limit: int | None = None) -> str:
    cleaned = " ".join(text.replace("\u3000", " ").split())
    if limit is not None and len(cleaned) > limit:
        return cleaned[: limit - 1] + "…"
    return cleaned


def build_result_entry(
    case_hit: dict[str, Any],
    case_doc: dict[str, Any],
    show_context: bool,
    context_window: int,
    include_full_text: bool = False,
) -> dict[str, Any]:
    full_text = case_doc.get("full_text", "")
    case_doc_payload = {
        "doc_id": case_doc.get("doc_id") or case_hit.get("doc_id") or "",
        "case_name": case_doc.get("case_name") or case_hit.get("case_name") or "",
        "reason": case_doc.get("reason") or case_hit.get("reason") or "",
        "trial_level": case_doc.get("trial_level") or case_hit.get("trial_level") or "",
        "court_name": case_doc.get("court_name") or case_hit.get("court_name") or "",
        "judge_date": case_doc.get("judge_date") or case_hit.get("judge_date") or "",
        "publish_date": case_doc.get("publish_date") or "",
        "litigants": case_doc.get("litigants") or [],
        "statutes": case_doc.get("statutes") or [],
        "full_text_hash": case_doc.get("full_text_hash") or "",
    }
    if include_full_text:
        case_doc_payload["full_text"] = full_text

    matched_chunks: list[dict[str, Any]] = []
    for chunk in case_hit["matched_chunks"]:
        chunk_payload = chunk.to_dict()
        if show_context:
            chunk_payload["context_text"] = build_context(
                full_text,
                chunk.char_start,
                chunk.char_end,
                context_window,
            )
        else:
            chunk_payload["context_text"] = ""
        matched_chunks.append(chunk_payload)

    return {
        **{
            key: value
            for key, value in case_hit.items()
            if key not in {"matched_chunks", "_rerank_chunks"}
        },
        "matched_chunks": matched_chunks,
        "case_doc": case_doc_payload,
    }


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
    routes = build_query_routes(profile) if query_profile_enabled else []
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
        attach_case_key_chunks(
            client=client,
            index_name=args.chunk_index,
            case_hits=case_hits,
            top_n=max(args.top_k, args.rerank_top_n),
        )
        case_hits = rerank_case_hits(
            query=build_rerank_query(profile) if query_profile_enabled else args.query,
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
    parser.set_defaults(query_profile=True)
    parser.set_defaults(query_profile_boost=True)
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
