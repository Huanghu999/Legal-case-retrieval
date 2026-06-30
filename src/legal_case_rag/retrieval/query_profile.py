from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .llm_query_rewriter import LlmQueryRewrite


NEGATIVE_TERMS = [
    "未",
    "没有",
    "无",
    "不能证明",
    "未举证",
    "不予支持",
    "不支持",
    "缺乏依据",
    "未约定",
    "未履行",
    "未完成",
    "未支付",
    "未返还",
    "未交付",
    "未办理",
    "未到庭",
    "未答辩",
    "不成立",
    "不存在",
    "无法证明",
    "证据不足",
    "否认收到",
    "主体不明",
    "未能证明",
    "拒绝支付",
    "未实际发货",
    "无真实交易合意",
    "预测订单",
    "刷单",
]

REQUEST_TERMS = [
    "返还",
    "解除",
    "赔偿",
    "支付利息",
    "资金占用利息",
    "抵扣",
    "确认合同无效",
    "继续履行",
    "违约金",
    "定金",
    "租金",
    "押金",
    "赔偿损失",
    "承担责任",
    "驳回",
    "货款",
    "对账",
    "欠款",
    "发票",
    "增值税发票",
    "送货单",
    "交付",
    "收货",
    "质量问题",
    "举证责任",
    "表见代理",
    "合同相对方",
    "逾期利息",
    "违约金调整",
    "股东混同",
]

DEFENSE_TERMS = [
    "不同意",
    "辩称",
    "抗辩",
    "主张已履行",
    "认为不存在",
    "已经支付",
    "已经返还",
    "合同无效",
    "超过诉讼时效",
]

DISPUTE_TERMS = [
    "争议",
    "焦点",
    "是否",
    "能否",
    "可否",
    "应否",
    "如何认定",
    "能不能",
    "支不支持",
    "支持",
    "不支持",
    "承担",
    "成立",
    "认定",
    "证明",
    "举证",
    "相对方",
    "表见代理",
    "质量",
    "抵扣",
    "对账",
    "短缺",
    "混同",
]

LEGAL_RELATION_TERMS = [
    "委托合同",
    "买卖合同",
    "房屋租赁合同",
    "租赁合同",
    "借款合同",
    "服务合同",
    "劳动合同",
    "建设工程",
    "物业服务",
    "侵权责任",
    "合同解除",
    "违约责任",
    "不当得利",
    "民间借贷",
    "保证责任",
    "定金",
    "押金",
    "资金占用",
    "事实买卖合同",
    "口头买卖合同",
    "买卖合同关系",
    "承揽合同",
    "表见代理",
    "合同相对方",
]

REASON_TERMS = [
    "委托合同纠纷",
    "买卖合同纠纷",
    "房屋租赁合同纠纷",
    "租赁合同纠纷",
    "金融借款合同纠纷",
    "民间借贷纠纷",
    "服务合同纠纷",
    "物业服务合同纠纷",
    "建设工程施工合同纠纷",
    "劳动合同纠纷",
    "不当得利纠纷",
    "侵权责任纠纷",
]

SUPPORT_TERMS = ["予以支持", "应予支持", "支持", "准许", "成立"]
REJECT_TERMS = ["不予支持", "驳回", "不支持", "不成立", "缺乏依据"]


@dataclass
class QueryRoute:
    name: str
    query: str
    route_type: str
    weight: float
    section_type: str = ""


@dataclass
class QueryProfile:
    raw_query: str
    core_reasons: list[str] = field(default_factory=list)
    request_types: list[str] = field(default_factory=list)
    dispute_focus: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    negative_facts: list[str] = field(default_factory=list)
    legal_relations: list[str] = field(default_factory=list)
    statutes: list[str] = field(default_factory=list)
    expected_tendency: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "core_reasons": self.core_reasons,
            "request_types": self.request_types,
            "dispute_focus": self.dispute_focus,
            "key_facts": self.key_facts,
            "negative_facts": self.negative_facts,
            "legal_relations": self.legal_relations,
            "statutes": self.statutes,
            "expected_tendency": self.expected_tendency,
            "routes": [route.__dict__ for route in build_query_routes(self)],
        }


def unique_keep_order(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = compact_text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
        if limit is not None and len(output) >= limit:
            break
    return output


def compact_text(text: str) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").split()).strip()


def split_clauses(text: str) -> list[str]:
    cleaned = compact_text(text)
    parts = re.split(r"[。！？!?；;\n]+", cleaned)
    return unique_keep_order([part.strip(" ，,、") for part in parts if part.strip()], limit=16)


def find_terms(text: str, terms: list[str], limit: int | None = None) -> list[str]:
    return unique_keep_order([term for term in terms if term in text], limit=limit)


def extract_statutes(text: str) -> list[str]:
    statutes = re.findall(r"《[^》]{2,40}》第[一二三四五六七八九十百千万零〇\d]+条(?:第[一二三四五六七八九十百千万零〇\d]+款)?", text)
    law_names = re.findall(r"《[^》]{2,40}》", text)
    return unique_keep_order(statutes + law_names, limit=8)


def infer_expected_tendency(text: str) -> str:
    support = any(term in text for term in SUPPORT_TERMS)
    reject = any(term in text for term in REJECT_TERMS)
    if support and reject:
        return "部分支持或需区分请求"
    if support:
        return "支持请求"
    if reject:
        return "不支持请求"
    return ""


def build_query_profile(query: str) -> QueryProfile:
    raw_query = compact_text(query)
    clauses = split_clauses(raw_query)
    negative_facts = [
        clause
        for clause in clauses
        if any(term in clause for term in NEGATIVE_TERMS)
    ]
    dispute_focus = [
        clause
        for clause in clauses
        if any(term in clause for term in DISPUTE_TERMS)
    ]
    key_facts = [
        clause
        for clause in clauses
        if clause not in negative_facts and any(term in clause for term in REQUEST_TERMS + LEGAL_RELATION_TERMS)
    ]

    return QueryProfile(
        raw_query=raw_query,
        core_reasons=find_terms(raw_query, REASON_TERMS, limit=4),
        request_types=find_terms(raw_query, REQUEST_TERMS, limit=8),
        dispute_focus=unique_keep_order(dispute_focus, limit=5),
        key_facts=unique_keep_order(key_facts, limit=6),
        negative_facts=unique_keep_order(negative_facts, limit=6),
        legal_relations=find_terms(raw_query, LEGAL_RELATION_TERMS, limit=8),
        statutes=extract_statutes(raw_query),
        expected_tendency=infer_expected_tendency(raw_query),
    )


def join_query_parts(parts: list[str], fallback: str) -> str:
    query = compact_text(" ".join(part for part in parts if compact_text(part)))
    return query or fallback


def build_focus_query(profile: QueryProfile) -> str:
    return join_query_parts(
        profile.dispute_focus
        + profile.request_types
        + profile.key_facts
        + profile.legal_relations
        + profile.core_reasons,
        profile.raw_query,
    )


def build_negative_query(profile: QueryProfile) -> str:
    return join_query_parts(
        profile.negative_facts
        + profile.request_types
        + profile.legal_relations
        + profile.core_reasons,
        "",
    )


def build_legal_query(profile: QueryProfile) -> str:
    return join_query_parts(
        profile.core_reasons
        + profile.legal_relations
        + profile.statutes
        + profile.request_types,
        "",
    )


def build_rewrite_legal_query(profile: QueryProfile, rewrite: LlmQueryRewrite | None) -> str:
    if not rewrite or not rewrite.used:
        return ""
    return join_query_parts(
        [
            rewrite.legal_issue,
            rewrite.main_leaf,
            " ".join(rewrite.focus_labels),
        ],
        "",
    )


def build_rewrite_statute_query(profile: QueryProfile, rewrite: LlmQueryRewrite | None) -> str:
    if not rewrite or not rewrite.used:
        return ""
    return join_query_parts(
        [
            rewrite.legal_issue,
            rewrite.statutes,
            rewrite.main_leaf,
        ],
        "",
    )


def build_rerank_query(profile: QueryProfile, rewrite: LlmQueryRewrite | None = None) -> str:
    rewrite_parts: list[str] = []
    if rewrite and rewrite.used:
        rewrite_parts = [
            label + value
            for label, value in [
                ("LLM核心争点：", rewrite.legal_issue),
                ("LLM事实要素：", rewrite.fact_elements),
                ("LLM法条：", rewrite.statutes),
            ]
            if value
        ]
    return join_query_parts(
        [
            profile.raw_query,
            "规则争议焦点：" + "；".join(profile.dispute_focus),
            "规则请求类型：" + "、".join(profile.request_types),
            "规则否定事实：" + "；".join(profile.negative_facts),
            "规则法律关系：" + "、".join(profile.legal_relations + profile.core_reasons),
            *rewrite_parts,
        ],
        profile.raw_query,
    )


def build_query_routes(profile: QueryProfile, rewrite: LlmQueryRewrite | None = None) -> list[QueryRoute]:
    routes = [
        QueryRoute("bm25_raw", profile.raw_query, "bm25", 1.0),
        QueryRoute("vector_raw", profile.raw_query, "vector", 0.8),
    ]
    focus_query = build_focus_query(profile)
    expanded_query = rewrite.expanded_query if rewrite and rewrite.used and rewrite.expanded_query else focus_query
    if expanded_query and expanded_query != profile.raw_query:
        routes.append(QueryRoute("bm25_focus", expanded_query, "bm25", 0.95))
        routes.append(QueryRoute("vector_focus", expanded_query, "vector", 1.20))
    legal_section_query = build_rewrite_legal_query(profile, rewrite)
    section_query = legal_section_query or focus_query or profile.raw_query
    if section_query:
        routes.extend(
            [
                QueryRoute("bm25_fine_tags", section_query, "bm25", 1.20, "fine_tags"),
                QueryRoute("bm25_fine_rule", section_query, "bm25", 1.30, "fine_rule"),
                QueryRoute("bm25_focus_tags", section_query, "bm25", 1.60, "focus_tags"),
                QueryRoute("bm25_focus_analysis", section_query, "bm25", 1.10, "focus_analysis"),
                QueryRoute("bm25_reasoning", section_query, "bm25", 1.10, "reasoning"),
                QueryRoute(
                    "bm25_facts",
                    rewrite.fact_elements if rewrite and rewrite.used and rewrite.fact_elements else section_query,
                    "bm25",
                    0.60,
                    "facts",
                ),
            ]
        )
    negative_query = build_negative_query(profile)
    if negative_query:
        routes.append(QueryRoute("bm25_negative", negative_query, "bm25", 1.20))
    legal_query = build_rewrite_statute_query(profile, rewrite) or build_legal_query(profile)
    if legal_query:
        routes.append(QueryRoute("bm25_legal", legal_query, "bm25", 0.80))
    return routes


def extract_negative_tags(text: str) -> list[str]:
    return find_terms(text, NEGATIVE_TERMS, limit=8)


def extract_outcome_tags(text: str) -> list[str]:
    tags: list[str] = []
    if any(term in text for term in SUPPORT_TERMS):
        tags.append("支持")
    if any(term in text for term in REJECT_TERMS):
        tags.append("不支持或驳回")
    if "部分支持" in text or ("支持" in text and "驳回" in text):
        tags.append("部分支持")
    return unique_keep_order(tags, limit=4)


def profile_match_bonus(
    profile: QueryProfile,
    *,
    chunk_text: str,
    reason: str = "",
    section_type: str = "",
    statutes: list[str] | None = None,
) -> float:
    text = compact_text(chunk_text)
    bonus = 0.0

    if section_type in {"fine_tags", "focus_tags"}:
        bonus += 0.03
    elif section_type in {"fine_rule", "focus_analysis"}:
        bonus += 0.02
    elif section_type in {"reasoning", "facts"}:
        bonus += 0.04
    elif section_type in {"defense", "judgment"}:
        bonus += 0.03
    elif section_type in {"header", "statutes"}:
        bonus -= 0.03

    if profile.core_reasons and any(item == reason or item in text for item in profile.core_reasons):
        bonus += 0.08
    if profile.legal_relations and any(item in text for item in profile.legal_relations):
        bonus += 0.05
    if profile.request_types and any(item in text for item in profile.request_types):
        bonus += 0.04
    if profile.negative_facts:
        negative_tags = extract_negative_tags(text)
        if negative_tags:
            bonus += 0.12
        if any(fact and fact in text for fact in profile.negative_facts):
            bonus += 0.06
    if profile.statutes:
        statute_text = " ".join(statutes or [])
        if any(item in text or item in statute_text for item in profile.statutes):
            bonus += 0.05

    return max(-0.05, min(0.28, bonus))
