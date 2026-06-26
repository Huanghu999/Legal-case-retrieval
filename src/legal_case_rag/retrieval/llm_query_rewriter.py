from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-pro"
MIMO_API_KEY_ENV = "MIMO_API_KEY"
MAX_FIELD_CHARS = 120


@dataclass
class LlmQueryRewrite:
    expanded_query: str = ""
    legal_issue: str = ""
    fact_elements: str = ""
    statutes: str = ""
    main_leaf: str = ""
    focus_labels: list[str] = field(default_factory=list)
    used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "expanded_query": self.expanded_query,
            "legal_issue": self.legal_issue,
            "fact_elements": self.fact_elements,
            "statutes": self.statutes,
            "main_leaf": self.main_leaf,
            "focus_labels": list(self.focus_labels),
            "used": self.used,
            "fallback_reason": self.fallback_reason,
        }


def compact_text(text: str) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").split()).strip()


def clean_field(value: Any, max_chars: int = MAX_FIELD_CHARS) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = compact_text(value)
    if not cleaned or len(cleaned) > max_chars:
        return ""
    return cleaned


def clean_focus_labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for item in value:
        label = clean_field(item, max_chars=40)
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= 4:
            break
    return labels


def parse_rewrite_response(content: Any) -> LlmQueryRewrite:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return LlmQueryRewrite(fallback_reason="invalid_json")
    else:
        payload = content

    if not isinstance(payload, dict):
        return LlmQueryRewrite(fallback_reason="invalid_json")

    rewrite = LlmQueryRewrite(
        expanded_query=clean_field(payload.get("expanded_query")),
        legal_issue=clean_field(payload.get("legal_issue")),
        fact_elements=clean_field(payload.get("fact_elements")),
        statutes=clean_field(payload.get("statutes")),
        main_leaf=clean_field(payload.get("main_leaf")),
        focus_labels=clean_focus_labels(payload.get("focus_labels")),
    )
    has_signal = any(
        [
            rewrite.expanded_query,
            rewrite.legal_issue,
            rewrite.fact_elements,
            rewrite.statutes,
            rewrite.main_leaf,
            rewrite.focus_labels,
        ]
    )
    if not has_signal:
        rewrite.fallback_reason = "empty_fields"
        return rewrite

    rewrite.used = True
    return rewrite


def build_rewrite_messages(query: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是法律类案检索 query 改写器，只输出 JSON，不要解释。"
                "任务是把用户 query 提取成与 benchmark/corpus 字段对齐的检索要素。"
                "字段对齐：expanded_query=综合检索 query；"
                "legal_issue=corpus.争议焦点.焦点评析.法律争点 或 corpus.细争点.裁判规则争点；"
                "fact_elements=corpus.争议焦点.焦点评析.案情核心 或 corpus.段落.查明事实；"
                "statutes=corpus.引用法条，只填 query 明确出现的法条；"
                "main_leaf=queries.主叶子 或 corpus.细争点.主叶子，不确定则空；"
                "focus_labels=corpus.争议焦点.焦点标签。"
                "字段边界：fact_elements 只写客观事实，如付款、交付、发票、对账、质量、通知；"
                "legal_issue 只写法律争点，如合同是否成立、主体是否适格、责任是否承担、金额如何认定；"
                "expanded_query 必须是空格分隔的检索短语，不要写完整句子。"
                "不要回答法律问题，不判断胜败，不生成结论。"
                "不要虚构案号、法院、当事人、金额、日期。"
                "只从用户 query 抽取事实，并补充通用法律检索术语。"
                "每个字符串字段不超过120个汉字；不确定字段留空，不要硬猜。"
                "示例1：用户 query=没有书面合同，但有微信对账和发票抵扣，能否认定买卖合同成立？"
                '输出={"expanded_query":"无书面合同 微信对账 增值税专用发票认证抵扣 送货交付 事实买卖合同成立 货款给付",'
                '"legal_issue":"无书面合同情况下根据微信对账、发票抵扣、送货交付等履行行为认定事实买卖合同成立",'
                '"fact_elements":"未签书面合同 微信对账 发票认证抵扣 送货交付",'
                '"statutes":"","main_leaf":"A1_口头或事实买卖合同成立认定","focus_labels":["合同成立与否","货款给付"]}。'
                "示例2：用户 query=承租人拖欠租金，出租人解除合同后押金能不能抵扣租金和违约金？"
                '输出={"expanded_query":"房屋租赁合同解除 拖欠租金 押金抵扣 违约金 租赁保证金返还",'
                '"legal_issue":"租赁合同解除后押金能否用于抵扣欠付租金、违约金及损失",'
                '"fact_elements":"承租人拖欠租金 出租人解除合同 合同约定押金 押金返还争议",'
                '"statutes":"","main_leaf":"","focus_labels":["押金返还","合同解除","违约责任"]}。'
                '输出格式：{"expanded_query":"","legal_issue":"","fact_elements":"","statutes":"","main_leaf":"","focus_labels":[]}'
            ),
        },
        {"role": "user", "content": query},
    ]


def create_mimo_client() -> Any:
    api_key = os.getenv(MIMO_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing {MIMO_API_KEY_ENV}.")
    if OpenAI is None:
        raise RuntimeError("Missing openai Python package.")
    return OpenAI(api_key=api_key, base_url=MIMO_BASE_URL, timeout=30)


def extract_message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text") or ""))
        return "".join(parts)
    return ""


def rewrite_query_with_llm(
    query: str,
    *,
    enabled: bool = True,
    client_factory: Callable[[], Any] | None = None,
) -> LlmQueryRewrite:
    if not enabled:
        return LlmQueryRewrite(fallback_reason="disabled")
    factory = client_factory or create_mimo_client
    if factory is create_mimo_client and not os.getenv(MIMO_API_KEY_ENV):
        return LlmQueryRewrite(fallback_reason="missing_api_key")
    try:
        client = factory()
        response = client.chat.completions.create(
            model=MIMO_MODEL,
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=build_rewrite_messages(query),
        )
        message = response.choices[0].message
        return parse_rewrite_response(extract_message_text(message))
    except Exception as exc:
        return LlmQueryRewrite(fallback_reason=f"llm_error:{type(exc).__name__}")
