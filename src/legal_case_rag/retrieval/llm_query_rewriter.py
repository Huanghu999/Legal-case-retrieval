from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
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
    focus_tags_query: str = ""
    fine_tags_query: str = ""
    fine_rule_query: str = ""
    focus_analysis_query: str = ""
    used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "expanded_query": self.expanded_query,
            "focus_tags_query": self.focus_tags_query,
            "fine_tags_query": self.fine_tags_query,
            "fine_rule_query": self.fine_rule_query,
            "focus_analysis_query": self.focus_analysis_query,
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


def load_rewrite_cache(path: str) -> dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_rewrite_cache(cache: dict[str, Any], path: str) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.write("\n")


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
        focus_tags_query=clean_field(payload.get("focus_tags_query"), max_chars=80),
        fine_tags_query=clean_field(payload.get("fine_tags_query"), max_chars=80),
        fine_rule_query=clean_field(payload.get("fine_rule_query"), max_chars=150),
        focus_analysis_query=clean_field(payload.get("focus_analysis_query"), max_chars=150),
    )
    has_signal = any(
        [
            rewrite.expanded_query,
            rewrite.focus_tags_query,
            rewrite.fine_tags_query,
            rewrite.fine_rule_query,
            rewrite.focus_analysis_query,
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
                "你是民商事判决书的要素抽取器，只输出 JSON，不要解释。"
                "输入是用户关于买卖合同纠纷的检索问题。严格依据用户问题抽取，信息不足填空字符串，不得编造。\n\n"
                "输出 JSON 必须且只能含这些键:\n"
                '{"expanded_query":"", "focus_tags_query":"", "fine_tags_query":"", "fine_rule_query":"", "focus_analysis_query":""}\n\n'
                "字段规则:\n"
                "1) expanded_query: 综合检索 query，空格分隔的检索短语，不要写完整句子，不超过60字。\n\n"
                "2) focus_tags_query: 对齐「争议焦点.焦点标签」，从下列受控词表选取(可多选，空格分隔)，只能用 code 原文，不得自创；选不出填空字符串。\n"
                "   词表: 货款给付 逾期付款利息 逾期付款违约金 质量异议 数量短缺 交付争议 合同解除 定金罚则 所有权保留 合同效力 合同成立与否 退货退款 价款约定不明 抵销抗辩 货物风险负担 第三人履行/代收货款 消费者退赔 诉讼时效 管辖程序\n\n"
                "3) fine_tags_query: 对齐「细争点.主叶子」+「细争点.细争点」，从下列受控词表选取(可多选，空格分隔)，只能用 code 原文；选不出填空字符串。\n"
                "   词表: A1_口头或事实买卖合同成立认定 A2_电子方式下单的成立与内容 A3_缔约主体与表见代理 A4_格式条款效力 A5_合同变更与补充协议效力 A6_合同无效 A7_预约框架合同与本约区分 "
                "B1_交付是否完成与风险转移 B2_数量短缺与过磅争议 B3_货款总额与结算依据 B4_对账单欠条证明力 B5_付款归属与冲抵 B6_已付或重复付款抗辩 B7_闭口合同与按实结算 B8_付款条件成就与拟制 "
                "C1_质量异议合理期间与默示认可 C2_质量瑕疵程度与拒付减价 C3_质量鉴定启动与举证责任 C4_先履行同时履行抗辩与票款顺序 C5_抵销抗辩 C6_诉讼时效抗辩 C7_不安抗辩 "
                "G1_根本违约与法定解除权 G2_约定解除条件成就 G3_解除时间与解除后返还 G4_违约损害赔偿范围 G5_不可抗力情势变更免责 "
                "D1_一人公司股东财产混同连带 D2_股东出资不实抽逃补充赔偿 D3_公司人格否认混同 D4_保证担保责任 D5_挂靠经营责任主体 D6_实际买受人与名义主体 D7_第三人代收代付货款 D8_债务加入与保证区分 "
                "E1_逾期利息起算点 E2_利息标准与上限 E3_违约金过高司法调整 E4_违约金利息定金并用 E5_价款约定不明填补 E6_定金罚则适用 "
                "F1_消费者食品药品惩罚赔偿 F2_知假买假职业打假资格 F3_网络购物平台责任 F4_所有权保留取回权 F5_分期付款买卖 F6_试用买卖凭样品买卖 F7_名为买卖实为借贷融资 F8_拍卖等特殊缔约 F9_消费欺诈退一赔三\n\n"
                "4) fine_rule_query: 对齐「细争点.裁判规则争点」，写一句裁判规则级争点(法律争点描述)，不超过80字。\n\n"
                "5) focus_analysis_query: 对齐「争议焦点.焦点评析」，把案情核心+法律争点+裁判要旨合并为一段检索文本，不超过120字。\n"
                "   - 案情核心: 1~2句，与争议直接相关的关键事实\n"
                "   - 法律争点: 1句，本案需法院裁断的具体法律问题\n"
                "   - 裁判要旨: 1~2句，法院的分析思路与裁判规则\n\n"
                "不要回答法律问题，不判断胜败，不生成结论。不要虚构案号、法院、当事人、金额、日期。只从用户 query 抽取事实。\n\n"
                "示例1：用户 query=没有书面合同，但有微信对账和发票抵扣，能否认定买卖合同成立？\n"
                '输出={"expanded_query":"无书面合同 微信对账 增值税专用发票认证抵扣 送货交付 事实买卖合同成立",'
                '"focus_tags_query":"合同成立与否 货款给付",'
                '"fine_tags_query":"A1_口头或事实买卖合同成立认定 B3_货款总额与结算依据",'
                '"fine_rule_query":"无书面合同时以送货单、对账、发票抵扣等履行行为推定买卖关系成立",'
                '"focus_analysis_query":"买方认证抵扣发票且双方微信对账，无书面合同送货单主体不明。法律争点：能否认定事实买卖合同成立并确定货款。裁判要旨：以实际履行行为推定合同成立。"}\n\n'
                "示例2：用户 query=设备经鉴定存在质量问题，合同约定尾款支付以调试合格为条件，卖方能否要求支付已到期的第二笔款？\n"
                '输出={"expanded_query":"设备质量鉴定 质量问题 尾款支付条件 调试合格 第二笔款已到期 付款条件成就",'
                '"focus_tags_query":"付款条件成就 质量争议",'
                '"fine_tags_query":"B8_付款条件成就与拟制 C2_质量瑕疵程度与拒付减价",'
                '"fine_rule_query":"合同约定付款条件成就时买方应按期付款，质量瑕疵不影响已到期无前置条件款项的支付",'
                '"focus_analysis_query":"设备经鉴定存在质量问题，合同约定尾款以调试合格为条件，第二笔款已到期且无前置条件。法律争点：质量瑕疵能否对抗已到期款项支付。裁判要旨：已到期无前置条件的款项不因质量争议而免除。"}\n'
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
    cache_path: str | None = None,
) -> LlmQueryRewrite:
    if not enabled:
        return LlmQueryRewrite(fallback_reason="disabled")

    cache: dict[str, Any] | None = None
    if cache_path:
        cache = load_rewrite_cache(cache_path)
        if query in cache:
            cached = cache[query]
            # 旧格式缓存（含 legal_issue 等旧字段）跳过，强制重新提取
            if isinstance(cached, dict) and cached.get("used") and "focus_tags_query" in cached:
                return LlmQueryRewrite(
                    expanded_query=cached.get("expanded_query", ""),
                    focus_tags_query=cached.get("focus_tags_query", ""),
                    fine_tags_query=cached.get("fine_tags_query", ""),
                    fine_rule_query=cached.get("fine_rule_query", ""),
                    focus_analysis_query=cached.get("focus_analysis_query", ""),
                    used=True,
                )

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
        rewrite = parse_rewrite_response(extract_message_text(message))

        if cache_path and cache is not None and rewrite.used:
            cache[query] = rewrite.to_dict()
            save_rewrite_cache(cache, cache_path)

        return rewrite
    except Exception as exc:
        return LlmQueryRewrite(fallback_reason=f"llm_error:{type(exc).__name__}")
