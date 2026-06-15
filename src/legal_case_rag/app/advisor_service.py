from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from .search_args import (
    build_search_args as build_shared_search_args,
    bool_param,
    int_param,
    normalize_sequence,
    str_param,
)
from ..retrieval import search as retrieval


MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-pro"
MIMO_API_KEY_ENV = "MIMO_API_KEY"

DEFAULT_APP_CONFIG = {
    "mode": "hybrid",
    "rerank": True,
    "query_profile": True,
    "query_profile_boost": True,
    "top_k": 8,
    "chunk_top_k": 3,
    "candidate_size": 80,
    "show_context": True,
    "context_window": 180,
    "rerank_top_n": 20,
    "reason": "",
    "trial_level": "",
    "court_name": "",
    "section_type": "",
    "judge_date_from": "",
    "judge_date_to": "",
}

SECTION_LABELS = {
    "header": "首部信息",
    "process": "审理经过",
    "claims": "原告诉称",
    "defense": "被告辩称",
    "facts": "本院查明",
    "reasoning": "本院认为",
    "judgment": "裁判结果",
    "statutes": "附法律依据",
}


def compact_text(text: str, limit: int) -> str:
    cleaned = " ".join(str(text or "").replace("\u3000", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def build_search_args(payload: dict[str, Any]):
    return build_shared_search_args(
        payload,
        default_config=DEFAULT_APP_CONFIG,
        retrieval_module=retrieval,
        verify_ssl_default=False,
    )


def build_case_prompt_payload(result: dict[str, Any]) -> dict[str, Any]:
    case_doc = result.get("case_doc", {})
    matched_chunks = result.get("matched_chunks", [])
    claims_text = ""
    facts_text = ""
    reasoning_text = ""
    judgment_text = ""

    for chunk in matched_chunks:
        section_type = str(chunk.get("section_type") or "")
        section_title = str(chunk.get("section_title") or "")
        text = compact_text(chunk.get("chunk_text", ""), 420)
        if not text:
            continue
        if not claims_text and section_type == "claims":
            claims_text = text
        if not facts_text and section_type == "facts":
            facts_text = text
        if not reasoning_text and (section_type == "reasoning" or section_title == "本院认为"):
            reasoning_text = text
        if not judgment_text and (section_type == "judgment" or section_title == "裁判结果"):
            judgment_text = text

    if not reasoning_text or not judgment_text:
        for chunk in matched_chunks:
            section_title = str(chunk.get("section_title") or "")
            text = compact_text(chunk.get("context_text", "") or chunk.get("chunk_text", ""), 420)
            if not text:
                continue
            if not reasoning_text and ("本院认为" in section_title or "本院认为" in text):
                reasoning_text = text
            if not judgment_text and ("裁判结果" in section_title or "判决如下" in text or "裁定如下" in text):
                judgment_text = text

    statutes = normalize_sequence(case_doc.get("statutes"))

    return {
        "doc_id": result.get("doc_id", ""),
        "case_name": case_doc.get("case_name") or result.get("case_name", ""),
        "reason": case_doc.get("reason") or result.get("reason", ""),
        "trial_level": case_doc.get("trial_level") or result.get("trial_level", ""),
        "court_name": case_doc.get("court_name") or result.get("court_name", ""),
        "judge_date": case_doc.get("judge_date") or result.get("judge_date", ""),
        "case_score": result.get("case_score", 0),
        "claims": claims_text,
        "facts": facts_text,
        "reasoning": reasoning_text,
        "judgment": judgment_text,
        "statutes": statutes[:6],
    }


def build_case_preview(result: dict[str, Any]) -> dict[str, Any]:
    case_doc = result.get("case_doc", {})
    chunks = result.get("matched_chunks", [])
    full_text_sections = {
        item["section_key"]: item
        for item in extract_hover_sections(case_doc)
    }
    chunk_sections = extract_hover_sections_from_chunks(chunks)
    extracted_sections = {
        section.get("id"): section
        for section in extract_case_sections(case_doc)
        if section.get("id") in {"reasoning", "judgment"}
    }

    def resolve_section(section_key: str, title: str, missing_text: str) -> dict[str, Any]:
        direct = full_text_sections.get(section_key)
        if direct and str(direct.get("context_text", "")).strip():
            return {
                "section_key": section_key,
                "section_title": title,
                "chunk_text": str(direct.get("chunk_text", "") or ""),
                "context_text": str(direct.get("context_text", "") or ""),
                "char_start": None,
                "char_end": None,
                "available": True,
            }

        chunk_section = chunk_sections.get(section_key)
        if chunk_section and str(chunk_section.get("context_text", "")).strip():
            return {
                "section_key": section_key,
                "section_title": title,
                "chunk_text": str(chunk_section.get("chunk_text", "") or ""),
                "context_text": str(chunk_section.get("context_text", "") or ""),
                "char_start": chunk_section.get("char_start"),
                "char_end": chunk_section.get("char_end"),
                "available": True,
            }

        extracted = extracted_sections.get(section_key)
        if extracted and str(extracted.get("content", "")).strip():
            content = str(extracted.get("content", "") or "")
            return {
                "section_key": section_key,
                "section_title": title,
                "chunk_text": content,
                "context_text": content,
                "char_start": None,
                "char_end": None,
                "available": True,
            }

        return {
            "section_key": section_key,
            "section_title": title,
            "chunk_text": "",
            "context_text": missing_text,
            "char_start": None,
            "char_end": None,
            "available": False,
        }

    preview_sections = [
        resolve_section("reasoning", "本院认为", "该文书未识别到“本院认为”部分。"),
        resolve_section("judgment", "裁判结果", "该文书未识别到“裁判结果”部分。"),
    ]

    return {
        "doc_id": result.get("doc_id", ""),
        "case_name": case_doc.get("case_name") or result.get("case_name", ""),
        "reason": case_doc.get("reason") or result.get("reason", ""),
        "trial_level": case_doc.get("trial_level") or result.get("trial_level", ""),
        "court_name": case_doc.get("court_name") or result.get("court_name", ""),
        "judge_date": case_doc.get("judge_date") or result.get("judge_date", ""),
        "case_score": result.get("case_score", 0),
        "hit_count": result.get("hit_count", 0),
        "litigants": normalize_sequence(case_doc.get("litigants")),
        "statutes": normalize_sequence(case_doc.get("statutes")),
        "preview_sections": preview_sections,
    }


def slice_text_by_anchor(
    full_text: str,
    start_needles: list[str],
    end_needles: list[str],
) -> str:
    start = find_anchor(full_text, start_needles)
    if start is None:
        return ""

    end_candidates = []
    for needle in end_needles:
        index = full_text.find(needle, start + 1)
        if index >= 0:
            end_candidates.append(index)
    end = min(end_candidates) if end_candidates else len(full_text)
    return full_text[start:end].strip()


def extract_hover_sections(case_doc: dict[str, Any]) -> list[dict[str, str]]:
    full_text = str(case_doc.get("full_text") or "")
    if not full_text:
        return []

    reasoning_text = slice_text_by_anchor(
        full_text,
        ["本院认为"],
        ["判决如下", "裁定如下", "附：相关法律条文", "附法律依据"],
    )
    judgment_text = slice_text_by_anchor(
        full_text,
        ["判决如下", "裁定如下"],
        ["如不服本判决", "如不服本裁定", "附：相关法律条文", "附法律依据"],
    )

    sections = []
    if reasoning_text:
        sections.append(
            {
                "section_key": "reasoning",
                "section_title": "本院认为",
                "chunk_text": reasoning_text,
                "context_text": reasoning_text,
            }
        )
    if judgment_text:
        sections.append(
            {
                "section_key": "judgment",
                "section_title": "裁判结果",
                "chunk_text": judgment_text,
                "context_text": judgment_text,
            }
        )
    return sections


def extract_hover_sections_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    section_map: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        section_type = str(chunk.get("section_type") or "")
        section_title = str(chunk.get("section_title") or "")
        if "reasoning" not in section_map and (
            section_type == "reasoning" or "本院认为" in section_title
        ):
            section_map["reasoning"] = {
                "chunk_text": str(chunk.get("chunk_text", "") or ""),
                "context_text": str(chunk.get("context_text", "") or chunk.get("chunk_text", "") or ""),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
            }
        if "judgment" not in section_map and (
            section_type == "judgment" or "裁判结果" in section_title or "判决如下" in str(chunk.get("chunk_text", "") or "")
        ):
            section_map["judgment"] = {
                "chunk_text": str(chunk.get("chunk_text", "") or ""),
                "context_text": str(chunk.get("context_text", "") or chunk.get("chunk_text", "") or ""),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
            }
        if "reasoning" in section_map and "judgment" in section_map:
            break
    return section_map


def build_prompt(query: str, results: list[dict[str, Any]]) -> str:
    cases = [build_case_prompt_payload(item) for item in results[:8]]
    case_json = json.dumps(cases, ensure_ascii=False, indent=2)
    return f"""
你是法律类案分析助手。你的任务是：仅基于给定的检索案件，为用户问题生成“可直接展示给用户的观点式类案分析”。

严格要求：
1. 只能引用下面给出的案件，不允许编造任何新案件。
2. 每个观点都必须给出 2 到 4 个支撑案件；只有在确实没有更多合适案例时，才允许只引用 1 个。
3. 支撑案件必须使用提供的原始 doc_id，不能改写。
4. 观点必须围绕“裁判倾向、适用前提、事实区分、合同约定的影响、裁判边界”展开，不要写成泛泛的法律科普。
5. 优先从“本院认为”和“裁判结果”中提炼规则，再结合事实模式组织观点。
6. 一个观点应当清晰表达“什么情况下通常支持 / 不支持 / 需看合同约定 / 需看履行情况”。
7. 观点要尽量有区分度，不要只是把同一句话重复换个说法。
8. 如果证据不足，要明确说“现有召回案例不足以支持更确定结论”。
9. 禁止输出“法院一般会”“可能会”这种空泛句式，除非后面紧跟适用条件和案例支撑。
10. 输出必须是纯 JSON，不要加 markdown 代码块。
11. 如果多个案例实际上表达的是同一裁判规则，应当合并成一个更有概括力的观点，而不是拆成多个重复观点。
12. 如果存在相反处理路径，应当把“支持”和“不支持”区分成不同观点，说明分界标准。

写作要求：
- 最好输出 2 到 4 个观点。
- 每个观点标题要像真实裁判规则总结，例如：
  - “观点一：承租人拖欠租金且合同未排除抵扣时，押金通常可用于抵扣欠租”
  - “观点二：合同对押金用途有明确限制时，应优先按约处理”
- `analysis` 不是摘要复述，而是要说明：
  1. 该观点的适用前提
  2. 法院为何这样处理
  3. 与相邻观点的区分边界
- `supporting_cases.reason` 要写成该案支持本观点的具体理由，不要只写“支持该观点”。
- `analysis` 尽量使用“当……时，法院通常……；但如果……，则……”这种有条件区分的表达。
- 回答面向真实用户阅读，优先给出能直接指导理解裁判倾向的结论，不要堆砌术语。

输出 JSON 结构如下：
{{
  "title": "类案分析",
  "summary": "用 2-3 句概括整体裁判倾向，并点明最关键的分歧标准",
  "viewpoints": [
    {{
      "title": "观点一：......",
      "analysis": "展开说明这一观点，要求有明确适用条件和裁判理由",
      "supporting_cases": [
        {{
          "doc_id": "案件号",
          "reason": "该案支持这一观点的具体依据，1 句"
        }}
      ]
    }}
  ],
  "notice": "必要时给出风险提示或适用边界"
}}

用户问题：
{query}

可引用案件：
{case_json}
""".strip()


def build_stream_preview_prompt(query: str, results: list[dict[str, Any]]) -> str:
    cases = [build_case_prompt_payload(item) for item in results[:6]]
    case_json = json.dumps(cases, ensure_ascii=False, indent=2)
    return f"""
你是法律类案分析助手。现在不要输出 JSON，也不要输出代码块。

请直接开始写“类案分析草稿”正文，不要复述任务，不要解释你将如何回答，不要重复下面的要求。

要求：
1. 只能依据给定案件，不得编造案件。
2. 先用 2-3 句给出整体裁判倾向。
3. 然后给出 2 到 4 个“观点”，每个观点都要点明适用前提和裁判边界。
4. 每个观点尽量带出 1 到 2 个代表性案号，格式直接写案号即可。
5. 语言要像法律分析，不要输出 JSON，不要输出 markdown 表格，不要写“以下是分析”这种空话。
6. 可以使用这种结构：
整体判断：
......

观点一：......
......

观点二：......
......

用户问题：
{query}

可引用案件：
{case_json}

现在直接从“整体判断：”开始输出正文。
""".strip()


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return stripped


def extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif hasattr(item, "text") and getattr(item, "text"):
                parts.append(str(getattr(item, "text")))
        if parts:
            return "\n".join(parts)

    model_extra = getattr(message, "model_extra", None) or {}
    for key in ["reasoning_content", "output_text", "text"]:
        value = model_extra.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def create_mimo_client() -> OpenAI:
    api_key = os.getenv(MIMO_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"缺少 Mimo API Key，请设置环境变量 {MIMO_API_KEY_ENV}。"
        )

    return OpenAI(
        api_key=api_key,
        base_url=MIMO_BASE_URL,
        timeout=120,
    )


def build_mimo_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是严谨的中文法律类案分析助手，只依据检索证据回答，输出纯 JSON。",
        },
        {"role": "user", "content": prompt},
    ]


def extract_delta_text(delta: Any) -> str:
    if delta is None:
        return ""
    content = getattr(delta, "content", None)
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif hasattr(item, "text") and getattr(item, "text"):
                parts.append(str(getattr(item, "text")))
        if parts:
            return "".join(parts)

    model_extra = getattr(delta, "model_extra", None) or {}
    for key in ["text", "content", "reasoning_content"]:
        value = model_extra.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def stream_mimo_text(prompt: str):
    client = create_mimo_client()
    stream = client.chat.completions.create(
        model=MIMO_MODEL,
        temperature=0.1,
        max_tokens=1800,
        response_format={"type": "json_object"},
        stream=True,
        messages=build_mimo_messages(prompt),
    )
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        text = extract_delta_text(delta)
        if text:
            yield text


def stream_mimo_preview_text(prompt: str):
    client = create_mimo_client()
    stream = client.chat.completions.create(
        model=MIMO_MODEL,
        temperature=0.2,
        max_tokens=1200,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": "你是严谨的中文法律类案分析助手，只依据检索证据回答。直接输出分析正文，不要复述任务、不要解释规则、不要说你将要做什么。",
            },
            {"role": "user", "content": prompt},
        ],
    )
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        text = extract_delta_text(delta)
        if text:
            yield text


def call_mimo(prompt: str) -> dict[str, Any]:
    client = create_mimo_client()
    response = client.chat.completions.create(
        model=MIMO_MODEL,
        temperature=0.1,
        max_tokens=1800,
        response_format={"type": "json_object"},
        messages=build_mimo_messages(prompt),
    )
    message = response.choices[0].message
    content = extract_message_text(message)
    if not content.strip():
        choice_extra = getattr(response.choices[0], "model_extra", None) or {}
        alt_text = choice_extra.get("text") or choice_extra.get("output_text") or ""
        content = str(alt_text or "")
    if not content.strip():
        raise RuntimeError("Mimo 返回了空内容。")
    return parse_json_object(content)


def make_fallback_answer(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    viewpoints = []
    for index, item in enumerate(results[:3], start=1):
        case_doc = item.get("case_doc", {})
        chunk = (item.get("matched_chunks") or [{}])[0]
        viewpoints.append(
            {
                "title": f"观点{index}：可重点参考 {case_doc.get('reason') or item.get('reason') or '相关裁判规则'}",
                "analysis": compact_text(
                    chunk.get("chunk_text", "当前可用案例较少，建议继续补充更贴近事实模式的检索样本。"),
                    320,
                ),
                "supporting_cases": [
                    {
                        "doc_id": item.get("doc_id", ""),
                        "reason": compact_text(
                            chunk.get("context_text", "") or chunk.get("chunk_text", ""),
                            160,
                        ),
                    }
                ],
            }
        )
    return {
        "title": "类案分析",
        "summary": f"围绕“{query}”，当前系统已基于召回案例整理出初步观点，但仍建议结合裁判原文核验适用前提和事实细节。",
        "viewpoints": viewpoints,
        "notice": "本次回答采用回退逻辑生成，建议复核案件原文与具体事实差异。",
    }


def normalize_answer(answer: dict[str, Any], results: list[dict[str, Any]], query: str) -> dict[str, Any]:
    valid_ids = {item.get("doc_id", "") for item in results}
    viewpoints = []
    for item in answer.get("viewpoints", []):
        support_items = []
        for support in item.get("supporting_cases", []):
            doc_id = str(support.get("doc_id", "")).strip()
            if doc_id and doc_id in valid_ids:
                support_items.append(
                    {
                        "doc_id": doc_id,
                        "reason": compact_text(support.get("reason", ""), 120),
                    }
                )
        if not support_items:
            continue
        viewpoints.append(
            {
                "title": str(item.get("title", "")).strip() or "观点",
                "analysis": compact_text(str(item.get("analysis", "")).strip(), 520),
                "supporting_cases": support_items,
            }
        )

    if not viewpoints:
        return make_fallback_answer(query, results)

    return {
        "title": str(answer.get("title", "")).strip() or "类案分析",
        "summary": str(answer.get("summary", "")).strip(),
        "viewpoints": viewpoints,
        "notice": str(answer.get("notice", "")).strip(),
    }


def prepare_advisor_context(payload: dict[str, Any]) -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    args = build_search_args(payload)
    if not args.query:
        raise RuntimeError("query 不能为空。")
    retrieval_payload = retrieval.run_search(args)
    results = retrieval_payload.get("results", [])
    return args, results


def build_empty_answer_payload(args: SimpleNamespace) -> dict[str, Any]:
    return {
        "query": args.query,
        "answer": {
            "title": "类案分析",
            "summary": "当前未召回到可用案例，暂时无法生成稳定的类案分析结论。",
            "viewpoints": [],
            "notice": "可以尝试放宽案由、日期或审级过滤，并改写检索问题后重试。",
        },
        "source_cases": {},
        "retrieval": {
            "mode": args.mode,
            "result_count": 0,
            "rerank_enabled": args.rerank,
            "cited_doc_ids": [],
        },
    }


def build_advisor_payload(
    *,
    args: SimpleNamespace,
    results: list[dict[str, Any]],
    answer: dict[str, Any],
) -> dict[str, Any]:
    source_cases = {
        item.get("doc_id", ""): build_case_preview(item)
        for item in results
        if item.get("doc_id")
    }
    cited_ids = []
    for viewpoint in answer.get("viewpoints", []):
        for support in viewpoint.get("supporting_cases", []):
            doc_id = support.get("doc_id")
            if doc_id and doc_id not in cited_ids:
                cited_ids.append(doc_id)

    return {
        "query": args.query,
        "answer": answer,
        "source_cases": source_cases,
        "retrieval": {
            "mode": args.mode,
            "result_count": len(results),
            "rerank_enabled": args.rerank,
            "cited_doc_ids": cited_ids,
        },
    }


def run_advisor_answer(payload: dict[str, Any]) -> dict[str, Any]:
    args, results = prepare_advisor_context(payload)
    if not results:
        return build_empty_answer_payload(args)

    prompt = build_prompt(args.query, results)
    try:
        raw_answer = call_mimo(prompt)
        answer = normalize_answer(raw_answer, results, args.query)
    except Exception:
        answer = make_fallback_answer(args.query, results)
    return build_advisor_payload(args=args, results=results, answer=answer)


def find_anchor(text: str, needles: list[str]) -> int | None:
    candidates = [text.find(needle) for needle in needles if needle and text.find(needle) >= 0]
    if not candidates:
        return None
    return min(candidates)


def extract_case_sections(case_doc: dict[str, Any]) -> list[dict[str, str]]:
    full_text = str(case_doc.get("full_text") or "")
    if not full_text:
        return []

    anchors: list[tuple[int, str, str]] = []
    anchor_specs = [
        ("process", "审理经过", ["本院于", "本案现已审理终结", "依法公开开庭进行了审理"]),
        ("claims", "原告诉称", ["原告向本院提出诉讼请求", "原告诉称", "上诉人请求", "申请人称"]),
        ("defense", "被告辩称", ["被告辩称", "被告未具答辩", "被上诉人辩称", "被申请人称"]),
        ("facts", "本院查明", ["经审理查明", "本院经审理认定", "查明事实如下"]),
        ("reasoning", "本院认为", ["本院认为"]),
        ("judgment", "裁判结果", ["判决如下", "裁定如下"]),
        ("statutes", "附法律依据", ["附：相关法律条文", "附法律依据"]),
    ]

    for section_id, title, needles in anchor_specs:
        position = find_anchor(full_text, needles)
        if position is not None:
            anchors.append((position, section_id, title))

    anchors.sort(key=lambda item: item[0])
    sections: list[dict[str, str]] = []

    first_anchor = anchors[0][0] if anchors else len(full_text)
    header_text = full_text[:first_anchor].strip()
    if header_text:
        sections.append(
            {
                "id": "header",
                "title": "首部信息",
                "content": header_text,
            }
        )

    for index, (start, section_id, title) in enumerate(anchors):
        end = anchors[index + 1][0] if index + 1 < len(anchors) else len(full_text)
        content = full_text[start:end].strip()
        if content:
            sections.append(
                {
                    "id": section_id,
                    "title": title,
                    "content": content,
            }
        )

    sections.sort(
        key=lambda item: {
            "header": 0,
            "process": 1,
            "claims": 2,
            "defense": 3,
            "facts": 4,
            "reasoning": 5,
            "judgment": 6,
            "statutes": 7,
            "full_text": 8,
        }.get(item["id"], 99)
    )

    if not sections and full_text.strip():
        sections.append(
            {
                "id": "full_text",
                "title": "全文",
                "content": full_text.strip(),
            }
        )
    return sections


def build_case_detail(doc_id: str) -> dict[str, Any]:
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
    if not case_doc:
        return {}

    return {
        "doc_id": case_doc.get("doc_id") or doc_id,
        "case_name": case_doc.get("case_name") or "",
        "reason": case_doc.get("reason") or "",
        "trial_level": case_doc.get("trial_level") or "",
        "court_name": case_doc.get("court_name") or "",
        "judge_date": case_doc.get("judge_date") or "",
        "publish_date": case_doc.get("publish_date") or "",
        "litigants": normalize_sequence(case_doc.get("litigants")),
        "statutes": normalize_sequence(case_doc.get("statutes")),
        "full_text": case_doc.get("full_text") or "",
        "full_text_hash": case_doc.get("full_text_hash") or "",
        "sections": extract_case_sections(case_doc),
    }
