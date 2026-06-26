from types import SimpleNamespace

from src.legal_case_rag.retrieval.llm_query_rewriter import (
    LlmQueryRewrite,
    build_rewrite_messages,
    parse_rewrite_response,
    rewrite_query_with_llm,
)


def test_parse_rewrite_response_accepts_aligned_fields():
    rewrite = parse_rewrite_response(
        """
        {
          "expanded_query": "买方认证抵扣增值税专用发票 微信对账 事实买卖合同成立",
          "legal_issue": "无书面合同情况下认定事实买卖合同关系成立",
          "fact_elements": "买方认证抵扣发票 双方微信对账 送货单主体不明",
          "statutes": "合同法第八条",
          "main_leaf": "A1_口头或事实买卖合同成立认定",
          "focus_labels": ["合同成立与否", "货款给付"]
        }
        """
    )

    assert rewrite.used is True
    assert "事实买卖合同" in rewrite.expanded_query
    assert rewrite.main_leaf == "A1_口头或事实买卖合同成立认定"
    assert rewrite.focus_labels == ["合同成立与否", "货款给付"]


def test_parse_rewrite_response_falls_back_for_invalid_json():
    rewrite = parse_rewrite_response("not json")

    assert rewrite.used is False
    assert rewrite.fallback_reason == "invalid_json"


def test_parse_rewrite_response_drops_dirty_fields():
    rewrite = parse_rewrite_response(
        {
            "expanded_query": "x" * 200,
            "legal_issue": "合同相对方认定",
            "fact_elements": ["wrong"],
            "focus_labels": ["合同成立与否", 3, " "],
            "unknown": "ignored",
        }
    )

    assert rewrite.used is True
    assert rewrite.expanded_query == ""
    assert rewrite.legal_issue == "合同相对方认定"
    assert rewrite.fact_elements == ""
    assert rewrite.focus_labels == ["合同成立与否"]


def test_rewrite_query_with_llm_falls_back_when_disabled():
    rewrite = rewrite_query_with_llm("法院如何认定合同相对方？", enabled=False)

    assert rewrite == LlmQueryRewrite(fallback_reason="disabled")


def test_rewrite_query_with_llm_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    rewrite = rewrite_query_with_llm("法院如何认定合同相对方？", enabled=True)

    assert rewrite == LlmQueryRewrite(fallback_reason="missing_api_key")


def test_rewrite_query_with_llm_reads_openai_compatible_response():
    message = SimpleNamespace(content='{"expanded_query":"事实买卖合同成立","legal_issue":"合同成立与否","fact_elements":"微信对账"}')
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])

    class ChatCompletions:
        def create(self, **kwargs):
            return response

    class Client:
        chat = SimpleNamespace(completions=ChatCompletions())

    rewrite = rewrite_query_with_llm("能否认定事实买卖合同？", client_factory=lambda: Client())

    assert rewrite.used is True
    assert rewrite.expanded_query == "事实买卖合同成立"


def test_build_rewrite_messages_include_field_aligned_few_shots():
    messages = build_rewrite_messages("没有书面合同但有微信对账，能否认定买卖合同成立？")
    prompt_text = "\n".join(message["content"] for message in messages)

    assert "示例1" in prompt_text
    assert "示例2" in prompt_text
    assert "事实买卖合同成立" in prompt_text
    assert "focus_labels" in prompt_text
    assert "不要回答法律问题" in prompt_text
