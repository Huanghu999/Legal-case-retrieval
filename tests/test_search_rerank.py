from src.legal_case_rag.retrieval import search_rerank


def test_case_rerank_uses_separate_guardrail_query(monkeypatch):
    captured = {}

    def fake_request_rerank_scores(**kwargs):
        captured["model_query"] = kwargs["query"]
        return [{"index": 0, "relevance_score": 0.9}]

    def fake_guardrail_adjustment(query, case_hit):
        captured["guardrail_query"] = query
        return {"bonus": 0.0, "penalty": 0.0, "missing": [], "conflicts": []}

    monkeypatch.setattr(search_rerank, "request_rerank_scores", fake_request_rerank_scores)
    monkeypatch.setattr(search_rerank, "rerank_guardrail_adjustment", fake_guardrail_adjustment)

    case_hits = [
        {
            "doc_id": "case-1",
            "case_score": 1.0,
            "matched_chunks": [],
        }
    ]

    search_rerank.rerank_case_hits(
        query="模型query：原始问题 核心争点 关键事实",
        guardrail_query="规则query：原始问题 否定事实",
        case_hits=case_hits,
        model_name="reranker",
        api_key="key",
        endpoint="https://example.test/rerank",
        top_n=1,
        timeout=5,
        max_chunks_per_doc=8,
        overlap_tokens=16,
        rank_safe=False,
    )

    assert captured["model_query"] == "模型query：原始问题 核心争点 关键事实"
    assert captured["guardrail_query"] == "规则query：原始问题 否定事实"
