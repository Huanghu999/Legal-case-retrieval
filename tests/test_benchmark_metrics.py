import math

from src.legal_case_rag.app.benchmark_service import (
    aggregate_metrics,
    build_method_comparison,
    build_query_diagnosis_export_rows,
    build_query_diagnosis_summary,
    diagnose_benchmark_query,
    dcg,
    evaluate_single_ranking,
)


def linear_dcg(gains):
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def test_evaluate_single_ranking_reports_expected_ndcg_from_expected_grade():
    query = {
        "query_id": "Q-test",
        "difficulty": "medium",
        "trap": False,
        "main_leaf": "contract",
    }
    qrels = {
        "A": {"grade_主档": 3, "grade_期望": 3.0},
        "B": {"grade_主档": 2, "grade_期望": 2.0},
        "C": {"grade_主档": 1, "grade_期望": 1.0},
    }
    ranking = ["B", "X", "A", "B", "C"]

    metrics = evaluate_single_ranking(query, ranking, qrels)

    actual_dcg = linear_dcg([2.0, 0.0, 3.0, 1.0])
    ideal_dcg = linear_dcg([3.0, 2.0, 1.0])
    expected = actual_dcg / ideal_dcg
    legacy_actual_dcg = dcg([2.0, 0.0, 3.0, 1.0])
    legacy_ideal_dcg = dcg([3.0, 2.0, 1.0])
    legacy_expected = legacy_actual_dcg / legacy_ideal_dcg

    assert math.isclose(metrics["expected_ndcg@10"], expected)
    assert math.isclose(metrics["expected_ndcg@20"], expected)
    assert math.isclose(metrics["expected_ndcg@50"], expected)
    assert math.isclose(metrics["ndcg@10"], legacy_expected)


def test_aggregate_metrics_includes_expected_ndcg_values():
    rows = [
        {"positive_count": 1, "expected_ndcg@20": 0.25},
        {"positive_count": 2, "expected_ndcg@20": 0.75},
    ]

    metrics = aggregate_metrics(rows)

    assert metrics["expected_ndcg@20"] == 0.5


def test_method_comparison_reports_expected_ndcg_delta():
    comparison = build_method_comparison(
        {
            "hybrid": {
                "queries": [
                    {
                        "query_id": "Q-test",
                        "query_text": "query",
                        "metrics": {"expected_ndcg@20": 0.25, "ndcg@10": 0.2, "mrr": 0.5},
                    }
                ]
            },
            "hybrid_rerank": {
                "queries": [
                    {
                        "query_id": "Q-test",
                        "query_text": "query",
                        "metrics": {"expected_ndcg@20": 0.75, "ndcg@10": 0.3, "mrr": 1.0},
                    }
                ]
            },
        }
    )

    row = comparison["queries"][0]

    assert row["delta_expected_ndcg@20"] == 0.5


def test_diagnose_benchmark_query_marks_recall_failure():
    row = {
        "query_id": "Q-001",
        "metrics": {
            "positive_count": 3,
            "expected_ndcg@20": 0.1,
        },
        "top20_has_positive": False,
        "top100_has_positive": False,
        "weak_top20_count": 0,
        "first_positive_rank": None,
    }

    diagnosis = diagnose_benchmark_query(row)

    assert diagnosis["label"] == "recall_failure"
    assert diagnosis["bucket"] == "recall"


def test_diagnose_benchmark_query_marks_front_pollution():
    row = {
        "query_id": "Q-002",
        "metrics": {
            "positive_count": 4,
            "expected_ndcg@20": 0.52,
        },
        "top20_has_positive": True,
        "top100_has_positive": True,
        "weak_top20_count": 12,
        "first_positive_rank": 8,
    }

    diagnosis = diagnose_benchmark_query(row)

    assert diagnosis["label"] == "front_pollution"
    assert diagnosis["bucket"] == "ranking"


def test_build_query_diagnosis_summary_counts_labels():
    rows = [
        {
            "query_id": "Q-001",
            "diagnosis": {"label": "recall_failure", "bucket": "recall"},
        },
        {
            "query_id": "Q-002",
            "diagnosis": {"label": "front_pollution", "bucket": "ranking"},
        },
        {
            "query_id": "Q-003",
            "diagnosis": {"label": "good", "bucket": "good"},
        },
    ]

    summary = build_query_diagnosis_summary(rows)

    assert summary["total_queries"] == 3
    assert summary["labels"]["recall_failure"] == 1
    assert summary["labels"]["front_pollution"] == 1
    assert summary["labels"]["good"] == 1
    assert summary["buckets"]["ranking"] == 1


def test_build_query_diagnosis_export_rows_flattens_query_fields():
    method_payload = {
        "queries": [
            {
                "query_id": "Q-001",
                "query_text": "test query",
                "difficulty": "hard",
                "trap": True,
                "main_leaf": "contract",
                "positive_doc_ids": ["A", "B"],
                "missed_positive_doc_ids": ["B"],
                "first_positive_rank": 12,
                "weak_top20_count": 11,
                "failure_type": "ranking_failure",
                "diagnosis": {
                    "label": "front_pollution",
                    "bucket": "ranking",
                    "reason": "top 20 contains too many weakly related cases",
                },
                "metrics": {
                    "positive_count": 2,
                    "expected_ndcg@20": 0.42,
                    "recall@20": 0.5,
                    "recall@100": 1.0,
                    "mrr": 0.2,
                },
            }
        ]
    }

    rows = build_query_diagnosis_export_rows("hybrid_rerank", method_payload)

    assert len(rows) == 1
    row = rows[0]
    assert row["method"] == "hybrid_rerank"
    assert row["query_id"] == "Q-001"
    assert row["diagnosis_label"] == "front_pollution"
    assert row["missed_positive_doc_ids"] == "B"
