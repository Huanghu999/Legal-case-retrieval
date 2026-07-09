# -*- coding: utf-8 -*-
"""
混合召回通道消融实验脚本 (不开 rerank，开启 LLM rewrite)。
逐一移除召回通道组件，评估每个组件对 benchmark 指标的影响。
输出 CSV + JSON 报告，方便判断哪些组件可以精简。

用法:
  python scripts/ablation_study.py
  python scripts/ablation_study.py --limit 10          # 快速调试，只跑 10 条 query
  python scripts/ablation_study.py --output-dir benchmark_runs/ablation_custom
  python scripts/ablation_study.py --configs full_hybrid,bm25_only  # 只跑指定配置
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.legal_case_rag.app import benchmark_service as bm
from src.legal_case_rag.retrieval import search as retrieval
from src.legal_case_rag.app.search_args import build_search_args


# ---------------------------------------------------------------------------
# 环境 & 工具
# ---------------------------------------------------------------------------

REQUIRED_ENV = ["OPENSEARCH_PASSWORD", "SILICONFLOW_API_KEY"]

def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

def validate_env() -> None:
    load_dotenv()
    missing = [n for n in REQUIRED_ENV if not os.getenv(n)]
    if missing:
        raise EnvironmentError(
            f"缺少环境变量: {', '.join(missing)}\n"
            "需要 OpenSearch 和 SiliconFlow API 访问。"
        )

def vo(v: Any) -> float:
    return float(v or 0.0)


# ---------------------------------------------------------------------------
# 消融配置 — 聚焦混合召回通道 (无 rerank，有 LLM rewrite)
# ---------------------------------------------------------------------------

DEFAULT_ROUTE_WEIGHTS = {
    "bm25_raw": 1.0,
    "bm25_focus": 0.95,
    "vector_focus": 1.20,
    "bm25_fine_rule": 1.30,
    "bm25_focus_tags": 1.60,
    "bm25_focus_analysis": 1.80,
    "bm25_fine_tags": 2.00,
}


def _zero(*names: str) -> dict[str, float]:
    """返回一份默认权重的副本，把指定 route 置零。"""
    w = dict(DEFAULT_ROUTE_WEIGHTS)
    for n in names:
        w[n] = 0.0
    return w


def build_ablation_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    def cfg(name: str, desc: str, **kw) -> dict[str, Any]:
        base = {
            "mode": "hybrid",
            "query_profile": True,
            "query_profile_boost": True,
            "llm_query_rewrite": True,
            "route_weight_overrides": {},
        }
        base.update(kw)
        return {"name": name, "desc": desc, **base}

    # ── 基线 ──
    configs.append(cfg("full_hybrid", "全量混合召回 + LLM rewrite (基线)"))

    # ── 双通道对比 ──
    configs.append(cfg("bm25_only", "仅 BM25 通道", mode="bm25"))
    configs.append(cfg("vector_only", "仅向量通道", mode="vector"))

    # ── Query Profile & Boost ──
    configs.append(cfg(
        "no_query_profile",
        "去掉多路展开 (仅 bm25_raw)",
        query_profile=False, query_profile_boost=False,
    ))
    configs.append(cfg(
        "no_profile_boost",
        "去掉 Profile 分值加成 (保留多路)",
        query_profile_boost=False,
    ))

    # ── LLM Rewrite ──
    configs.append(cfg(
        "no_llm_rewrite",
        "去掉 LLM Rewrite",
        llm_query_rewrite=False,
    ))

    # ── 逐一去掉各路由 ──
    for route, label in [
        ("bm25_fine_rule",       "fine_rule 裁判规则"),
        ("bm25_focus_tags",      "focus_tags 争议焦点标签"),
        ("bm25_focus_analysis",  "focus_analysis 焦点评析"),
        ("bm25_fine_tags",       "fine_tags 主叶子+细争点"),
        ("vector_focus",         "vector_focus"),
        ("bm25_focus",           "bm25_focus"),
        ("bm25_raw",             "bm25_raw"),
    ]:
        configs.append(cfg(f"no_{route}", f"去掉 {label}", route_weight_overrides=_zero(route)))

    # ── 组合消融 ──
    configs.append(cfg(
        "raw_and_focus_only",
        "仅保留 raw + focus (4 路)",
        route_weight_overrides=_zero("bm25_fine_rule", "bm25_focus_tags", "bm25_focus_analysis", "bm25_fine_tags"),
    ))
    configs.append(cfg(
        "minimal_bm25_raw",
        "最精简: 仅 bm25_raw",
        mode="bm25", query_profile=False, query_profile_boost=False,
        llm_query_rewrite=False,
    ))

    return configs


# ---------------------------------------------------------------------------
# 运行单个消融
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "ndcg@10", "expected_ndcg@10", "expected_ndcg@20", "expected_ndcg@50",
    "hit@5", "hit@10",
    "recall@20", "recall@50", "recall@100",
    "mrr", "map",
]


def run_single_ablation(
    config: dict[str, Any],
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, dict[str, Any]]],
    top_k: int,
    candidate_size: int,
    chunk_top_k: int,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    metric_rows: list[dict[str, Any]] = []
    cache_path = str(bm.DEFAULT_REWRITE_CACHE_PATH)

    for query in queries:
        payload = {
            "query": query.get("query_text", ""),
            "mode": config["mode"],
            "rerank": False,
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "candidate_size": candidate_size,
            "rerank_top_n": 0,
            "rerank_model_weight": 0,
            "rerank_min_interval_ms": 0,
            "rerank_max_retries": 0,
            "rerank_rank_safe": False,
            "rerank_max_rank_promotion": 0,
            "show_context": False,
            "query_profile": config["query_profile"],
            "query_profile_boost": config["query_profile_boost"],
            "llm_query_rewrite": config["llm_query_rewrite"],
            "llm_rewrite_cache_path": cache_path if config["llm_query_rewrite"] else "",
            "route_weight_overrides": config.get("route_weight_overrides", {}),
        }
        args = build_search_args(
            payload,
            default_config=bm.BENCHMARK_APP_CONFIG,
            retrieval_module=retrieval,
            verify_ssl_default=False,
        )
        try:
            result = retrieval.run_search(args)
        except Exception as exc:
            errors.append({"query_id": str(query.get("query_id", "")), "error": str(exc)})
            continue

        results = result.get("results", [])
        ranking = [item.get("doc_id") for item in results if item.get("doc_id")]
        rels = qrels.get(query["query_id"], {})
        metric = bm.evaluate_single_ranking(query, ranking, rels)
        metric_rows.append(metric)

    return {
        "config_name": config["name"],
        "config_desc": config["desc"],
        "mode": config["mode"],
        "query_profile": config["query_profile"],
        "query_profile_boost": config["query_profile_boost"],
        "llm_query_rewrite": config["llm_query_rewrite"],
        "route_weight_overrides": {k: v for k, v in config.get("route_weight_overrides", {}).items() if v == 0.0},
        "metrics": bm.aggregate_metrics(metric_rows),
        "error_count": len(errors),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# 输出报告
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "rank", "config_name", "config_desc", "mode",
    "query_profile", "query_profile_boost", "llm_query_rewrite",
    "zeroed_routes",
    "queries", "queries_with_positive", "error_count",
    "ndcg@10", "expected_ndcg@10", "expected_ndcg@20", "expected_ndcg@50",
    "hit@5", "hit@10",
    "recall@20", "recall@50", "recall@100",
    "mrr", "map",
    "delta_eNDCG20", "delta_recall20", "delta_mrr",
    "duration_ms",
]


def write_report(
    results: list[dict[str, Any]],
    output_dir: Path,
    baseline_name: str = "full_hybrid",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics: dict[str, float] = {}
    for r in results:
        if r["config_name"] == baseline_name:
            baseline_metrics = {k: vo(v) for k, v in r["metrics"].items() if k in METRIC_NAMES}
            break

    ranked = sorted(results, key=lambda r: vo(r["metrics"].get("expected_ndcg@20")), reverse=True)

    # CSV
    csv_path = output_dir / "ablation_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rank, r in enumerate(ranked, 1):
            m = r["metrics"]
            zeroed = ",".join(sorted(r.get("route_weight_overrides", {}).keys()))
            row = {
                "rank": rank,
                "config_name": r["config_name"],
                "config_desc": r["config_desc"],
                "mode": r["mode"],
                "query_profile": r["query_profile"],
                "query_profile_boost": r["query_profile_boost"],
                "llm_query_rewrite": r["llm_query_rewrite"],
                "zeroed_routes": zeroed,
                "queries": m.get("queries"),
                "queries_with_positive": m.get("queries_with_positive"),
                "error_count": r["error_count"],
                "duration_ms": r.get("duration_ms", 0),
            }
            for name in METRIC_NAMES:
                row[name] = m.get(name)
            for name in ["expected_ndcg@20", "recall@20", "mrr"]:
                val = vo(m.get(name))
                base_val = baseline_metrics.get(name, 0)
                row[f"delta_{name.replace('expected_ndcg@20', 'eNDCG20').replace('recall@20', 'recall20').replace('mrr', 'mrr')}"] = round(val - base_val, 4) if base_val else None
            writer.writerow(row)

    # JSON
    json_path = output_dir / "ablation_report.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "baseline": baseline_name,
        "baseline_metrics": baseline_metrics,
        "experiments": [],
    }
    for r in ranked:
        m = r["metrics"]
        exp = {
            "name": r["config_name"],
            "desc": r["config_desc"],
            "mode": r["mode"],
            "query_profile": r["query_profile"],
            "query_profile_boost": r["query_profile_boost"],
            "llm_query_rewrite": r["llm_query_rewrite"],
            "zeroed_routes": sorted(r.get("route_weight_overrides", {}).keys()),
            "metrics": {k: m.get(k) for k in METRIC_NAMES},
            "delta": {},
            "error_count": r["error_count"],
            "duration_ms": r.get("duration_ms", 0),
        }
        for name in METRIC_NAMES:
            val = vo(m.get(name))
            base_val = baseline_metrics.get(name, 0)
            exp["delta"][name] = round(val - base_val, 4) if base_val else None
        report["experiments"].append(exp)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    b_end20 = baseline_metrics.get("expected_ndcg@20", 0)
    b_r20 = baseline_metrics.get("recall@20", 0)
    b_mrr = baseline_metrics.get("mrr", 0)

    print(f"\n{'='*110}")
    print(f" 混合召回消融实验结果 (无 Rerank, 有 LLM Rewrite)  |  按 expected_ndcg@20 排序")
    print(f"{'='*110}")
    print(f"{'#':<3} {'配置':<30} {'eNDCG@20':>10} {'Δ':>8} {'Recall@20':>10} {'Δ':>8} {'MRR':>8} {'Δ':>8} {'NDCG@10':>10}")
    print(f"{'-'*110}")
    for rank, r in enumerate(ranked, 1):
        m = r["metrics"]
        e20 = vo(m.get("expected_ndcg@20"))
        rc20 = vo(m.get("recall@20"))
        mrr_v = vo(m.get("mrr"))
        n10 = vo(m.get("ndcg@10"))
        d_e = e20 - b_end20
        d_r = rc20 - b_r20
        d_m = mrr_v - b_mrr
        tag = " <-- baseline" if r["config_name"] == baseline_name else ""
        print(
            f"{rank:<3} {r['config_name']:<30} "
            f"{e20:>10.4f} {d_e:>+8.4f} "
            f"{rc20:>10.4f} {d_r:>+8.4f} "
            f"{mrr_v:>8.4f} {d_m:>+8.4f} "
            f"{n10:>10.4f}{tag}"
        )

    # 精简建议
    print(f"\n{'='*110}")
    print(" 精简建议 (Δ eNDCG@20 > -0.005 的组件可考虑移除)")
    print(f"{'='*110}")
    removable = []
    for r in ranked:
        if r["config_name"] == baseline_name:
            continue
        m = r["metrics"]
        delta = vo(m.get("expected_ndcg@20")) - b_end20
        if delta > -0.005:
            removable.append((r["config_name"], r["desc"], delta))
    if removable:
        for name, desc, delta in sorted(removable, key=lambda x: -x[2]):
            print(f"  [OK] {name:<30} delta={delta:+.4f}  {desc}")
    else:
        print("  [X] all components cause >0.5% drop when removed")

    harmful = []
    for r in ranked:
        if r["config_name"] == baseline_name:
            continue
        m = r["metrics"]
        delta = vo(m.get("expected_ndcg@20")) - b_end20
        if delta < -0.01:
            harmful.append((r["config_name"], r["desc"], delta))
    if harmful:
        print(f"\n 核心组件 (移除后 eNDCG@20 下降 > 1%):")
        for name, desc, delta in sorted(harmful, key=lambda x: x[2]):
            print(f"  [X] {name:<30} delta={delta:+.4f}  {desc}")

    print(f"\n结果已保存: {csv_path} / {json_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="混合召回通道消融实验 (无 Rerank)")
    p.add_argument("--limit", type=int, default=58)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--candidate-size", type=int, default=300)
    p.add_argument("--chunk-top-k", type=int, default=2)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--configs", type=str, default="", help="逗号分隔的配置名")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_env()
    except EnvironmentError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_dir = args.output_dir or Path("benchmark_runs") / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    queries = bm.load_benchmark_queries(limit=args.limit)
    qrels = bm.load_benchmark_qrels()
    print(f"加载 {len(queries)} 条 query, {len(qrels)} 条 qrels")

    configs = build_ablation_configs()
    if args.configs:
        selected = set(args.configs.split(","))
        configs = [c for c in configs if c["name"] in selected]
        if not configs:
            print(f"未找到指定配置: {args.configs}", file=sys.stderr)
            return 1

    print(f"共 {len(configs)} 个消融配置 (无 Rerank, 有 LLM Rewrite)\n")

    results: list[dict[str, Any]] = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {config['name']}: {config['desc']} ...", end="", flush=True)
        started = time.perf_counter()
        result = run_single_ablation(
            config, queries, qrels,
            top_k=args.top_k,
            candidate_size=args.candidate_size,
            chunk_top_k=args.chunk_top_k,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        result["duration_ms"] = duration_ms
        results.append(result)

        m = result["metrics"]
        print(
            f"  eNDCG@20={vo(m.get('expected_ndcg@20')):.4f}  "
            f"Recall@20={vo(m.get('recall@20')):.4f}  "
            f"MRR={vo(m.get('mrr')):.4f}  "
            f"({duration_ms}ms)"
        )

    write_report(results, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
