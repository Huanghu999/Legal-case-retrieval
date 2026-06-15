# -*- coding: utf-8 -*-
"""
CaseLaw-Bench 独立评测脚本(纯 Python 标准库,无第三方依赖)。
口径:NDCG@10 用连续 grade_期望;Recall@20 / MRR / MAP 用 grade_主档≥2(类案阈值)。
未在 qrels 列出的候选记 grade 0(TREC 约定)。按 query 宏平均。含/不含锚点两版。

用法:
  python eval.py --qrels data/qrels.jsonl --run your_run.jsonl
  python eval.py --selftest        # 自测:oracle 应 NDCG@10≈1
run 文件每行:{"query_id": "...", "ranking": ["doc_id", ...]}
"""
import json, argparse, math

REL_THRESH = 2  # 类案阈值 grade_主档≥2


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_qrels(path):
    out = {}
    for r in read_jsonl(path):
        d = {c["doc_id"]: (float(c.get("grade_期望", 0)), int(c.get("grade_主档", 0)))
             for c in r.get("candidates", [])}
        out[r["query_id"]] = {"doc": d, "src": r.get("query_source_doc")}
    return out


def dcg(gs):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gs))


def ndcg(ranking, gmap, k=10):
    idcg = dcg(sorted(gmap.values(), reverse=True)[:k])
    return dcg([gmap.get(d, 0.0) for d in ranking[:k]]) / idcg if idcg > 0 else 0.0


def recall(ranking, rel, k=20):
    return None if not rel else sum(1 for d in ranking[:k] if d in rel) / len(rel)


def mrr(ranking, rel):
    for i, d in enumerate(ranking, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ap(ranking, rel):
    if not rel:
        return None
    hit = s = 0
    for i, d in enumerate(ranking, 1):
        if d in rel:
            hit += 1; s += hit / i
    return s / len(rel)


def evaluate(qrels, runs, drop_anchor=False):
    nd, rc, rr, mp = [], [], [], []
    for qid, ranking in runs.items():
        if qid not in qrels:
            continue
        q = qrels[qid]
        gmap = {d: ge for d, (ge, gm) in q["doc"].items()}
        rel = {d for d, (ge, gm) in q["doc"].items() if gm >= REL_THRESH}
        rk = ranking
        if drop_anchor and q.get("src"):
            rk = [d for d in rk if d != q["src"]]
            gmap = {d: v for d, v in gmap.items() if d != q["src"]}
            rel -= {q["src"]}
        nd.append(ndcg(rk, gmap))   # NDCG 用连续 grade_期望,含无类案 query
        if rel:                      # 二值指标跳过无正例 query(IR 惯例)
            rc.append(recall(rk, rel)); rr.append(mrr(rk, rel)); mp.append(ap(rk, rel))
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {"NDCG@10": avg(nd), "Recall@20": avg(rc), "MRR": avg(rr), "MAP": avg(mp), "n_query": len(nd)}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--qrels", default="data/qrels.jsonl")
    ap_.add_argument("--run", default="")
    ap_.add_argument("--selftest", action="store_true")
    a = ap_.parse_args()
    qrels = load_qrels(a.qrels)
    print(f"qrels: {len(qrels)} queries")
    if a.selftest or not a.run:
        allids = sorted({d for q in qrels.values() for d in q["doc"]})
        oracle = {qid: sorted(q["doc"], key=lambda d: -q["doc"][d][0]) +
                  [d for d in allids if d not in q["doc"]] for qid, q in qrels.items()}
        rnd = {qid: sorted(allids, key=lambda d: hash((qid, d))) for qid in qrels}
        print("oracle:", evaluate(qrels, oracle))
        print("random:", evaluate(qrels, rnd))
        return
    runs = {r["query_id"]: r["ranking"] for r in read_jsonl(a.run)}
    print("含锚点  :", evaluate(qrels, runs, drop_anchor=False))
    print("不含锚点:", evaluate(qrels, runs, drop_anchor=True))


if __name__ == "__main__":
    main()
