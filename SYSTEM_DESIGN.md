# 类案召回 RAG 系统设计说明

## 1. 系统定位

当前项目的核心目标是基于 CaseLaw-Bench 标注数据集，构建一个可检索、可调参、可评估的类案召回 RAG 系统。

系统目前主要解决两件事：

- 给定一段法律事实或争议问题，召回相似案件，并展示命中的案件、片段和分数。
- 使用标注好的 `queries.jsonl` / `qrels.jsonl` 对召回链路做离线评估，观察 Hybrid 召回和 Hybrid + Rerank 的效果差异。

当前评估只关注类案召回与排序质量，不评估最终大模型生成回答质量。

## 2. 数据来源

当前系统使用的数据源固定为：

```text
data/caselaw-benchmark release/data/
```

核心文件包括：

- `corpus.jsonl`：案件库，共 751 个案件，是索引构建的唯一案件来源。
- `queries.jsonl`：评估查询，共 58 条 query，每条 query 有自然语言问题、来源案件、主叶子、难度、陷阱等字段。
- `qrels.jsonl`：标准答案，每条 query 对应若干 candidate 案件及相关性标注。
- `qrels.explained.jsonl`：带解释的标注文件，主要用于人工分析，不参与默认评估计算。

评估中的正例定义为：

```text
grade_主档 >= 2
```

其中：

- `grade=3` / `grade=2`：强相关或可接受的类案，计为正例。
- `grade=1`：弱相关干扰项，不计为正例。
- `grade=0`：负例。

主报告采用去锚点口径，即评估时会排除 query 自身来源案件 `query_source_doc`。

## 3. 索引构建流程

索引重建入口是：

```powershell
python scripts\rebuild_benchmark_rag_index.py
```

该脚本会依次执行三步：

1. 从 `corpus.jsonl` 构建案件级 JSONL 和 chunk 级 JSONL。
2. 对 chunk 文本调用 embedding 服务生成向量。
3. 将案件和 chunk 写入 OpenSearch。

默认输出目录：

```text
benchmark_dataset/
```

默认索引名：

```text
caselaw_benchmark_cases_v1
caselaw_benchmark_chunks_v1
```

索引构建代码位于：

```text
src/legal_case_rag/data_pipeline/benchmark_dataset_builder.py
src/legal_case_rag/data_pipeline/embed_chunks.py
src/legal_case_rag/data_pipeline/opensearch_ingest.py
```

## 4. Chunk 设计

系统不是把整篇案件文书直接作为一个检索单元，而是将每个案件拆成多个结构化 section。

当前主要 section 包括：

- `case_profile`：案件画像，包含案由、审级、法院、法律关系、标的物类型等。
- `fine_issue`：细争点，包含主叶子、细争点、裁判规则争点等。
- `focus`：争议焦点，包含焦点标签、案情核心、法律争点、裁判要旨等。
- `claims`：诉称。
- `facts`：查明事实。
- `reasoning`：本院认为。
- `judgment`：裁判结果。
- `statutes`：引用法条。

不同 section 有不同权重：

```text
fine_issue   1.55
focus        1.45
case_profile 1.20
reasoning    1.15
facts        1.00
claims       0.75
judgment     0.70
statutes     0.45
```

设计思路是：类案召回不应该只看事实相似，更应该优先匹配细争点、争议焦点、裁判规则和本院认为。

## 5. 检索入口

Web 检索入口是：

```text
legal_rag_web.py
```

主要接口：

- `/api/search`：单条 query 检索。
- `/api/benchmark/evaluate`：运行 benchmark 评估。
- `/api/cases/<doc_id>`：按 `doc_id` 回查案件全文。

前端页面：

```text
templates/legal_rag_index.html
static/legal_rag.js
static/legal_rag.css
```

启动方式：

```powershell
python legal_rag_web.py
```

默认访问：

```text
http://127.0.0.1:7860/
```

## 6. Query Profile

系统会先对用户输入的自然语言 query 做轻量分析，构造 query profile。

相关代码：

```text
src/legal_case_rag/retrieval/query_profile.py
```

Query profile 会抽取：

- 案由或法律关系词。
- 请求类型词，例如货款、对账、发票、违约金、逾期利息等。
- 争议焦点词，例如是否成立、能否支持、举证责任、合同相对方等。
- 否定事实词，例如未实际发货、拒绝支付、主体不明、未能证明等。
- 引用法条。
- 预期裁判倾向。

这些信息不会使用 qrels 标注，也不会使用 `query_source_doc`、难度、陷阱等评估字段。

## 7. 多路召回设计

当前系统采用多路召回，而不是单一路径检索。

主要 route 包括：

- `bm25_raw`：原始 query 的 BM25 召回。
- `vector_raw`：原始 query 的向量召回。
- `bm25_focus`：基于争议焦点、请求类型、关键事实、法律关系构造的 BM25 召回。
- `vector_focus`：基于焦点 query 的向量召回。
- `bm25_negative`：基于否定事实、请求类型、法律关系构造的 BM25 召回。
- `bm25_legal`：基于案由、法律关系、法条、请求类型构造的 BM25 召回。

多路召回的结果通过 RRF 融合：

```text
Reciprocal Rank Fusion
```

这样做的目的是避免单一路径遗漏：BM25 擅长精确词匹配，向量召回擅长语义泛化，焦点和否定事实 route 用于强化法律争点匹配。

## 8. 案件级聚合

OpenSearch 返回的是 chunk 级结果，但 benchmark 评估的是案件级 `doc_id`。

因此系统会把多个 chunk 命中聚合成一个 case hit。

聚合逻辑位于：

```text
src/legal_case_rag/retrieval/search.py
```

核心函数：

```text
aggregate_case_hits()
```

案件级分数主要考虑：

- 命中 chunk 的基础分数。
- chunk 所属 section 权重。
- 是否命中关键 section，如 `fine_issue`、`focus`、`reasoning`、`facts`。
- 同一案件是否被多个 route 命中。
- 同一案件是否有多个 chunk 命中。
- top chunk 是否来自关键 section。

聚合后的结果才会进入前端展示和 benchmark 评估。

## 9. Rerank 设计

当前 rerank 是案件级 rerank，不再直接对 chunk 做最终排序。

流程是：

1. Hybrid 先召回并聚合出案件级 TopN。
2. 为每个候选案件构造 rerank passage。
3. 调用 BGE reranker 服务得到相关性分数。
4. 将 hybrid 分数和 rerank 分数做融合。

默认模型：

```text
BAAI/bge-reranker-v2-m3
```

默认融合方式：

```text
final_score = hybrid_norm * (1 - rerank_model_weight) + rerank_norm * rerank_model_weight
```

当前默认：

```text
rerank_model_weight = 0.35
```

但从当前结果看，如果主要追求 `Recall@20`，这个权重可能偏高。更稳的实验范围是：

```text
0.15 - 0.30
```

## 10. 前端评估能力

`legal_rag_web.py` 页面已经集成 benchmark 评估。

评估入口会读取：

```text
queries.jsonl
qrels.jsonl
```

默认评估：

- query 数：58。
- TopK：100。
- candidate size：300。
- 方法：`hybrid` 和 `hybrid_rerank`。
- 主口径：去锚点。

输出指标包括：

- `NDCG@10`
- `Recall@20`
- `Recall@50`
- `Recall@100`
- `MRR`
- `MAP`
- `Hit@5`
- `Hit@10`

页面也支持调节：

- candidate size。
- rerank topN。
- rerank 融合权重。
- rerank 限流间隔。
- rerank 重试次数。

## 11. 当前性能现象

目前系统的主要现象是：

- Hybrid 召回已经能接近官方 BM25 基线附近。
- 加 rerank 后，部分排序指标可能提升，例如 `MRR`、`NDCG@10`。
- 但 `Recall@20` 有时会下降。

这个现象是合理的，原因主要是：

- 数据集中 `grade=1` 弱相关干扰项非常多。
- Reranker 容易把语义表面相似的弱相关案件排到前面。
- 如果 rerank 权重太高，会破坏 hybrid 原本较稳定的召回排序。
- Rerank 服务有时会限流，导致 hybrid 和 rerank 的成功 query 数不一致，比较口径可能不完全公平。

## 12. 当前主要瓶颈

当前系统最大的瓶颈不是“召回不到”，而是“召回后的强弱相关区分不够稳”。

具体表现为：

- `Recall@100` 相对较高，说明候选池里已有不少正例。
- `Recall@20` 和 `NDCG@10` 仍有提升空间，说明 Top20 排序还不够稳定。
- `grade=1` 弱相关案件容易抢占前排。
- Rerank passage 仍可能包含噪声，导致 reranker 更关注事实相似，而不是争点和裁判规则相似。

## 13. 下一步优化方向

建议按低风险到高收益的顺序继续优化。

### 13.1 评估诊断增强

优先补充：

- paired-only 对比：只比较 hybrid 和 rerank 都成功的 query。
- per-query delta：展示每条 query rerank 前后指标变化。
- grade=1 干扰统计：统计 Top20 中弱相关案件数量。
- 按主叶子、难度、陷阱分组观察掉分来源。

这一步能让后续优化不再靠感觉调参。

### 13.2 Rerank 稳定性优化

建议尝试：

- 降低默认 rerank 权重到 `0.20` 或 `0.25`。
- 限制 rerank 后的最大上升名次。
- 分桶 rerank，例如只在 Hybrid 1-20、21-50、51-100 内部分别重排。
- 优化 rerank passage，突出细争点、争议焦点、本院认为，减少诉称和长事实噪声。

### 13.3 召回策略优化

可以继续加强：

- 分 section BM25 字段权重。
- 针对主叶子和细争点的 query expansion。
- 对 `fine_issue`、`focus`、`reasoning` 做专门 route。
- 对 negative facts 做更精细的否定事实匹配。
- 自动网格搜索 route 权重和 rerank 权重。

### 13.4 模型替换实验

目前不建议优先替换 embedding 模型。

原因是当前 `Recall@100` 已经不低，说明向量召回不是最大短板。

Reranker 可以作为第二阶段做 A/B 测试，但在替换模型前，应先完成 rerank 诊断和融合策略优化。

## 14. 总体设计思想

当前系统的核心设计思想可以概括为：

```text
结构化案件切分
+ 法律关键要素 query profile
+ BM25 / 向量多路召回
+ RRF 融合
+ 案件级聚合
+ 可控权重 rerank
+ 标注集闭环评估
```

它不是一个单纯的向量搜索系统，而是一个面向法律类案召回的混合检索系统。

系统当前已经具备完整闭环：数据构建、索引、召回、重排、前端调试、benchmark 评估。接下来最重要的是增强诊断能力和 rerank 稳定性，让优化从“看总分”变成“知道哪类 query 为什么掉分”。
