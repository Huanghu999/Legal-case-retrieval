# 类案召回评估系统深度说明

本文档说明当前项目中 CaseLaw-Bench 评估系统的完整逻辑：数据集是什么、系统如何跑评估、指标如何计算、结果如何解释、当前设计是否合理，以及下一阶段应该怎样改进。

当前评估主口径只有一套：**完整 ranking 评估**。`query_source_doc` 会作为来源案件和调试锚点保留在候选池、检索结果和明细报告中，但不会从排名中剔除，也不会再单独计算“去锚点后”的主指标。

## 1. 评估对象

本系统评估的是类案检索链路，而不是最终生成式问答质量。也就是说，评估关注点不是“模型回答得像不像法律意见”，而是：

- 给定一个自然语言法律问题，系统能否召回相关案件。
- 召回后，强相关案件能否排在前面。
- Hybrid 召回、query profile、多路召回、case 聚合、rerank 等模块对最终排序是否有帮助。
- 系统在哪类 query 上稳定，在哪类 query 上容易掉分。
- 评估结果是否能指导后续工程调参，而不是只给出一个总分。

这件事在法律检索里很重要。类案检索并不是普通语义相似度任务，它要同时匹配基本事实、争议焦点、法律关系、裁判规则和关键抗辩。两个案件表面都在讲“欠款”或“发票”，但如果一个真正的裁判规则落在股东混同，另一个落在买卖合同交付义务，它们就不应被当作强类案。

## 2. 数据集说明

当前评估数据来自：

```text
data/caselaw-benchmark release/data/
```

核心文件如下：

| 文件 | 作用 |
|---|---|
| `corpus.jsonl` | 候选案件库，共 751 篇结构化案件，是索引构建和检索评估的目标池。 |
| `queries.jsonl` | 58 条自然语言 query，每条包含 `query_id`、`query_text`、`query_source_doc`、主叶子、难度、陷阱标记等字段。 |
| `qrels.jsonl` | 标准答案文件，每条 query 对应一组候选案件及分级相关性。 |
| `qrels.explained.jsonl` | 带解释的标注版本，主要用于人工分析和错误复盘，不参与默认指标计算。 |
| `qrels.trec.txt` | TREC 格式 qrels，便于和外部 IR 工具对齐。 |

数据集当前范围是中文民商事类案检索，Phase 1 聚焦上海、买卖合同纠纷、2022 年案件。规模为 58 条 query、751 个候选案件、43,558 个已判对，并包含 19 条“表里不一”的陷阱 query。

## 3. 标注口径

`qrels.jsonl` 中每个候选案件有多种相关性字段，当前系统主要使用：

| 字段 | 用途 |
|---|---|
| `grade_主档` | 离散相关性等级，当前二值正例判断使用它。 |
| `grade_期望` | 连续相关性期望，当前 NDCG 增益使用它。 |
| `grade_std` | 标注不确定度，可用于人工复核优先级和风险分析。 |

当前正例定义为：

```text
grade_主档 >= 2
```

等级含义：

| grade | 含义 | 当前评估处理 |
|---|---|---|
| 3 | 高度类案，法律争点和裁判规则高度一致 | 正例 |
| 2 | 类案，需要类比调整但仍有实质参考价值 | 正例 |
| 1 | 弱相关，表面类似或同属买卖框架但关键规则不同 | 非正例，但作为干扰项分析 |
| 0 | 不相关或无效 | 非正例 |

这个设计的关键点是，`grade=1` 不算正例。因为单一买卖合同数据池里弱相关项非常多，如果把 `grade=1` 也算正例，系统只要检索到“同题材案件”就能拿分，无法逼迫模型区分真正的裁判规则相似。

## 4. 为什么使用完整 ranking

早期评估系统曾经同时展示“含锚点”和“去锚点”两套口径。现在已经统一为完整 ranking，原因有四个。

第一，真实产品不会在用户检索时强行剔除来源案件。`query_source_doc` 本质上是 query 的高相关来源案件，把它留在候选池中更贴近产品行为。

第二，剔除锚点会引入额外的人为任务定义。系统到底是在检索“包括来源案在内的类案”，还是检索“除来源案之外的相似案”，这是两个任务。当前项目目标是类案召回工作台，而不是专门做 leave-one-out 检索。

第三，完整 ranking 更利于端到端回归。用户看到的就是完整排序，benchmark 也应该评价这条完整链路的输出。

第四，`query_source_doc` 仍然保留诊断价值。它可以帮助判断系统是否连最明显的来源案件都找不到，但它不再改变指标计算口径。

因此当前主报告中所有指标都基于完整 ranking：

```text
ranking = [doc_id_1, doc_id_2, ...]
metrics = f(ranking, qrels)
```

不会执行：

```text
ranking_without_anchor = ranking - query_source_doc
```

## 5. 评估运行链路

Web 评估入口位于 [legal_rag_web.py](d:/2405data/legal_rag_web.py) 的 `/api/benchmark/evaluate`。页面上的 benchmark 按真实检索链路逐条运行 query，而不是读取预生成 run 文件。

每条 query 的流程是：

1. 从 `queries.jsonl` 读取 `query_text`。
2. 调用当前在线检索链路，通常是 `hybrid` 或 `hybrid_rerank`。
3. 获取最终案件级 ranking。
4. 对 ranking 按 `doc_id` 去重，保证每个案件只计一次。
5. 读取该 query 在 `qrels.jsonl` 中的相关性标注。
6. 计算单 query 指标。
7. 汇总所有成功 query 的宏平均。
8. 生成按难度、陷阱、主叶子的分组指标。
9. 对 `hybrid` 与 `hybrid_rerank` 做 paired-only delta 诊断。

这里的“真实检索链路”包括 query profile、多路 BM25 / vector 召回、case-level RRF、case-level rerank、rank-safe 限制等逻辑。因此评估结果代表当前系统整体表现，不是某个孤立模块的离线模拟分数。

## 6. 指标定义

当前主指标包括：

| 指标 | 含义 | 使用字段 |
|---|---|---|
| `NDCG@10` | 前 10 个结果的分级排序质量 | `grade_期望` |
| `Recall@20` | 前 20 个结果覆盖了多少正例 | `grade_主档 >= 2` |
| `Recall@50` | 前 50 个结果覆盖了多少正例 | `grade_主档 >= 2` |
| `Recall@100` | 前 100 个结果覆盖了多少正例 | `grade_主档 >= 2` |
| `MRR` | 第一个正例出现得有多早 | `grade_主档 >= 2` |
| `MAP` | 正例在整个 ranking 中的平均精度 | `grade_主档 >= 2` |
| `Hit@5` | 前 5 是否至少命中一个正例 | `grade_主档 >= 2` |
| `Hit@10` | 前 10 是否至少命中一个正例 | `grade_主档 >= 2` |

### NDCG@10

NDCG 衡量前排结果是否既相关又排序正确。当前使用 `grade_期望` 作为 gain，计算方式是：

```text
DCG@10 = sum((2^gain_i - 1) / log2(i + 1))
NDCG@10 = DCG@10 / IDCG@10
```

它适合做法律检索主指标，因为它不会把所有正例等同处理。一个 `grade=3` 的高度类案排到第 1，比一个边界 `grade=2` 排到第 1 更有价值。

### Recall@20 / @50 / @100

Recall 关注正例覆盖。比如：

```text
Recall@20 = Top20 中正例数 / qrels 中正例总数
```

这可以回答“系统有没有把相关案件找进候选池”。如果 `Recall@100` 高而 `Recall@20` 低，说明召回池中有正例，但排序还不够好。如果 `Recall@100` 也低，说明召回阶段本身还有缺口。

### MRR

MRR 只看第一个正例的位置：

```text
MRR = 1 / first_positive_rank
```

它适合衡量用户是否能很快看到一个有用案例。对于法律检索工作台，MRR 高意味着用户第一屏就有抓手。

### MAP

MAP 看所有正例在 ranking 中的整体分布。相比 MRR，它不只奖励第一个正例，而是奖励多个正例都排得靠前。

### Hit@5 / Hit@10

Hit 指标非常直观：前 5 或前 10 有没有至少一个正例。它不能反映细粒度排序质量，但适合作为产品可用性的粗信号。

## 7. 单 Query 诊断字段

除了主指标，系统还记录一组诊断字段：

| 字段 | 含义 |
|---|---|
| `positive_count` | qrels 中该 query 的正例数量。 |
| `first_positive_rank` | 第一个正例出现的名次。 |
| `top20_has_positive` | Top20 是否命中正例。 |
| `top100_has_positive` | Top100 是否命中正例。 |
| `weak_top20_count` | Top20 中 `grade=1` 弱相关案件数量。 |
| `failure_type` | 失败类型归因。 |
| `missed_positive_doc_ids` | Top100 未召回的正例列表。 |

`failure_type` 当前主要分为：

| 类型 | 含义 |
|---|---|
| `no_positive` | qrels 中没有正例，这类 query 的二值指标不可定义。 |
| `recall_failure` | Top100 没有正例，召回阶段失败。 |
| `ranking_failure` | Top100 有正例但 Top20 没有，排序阶段失败。 |
| `hit_top20` | Top20 命中正例。 |

这组字段比总分更适合指导工程优化。例如，`weak_top20_count` 高说明系统容易被弱相关干扰项吸走；`recall_failure` 多说明 route 或索引表达有问题；`ranking_failure` 多说明 rerank passage、融合权重或 case 聚合策略更值得看。

## 8. Hybrid 与 Rerank 对比

当前 benchmark 支持同时运行：

```text
hybrid
hybrid_rerank
```

主报告分别展示两套完整指标。除此之外，系统还会做 paired-only delta 诊断：只比较 hybrid 和 rerank 都成功返回的 query。

诊断字段包括：

- `delta_ndcg@10`
- `delta_mrr`
- `delta_map`
- `delta_recall@20`
- `hybrid_first_positive_rank`
- `rerank_first_positive_rank`
- `hybrid_weak_top20_count`
- `rerank_weak_top20_count`

这个 paired-only 分析不是主指标，而是用来回答“rerank 改善了哪些 query，又伤害了哪些 query”。它尤其适合发现 reranker 把 `grade=1` 弱相关案件推到前排的问题。

## 9. 结果如何解释

建议按下面顺序读评估结果：

1. 先看 `NDCG@10`，判断第一屏排序质量。
2. 再看 `Recall@20`，判断前 20 是否覆盖足够多正例。
3. 对照 `Recall@100`，区分召回失败还是排序失败。
4. 看 `MRR`，判断用户最快多久能看到一个强相关案例。
5. 看 `MAP`，判断多个正例整体是否靠前。
6. 看 `weak_top20_count`，判断弱相关干扰是否严重。
7. 看 per-query 明细，找出具体掉分来源。

典型解释方式：

```text
Recall@100 高，Recall@20 低：
候选池里有正例，但排序没有把它们推上来。优先优化 case 聚合、rerank passage、rerank 权重和 rank-safe。

Recall@100 低：
正例没有进入候选池。优先优化 chunk 切分、query route、BM25 字段权重、向量表达或 query expansion。

NDCG@10 升，Recall@20 降：
rerank 可能把少数高置信候选推前，但牺牲了正例覆盖。需要看弱相关干扰项和 rank promotion。

MRR 高，MAP 低：
系统能找到一个好案例，但后续正例排序分散。适合增强多正例覆盖和 case-level aggregation。
```

## 10. 当前设计合理性

当前评估设计整体是合理的，理由如下。

第一，它评估的是案件级 ranking，和最终用户看到的对象一致。虽然底层检索以 chunk 为单位，但 benchmark 判断的是 `doc_id`，这符合类案检索产品形态。

第二，它使用分级相关性而不是简单二值相关。法律类案有明显强弱层级，`grade=3`、`grade=2`、`grade=1` 的边界对排序质量有实质意义。

第三，它同时保留 NDCG、Recall、MRR、MAP 和 Hit 指标。单一指标容易误导，多指标组合能区分“召回不到”“排不上来”“第一条很好但整体弱”等不同问题。

第四，它直接跑真实链路，能覆盖工程上的真实失败因素，例如 OpenSearch、embedding、rerank 限流、case 聚合、参数默认值等。

第五，它保留 per-query 明细和 paired-only 诊断，能把“总分变化”拆解到具体 query 和具体失败类型。

但也要诚实看待它的局限：

- 数据集范围较窄，目前只覆盖上海、买卖合同纠纷、2022 年。
- 标注主要来自大模型判官集成，不是完整人工金标。
- `grade=2` 是最容易有边界争议的等级。
- query 数量只有 58 条，0.01 级别的变化可能只是噪声。
- 当前没有 bootstrap 置信区间，统计显著性不足。
- benchmark 运行依赖外部 API 和在线服务，失败率、限流、延迟会影响可复现性。

因此当前评估适合内部调参、回归测试、模块对比和研究型 benchmark，不适合直接宣称系统具有广泛泛化能力。

## 11. 当前主要风险

### 11.1 弱相关干扰

当前数据集中 `grade=1` 很多，且很多弱相关案件在表面词汇、合同类型、诉请结构上和 query 很像。系统如果过度依赖文本相似度，容易把弱相关案件排到 Top20。

应重点观察：

- `weak_top20_count`
- rerank 前后 `weak_top20_count` 是否上升
- 被推前的 `grade=1` 是否与 query 在关键裁判规则上冲突

### 11.2 Rerank 过强

Rerank 可以提高 NDCG 和 MRR，但如果权重过高，可能破坏 hybrid 原本较稳的召回覆盖。当前比较稳妥的方向是让 rerank 做局部精排，而不是完全重写原 ranking。

当前系统已有 `rerank_model_weight` 和 `rerank_max_rank_promotion`，它们应该作为核心调参项。

### 11.3 Query 数量偏小

58 条 query 足以发现明显趋势，但不足以支撑很细的结论。比如某个参数让 `NDCG@10` 提升 0.005，不应该直接认为它更好。

### 11.4 运行环境波动

Benchmark 会真实调用 embedding / rerank API。网络失败、限流、重试、模型服务版本变化都会影响评估。后续需要把运行配置、失败 query、耗时、服务版本写入报告。

## 12. 后续改进方向

优先级建议如下。

1. 增加 bootstrap 置信区间。对 query 做重采样，给 `NDCG@10`、`Recall@20`、`MRR`、`MAP` 输出 95% CI，避免把小波动当提升。

2. 增加 per-query delta 报表。把每次实验和 baseline 做逐 query 对比，输出提升最大、下降最大、弱相关上升最大、首正例后移最大等列表。

3. 增加失败类型聚合。按 `recall_failure`、`ranking_failure`、`weak_interference`、`rerank_regression` 分桶统计。

4. 增加 qrels 不确定度分析。对 `grade_std` 高的 query / doc 对单独标注，避免把标注边界样本当作硬结论。

5. 增加成本和延迟指标。离线分数不能单独决定上线策略，必须同时看平均耗时、P95、rerank 调用次数、失败率和 API 成本。

6. 增加固定 run 文件评估。除了在线真实链路，也应该支持把 ranking 保存为 run 文件，保证同一结果可重复评估。

7. 扩展数据域。当前结论不能外推到其他地区、案由和年份，后续应引入租赁、借贷、服务合同、建设工程等更多案由。

8. 建立人工复核闭环。优先复核 `grade_std` 高、rerank 大幅改变、系统和 qrels 强冲突的样本。

9. 增加 hidden set。把调参集和最终报告集分开，避免对 58 条 query 过拟合。

10. 将 anchor 作为诊断维度而非主口径。主指标继续完整 ranking；可额外统计来源案件命中率、来源案件名次、来源案件被 rerank 推动情况。

## 13. 运行入口

启动 Web 工作台：

```powershell
python legal_rag_web.py
```

访问：

```text
http://127.0.0.1:7860/
```

页面底部 benchmark 区会调用：

```text
POST /api/benchmark/evaluate
```

常用参数包括：

| 参数 | 含义 |
|---|---|
| `limit` | 评估 query 数量，最多 58。 |
| `top_k` | 评估返回 ranking 深度，默认 100。 |
| `candidate_size` | 每路召回候选数。 |
| `rerank_top_n` | rerank 的候选案件数。 |
| `rerank_model_weight` | rerank 分数融合权重。 |
| `rerank_rank_safe` | 是否限制 rerank 最大上升名次。 |
| `rerank_max_rank_promotion` | 最大上升名次。 |
| `rerank_min_interval_ms` | rerank 限流间隔。 |
| `rerank_max_retries` | rerank 重试次数。 |

离线网格搜索入口：

```powershell
python scripts\grid_search_hybrid_weights.py
```

常见输出：

```text
benchmark_runs/latest_rerank_eval_details.json
benchmark_runs/hybrid_grid_*/results.csv
benchmark_runs/hybrid_grid_*/best.json
benchmark_runs/hybrid_grid_*/errors.jsonl
```

## 14. 一句话总结

当前评估系统的核心思想是：

```text
用 CaseLaw-Bench 的分级 qrels，对真实在线检索链路输出的案件级完整 ranking 做宏平均评估，
并通过 per-query 诊断把总分拆解为召回、排序、弱相关干扰和 rerank 影响。
```

它已经适合作为项目内部的主回归标准。下一阶段最重要的不是继续堆更多总分，而是补足统计置信、失败归因、人工复核和跨域数据。
