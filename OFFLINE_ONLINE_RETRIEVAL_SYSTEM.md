# 离线解析与在线召回系统深度说明

本文档说明当前 Legal Case RAG 项目的完整工程链路：离线如何把 CaseLaw-Bench 结构化案件解析成可检索索引，在线如何把用户 query 转成多路召回、案件级聚合、精排和可回溯结果。

当前项目不是一个单纯向量检索 demo，而是一套面向法律类案任务的混合检索系统。它的核心思想是：

```text
结构化案件解析
+ 法律要素化 chunk
+ BM25 / 向量多路召回
+ query profile 路由
+ 案件级聚合
+ case-level rerank
+ 原文证据回溯
+ benchmark 闭环评估
```

## 1. 总体架构

系统分为两条链路：

```text
离线链路：
corpus.jsonl
  -> benchmark_dataset_builder.py
  -> cases.jsonl / chunks.jsonl / mappings
  -> embed_chunks.py
  -> chunks_embedded.jsonl
  -> opensearch_ingest.py
  -> OpenSearch case index + chunk index

在线链路：
user query
  -> query_profile.py
  -> BM25 / vector / focus / legal / negative routes
  -> OpenSearch chunk hits
  -> case-level RRF aggregation
  -> attach key chunks
  -> case-level rerank
  -> fetch case docs
  -> API / Web UI / benchmark
```

离线链路负责把案件变成“可检索、可解释、可评估”的结构化索引。在线链路负责把自然语言 query 变成一组候选案件，并给出排序、证据片段、来源 route 和原文回溯能力。

## 2. 数据输入

当前数据源固定为：

```text
data/caselaw-benchmark release/data/corpus.jsonl
```

每行是一篇结构化案件，主要字段包括：

- `doc_id`
- `案号`
- `法院`
- `审级`
- `案由`
- `裁判日期`
- `当事人`
- `法律关系`
- `标的物类型`
- `争议焦点`
- `细争点`
- `裁判结果_标签`
- `引用法条`
- `段落`

其中 `段落` 通常包含：

- `诉称`
- `查明事实`
- `本院认为`
- `裁判结果`

这批数据已经在 CaseLaw-Bench 侧完成了规则抽取和大模型结构化抽取。本项目离线链路的职责不是重新理解原始裁判文书，而是把这些结构化字段重新组织成适合检索的 case 文档和 chunk 文档。

## 3. 离线构建入口

一键重建入口是：

```powershell
python scripts\rebuild_benchmark_rag_index.py
```

底层依次调用三类能力：

1. `benchmark_dataset_builder.py`：构建案件 JSONL、chunk JSONL 和 OpenSearch mapping。
2. `embed_chunks.py`：为 chunk 生成 embedding。
3. `opensearch_ingest.py`：创建索引并写入 case / chunk 文档。

也可以分步执行，便于调试。

## 4. 案件级文档设计

案件级文档由 [benchmark_dataset_builder.py](d:/2405data/src/legal_case_rag/data_pipeline/benchmark_dataset_builder.py) 中的 `build_case_doc()` 生成。

它的目标是保存“案件本体”，用于：

- 检索结果展示。
- 根据 `doc_id` 回查原文。
- drawer 中展示案由、法院、审级、裁判日期、当事人、法条。
- benchmark 中把 chunk 级命中回到 case 级评估。

主要字段包括：

| 字段 | 含义 |
|---|---|
| `doc_id` | 案件唯一 ID。 |
| `case_code` | 案号或从 `doc_id` 派生的案件编号。 |
| `case_name` | 当前主要使用案号作为展示名。 |
| `court_name` | 法院。 |
| `court_region` | 法院区域，目前简单识别上海。 |
| `case_type` | 案件类型，当前为民事。 |
| `trial_level` | 审级。 |
| `reason` | 案由。 |
| `judge_date` | 裁判日期。 |
| `litigants` | 当事人列表。 |
| `statutes` | 引用法条。 |
| `full_text` | 项目重组后的全文文本。 |
| `full_text_hash` | 全文 hash。 |
| `source_file` | 来源文件。 |
| `source_row_id` | 来源行号。 |
| `schema_version` | 数据结构版本。 |

`full_text` 不是原始裁判文书逐字全文，而是由结构化字段重组后的“检索与回溯全文”。它按案号、法院、案由、法律关系、争议焦点、细争点、诉称、事实、本院认为、裁判结果、法条等顺序拼接。

这个设计的好处是展示和回溯时内容更干净，法律要素更集中；代价是它不是严格意义上的原文版式复刻。

## 5. Chunk 设计

系统不是直接把整篇案件作为一个检索单元，而是按法律结构拆成多个 section，再按长度切 chunk。

当前 section 包括：

| section_type | section_title | 用途 |
|---|---|---|
| `case_profile` | 案件画像 | 案由、审级、法院、法律关系、标的物、裁判结果标签。 |
| `fine_issue` | 细争点 | 主叶子、细争点、裁判规则争点。 |
| `focus` | 争议焦点 | 焦点标签、案情核心、法律争点、裁判要旨、焦点原文。 |
| `claims` | 诉称 | 当事人诉讼请求和主张。 |
| `facts` | 查明事实 | 法院查明的事实。 |
| `reasoning` | 本院认为 | 裁判理由和规则适用。 |
| `judgment` | 裁判结果 | 判项。 |
| `statutes` | 引用法条 | 法条依据。 |

这个切分方式体现了法律类案检索的核心判断：类案相似不只看事实文本，还要重点看争议焦点和裁判规则。因此 `fine_issue`、`focus`、`reasoning` 的权重更高。

当前 section 权重：

| section_type | weight |
|---|---:|
| `fine_issue` | 1.55 |
| `focus` | 1.45 |
| `case_profile` | 1.20 |
| `reasoning` | 1.15 |
| `facts` | 1.00 |
| `claims` | 0.75 |
| `judgment` | 0.70 |
| `statutes` | 0.45 |

## 6. Chunk 切分策略

`build_chunks()` 会遍历每个 section。对长文本 section 采用较短窗口和 overlap：

```text
facts / reasoning / claims:
  max_chars = 900
  overlap = 120

其他 section:
  max_chars = 1600
  overlap = 0
```

原因是事实、说理、诉称往往比较长，且关键信息可能跨段落；适度 overlap 可以降低切分边界损失。结构化短 section 如 `fine_issue`、`focus` 通常信息密度高，不需要 overlap。

每个 chunk 生成以下核心字段：

| 字段 | 含义 |
|---|---|
| `chunk_id` | `doc_id#section_type#section_index#chunk_index`。 |
| `doc_id` | 所属案件。 |
| `chunk_text` | 用于展示和 BM25 的 chunk 正文。 |
| `embedding_text` | 用于 embedding 的增强文本。 |
| `section_type` | 章节类型。 |
| `section_title` | 章节中文名。 |
| `section_index` | section 在案件内的顺序。 |
| `chunk_index_in_case` | chunk 在案件内的全局顺序。 |
| `chunk_hash` | chunk 文本 hash。 |
| `case_name` / `reason` / `court_name` 等 | 冗余案件元数据，便于 chunk 命中直接展示。 |
| `section_weight` | section 权重。 |
| `embedding_model` / `embedding_dim` | 向量元信息。 |

`embedding_text` 会在 `chunk_text` 外增加案由、审级、法院、章节名：

```text
案由：...
审级：...
法院：...
章节：...
正文：...
```

这样做是为了让向量表达不只看到局部片段，还能带上案件上下文和 section 语义。

## 7. OpenSearch 索引设计

系统使用两个索引：

```text
caselaw_benchmark_cases_v1
caselaw_benchmark_chunks_v1
```

### Case Index

case index 保存案件级文档。主要服务于：

- `doc_id` 回查。
- 检索结果展示。
- 原文 drawer。
- 案件元数据读取。

mapping 特点：

- `doc_id`、`case_code`、`court_name`、`trial_level` 等为 `keyword`。
- `case_name` 同时有 `text` 和 `keyword` 子字段。
- `reason` 同时有 `keyword` 和 `text` 子字段。
- `full_text` 为 `text`。
- `litigants` 为 disabled object，避免复杂结构影响 mapping。

### Chunk Index

chunk index 保存检索单元。主要服务于：

- BM25 检索。
- kNN 向量检索。
- section 过滤。
- route 聚合。
- rerank passage 构造。

mapping 特点：

- `dynamic` 为 `strict`，防止脏字段进入索引。
- `embedding` 是 `knn_vector`，维度 1024。
- HNSW 配置为 `lucene` engine、`cosinesimil`。
- `chunk_text` 和 `embedding_text` 是 `text`。
- `section_type`、`section_title`、`doc_id` 等是 `keyword`。
- `section_weight` 是 `float`。

## 8. Embedding 生成

向量生成脚本是 [embed_chunks.py](d:/2405data/src/legal_case_rag/data_pipeline/embed_chunks.py)。

默认配置：

```text
API base: https://api.siliconflow.cn/v1
model: BAAI/bge-m3
api key env: SILICONFLOW_API_KEY
batch size: 32
```

脚本读取：

```text
benchmark_dataset/caselaw_benchmark_chunks_v1.jsonl
```

输出：

```text
benchmark_dataset/caselaw_benchmark_chunks_v1_embedded.jsonl
```

关键能力：

- 支持 `--resume`，会跳过已写入输出文件的 `chunk_id`。
- 支持 `--overwrite`。
- 支持 `--dry-run` 检查 payload。
- 每批失败会指数退避重试。
- 写入时补充 `embedding`、`embedding_model`、`embedding_dim`。

这里有一个工程上很重要的设计：embedding 生成是离线固化的，在线 query 只需要生成 query embedding，不需要为 corpus 重算向量。

## 9. OpenSearch 入库

入库脚本是 [opensearch_ingest.py](d:/2405data/src/legal_case_rag/data_pipeline/opensearch_ingest.py)。

默认输入：

```text
benchmark_dataset/caselaw_benchmark_cases_v1.jsonl
benchmark_dataset/caselaw_benchmark_chunks_v1_embedded.jsonl
```

如果 embedded chunk 文件不存在，会 fallback 到：

```text
benchmark_dataset/caselaw_benchmark_chunks_v1.jsonl
```

注意：如果没有 embedding 字段，BM25 可以用，但 vector 检索不可用或会失败。

入库流程：

1. 读取 mapping 文件。
2. 检查 case / chunk JSONL 数量。
3. 如果指定 `--delete-existing`，先删除旧索引。
4. 如果未指定 `--skip-create`，创建索引。
5. 使用 `_bulk` 批量写入 case 文档。
6. 使用 `_bulk` 批量写入 chunk 文档。

默认认证：

```text
host: https://localhost:9200
username: admin
password env: OPENSEARCH_PASSWORD
```

## 10. 在线 API 入口

Web 服务入口是 [legal_rag_web.py](d:/2405data/legal_rag_web.py)。

主要接口：

| 接口 | 作用 |
|---|---|
| `GET /` | 类案检索评估台页面。 |
| `GET /api/health` | 检查 OpenSearch 密码、SiliconFlow key、默认参数。 |
| `POST /api/search` | 单条 query 检索。 |
| `POST /api/benchmark/evaluate` | 跑 CaseLaw-Bench benchmark。 |
| `GET /api/cases/<doc_id>` | 回查单篇案件全文。 |

前端文件：

```text
templates/legal_rag_index.html
static/legal_rag.js
static/legal_rag.css
```

参数构造由 [search_args.py](d:/2405data/src/legal_case_rag/app/search_args.py) 统一处理。这样 Web 搜索和 benchmark 使用同一套默认值、环境变量和参数解析逻辑。

## 11. Query Profile

在线检索首先会调用 [query_profile.py](d:/2405data/src/legal_case_rag/retrieval/query_profile.py) 对用户 query 做轻量结构化。

它抽取：

- 案由词：如买卖合同纠纷、委托合同纠纷。
- 请求类型：如返还、解除、违约金、货款、发票、押金。
- 争议焦点：如是否、能否、如何认定、举证责任、表见代理。
- 否定事实：如未付款、未交付、不能证明、不予支持。
- 法律关系：如买卖合同、租赁合同、民间借贷、保证责任。
- 引用法条：通过正则抽取《法律名》第X条。
- 预期裁判倾向：支持、不支持、部分支持。

Query Profile 的目标不是做复杂 NLP，而是用低成本规则把法律检索中最关键的信号显式化。它不会读取 qrels，也不会使用 `query_source_doc`、难度、陷阱等评估字段。

## 12. 多路召回 Route

`build_query_routes()` 会把一个 query 扩展成多条召回 route。典型 route 包括：

| route | 类型 | 作用 |
|---|---|---|
| `bm25_raw` | BM25 | 原始 query 关键词匹配。 |
| `vector_raw` | vector | 原始 query 语义匹配。 |
| `bm25_focus` | BM25 | 争议焦点、请求类型、关键事实增强。 |
| `vector_focus` | vector | focus query 的语义召回。 |
| `bm25_fine_issue` | BM25 + section filter | 只搜 `fine_issue`。 |
| `bm25_focus_section` | BM25 + section filter | 只搜 `focus`。 |
| `bm25_reasoning` | BM25 + section filter | 只搜 `reasoning`。 |
| `bm25_facts` | BM25 + section filter | 只搜 `facts`。 |
| `bm25_negative` | BM25 | 强化否定事实匹配。 |
| `bm25_legal` | BM25 | 强化案由、法律关系、法条匹配。 |

这套 route 设计解决的是单路检索的盲点：

- BM25 擅长精确词和法律术语。
- 向量擅长语义泛化。
- focus route 强调争议焦点。
- negative route 强调“不成立、未交付、未付款、证据不足”等否定事实。
- section route 强迫系统去看最有判别力的章节。

## 13. BM25 检索

BM25 查询由 `bm25_query_body()` 构造。

当前字段权重：

```text
chunk_text^3.0
embedding_text^2.0
section_title^1.4
reason^1.8
case_name^1.2
```

检索结构是：

```text
bool:
  must:
    multi_match(query, fields)
  filter:
    reason / trial_level / court_name / section_type / judge_date
```

也就是说，用户筛选条件以 filter 方式进入，不参与打分；query 文本通过 multi_match 参与 BM25 打分。

## 14. 向量检索

向量检索由 `search_vector()` 执行：

1. 调用 SiliconFlow embedding API，把 query 转成向量。
2. 在 chunk index 的 `embedding` 字段上执行 kNN。
3. 如果有筛选条件，将 filter 放入 kNN 查询。
4. 返回 chunk hit。

默认 query embedding 模型与离线 chunk embedding 模型一致：

```text
BAAI/bge-m3
```

这点很重要。若离线和在线模型不一致，向量空间会不一致，召回质量会显著下降。

## 15. Query Profile Bonus

每条 route 返回 chunk hit 后，如果启用 `query_profile_boost`，系统会调用 `profile_match_bonus()` 给 chunk 加小幅加分。

主要加分因素：

- chunk 位于 `reasoning`、`facts` 等关键 section。
- 案由与 query 的核心案由一致。
- 法律关系命中。
- 请求类型命中。
- query 有否定事实，chunk 也出现否定事实标签。
- 法条命中。

bonus 有上下限：

```text
min = -0.05
max = 0.28
```

这不是主排序模型，而是一层轻量法律特征校正。它的作用是让同等 BM25 / 向量得分下，更贴合法律要素的片段靠前。

## 16. 案件级聚合

OpenSearch 返回的是 chunk，但用户和 benchmark 关心的是案件。因此系统必须把 chunk hit 聚合成 case hit。

当前主聚合函数是 `case_level_reciprocal_rank_fusion()`。

流程如下：

1. 每条 route 先将 chunk hit 按 `doc_id` 分组。
2. 每个案件内部计算 route-level case score。
3. 每条 route 得到一个案件 ranking。
4. 对不同 route 的案件 ranking 做 Reciprocal Rank Fusion。
5. 叠加案件级结构 bonus。
6. 得到最终 hybrid case ranking。

RRF 公式是：

```text
rrf_score = route_weight / (k + rank)
```

当前 `k=60`。RRF 的好处是对不同检索通道的原始分数不敏感。BM25 分数、向量分数和 profile bonus 尺度不同，直接相加不稳；RRF 只使用名次，更适合多路融合。

## 17. 案件级结构加分

case 聚合时还会加入多类结构信号：

| 信号 | 作用 |
|---|---|
| key section bonus | 命中 `fine_issue`、`focus`、`reasoning`、`facts` 越多越好。 |
| fine/focus bonus | 命中最核心争点章节额外加分。 |
| dual channel bonus | 同时被 BM25 和 vector 命中，说明词面和语义都支持。 |
| route coverage bonus | 被多个 route 命中，说明证据来源更稳。 |
| multi chunk bonus | 同案多个 chunk 命中，说明不是孤立偶然命中。 |

这个设计的直觉是：一个真正相关案件往往不会只在一个片段、一个通道、一个 route 中偶然出现，而会在争点、事实、说理等多个结构位置形成一致信号。

## 18. Case-Level Rerank

当前 rerank 是案件级 rerank，不再直接对 chunk 做最终排序。

流程：

1. Hybrid 先得到案件级候选。
2. `attach_case_key_chunks()` 为候选案件补充关键 section chunk。
3. `select_case_rerank_chunks()` 选择最适合送入 reranker 的 chunk。
4. `build_case_rerank_passage()` 把一个案件组织成 passage。
5. 调用 BGE reranker。
6. 归一化 hybrid 分数和 rerank 分数。
7. 叠加结构调整和 guardrail 调整。
8. 根据 `rerank_model_weight` 融合。
9. 如果开启 rank-safe，限制候选最大上升名次。

默认 reranker：

```text
BAAI/bge-reranker-v2-m3
```

默认融合逻辑：

```text
final_score = hybrid_norm * (1 - rerank_model_weight) + rerank_norm * rerank_model_weight
```

当前默认 `rerank_model_weight` 为 0.25。

## 19. Rerank Passage 设计

`build_case_rerank_passage()` 不会简单拼所有 chunk，而是按法律重要性选择和组织内容。

优先 section：

```text
fine_issue
focus
reasoning
facts
case_profile
claims
judgment
```

每类 section 有预算，例如：

| section | budget |
|---|---:|
| `fine_issue` | 700 |
| `focus` | 700 |
| `reasoning` | 900 |
| `case_profile` | 420 |
| `facts` | 320 |
| `judgment` | 240 |
| `claims` | 220 |

这样做是为了让 reranker 读到“决定裁判规则的关键信息”，而不是被冗长事实或诉称噪声淹没。

## 20. Guardrail 与 Rank-Safe

Rerank 有两个保护机制。

### Guardrail

`rerank_guardrail_adjustment()` 会检查 query 中是否存在一些必须匹配的法律因素，例如：

- 所有权保留。
- 定金罚则。
- 发票争议。
- 第三方供货。
- 对账单沉默。
- 偷盖章 / 冒盖章。
- 口头合同证据。
- 解除时间。
- 委托付款或第三方付款。

如果 query 明确要求某个因素，但候选案件 passage 中缺失，系统会施加 penalty。也会检查一些事实冲突，例如 query 关注“质量瑕疵”，候选却主要是“未交货”。

### Rank-Safe

`apply_rank_safe_rerank()` 限制 rerank 后候选最多上升多少名。默认：

```text
DEFAULT_RERANK_RANK_SAFE = True
DEFAULT_RERANK_MAX_RANK_PROMOTION = 20
```

这能防止 reranker 把原本很靠后的弱相关案件一下推到前排。对于当前数据集，rank-safe 很重要，因为弱相关干扰项多，reranker 容易被表面语义相似误导。

## 21. 结果组装与原文回溯

最终 `run_search()` 会获取 TopK 案件的 case doc，并调用 `build_result_entry()` 组装结果。

返回内容包括：

- query 和 mode。
- rerank 配置。
- filters。
- query_profile。
- query_routes。
- results。

每个 result 包含：

- `doc_id`
- `case_score`
- `case_name`
- `reason`
- `trial_level`
- `court_name`
- `judge_date`
- `matched_chunks`
- `case_doc`
- rerank 相关分数和调整项。

如果 `show_context=true`，系统会基于 chunk 的 `char_start` / `char_end` 和 case `full_text` 构造上下文，并用：

```text
【命中】...【/命中】
```

标记命中片段。前端再渲染为高亮。

当前需要注意一个限制：离线 chunk 的 `char_start` / `char_end` 目前是 chunk 内部范围，不一定严格对应重组全文中的真实位置。因此上下文回溯是可用的调试能力，但还不是精确原文定位系统。后续如果要做严肃证据定位，应在离线构建阶段记录 chunk 在 `full_text` 中的真实 offset。

## 22. Benchmark 如何复用在线链路

Benchmark 不另写一套检索，而是调用同一个 `run_search()`。

这带来两个优点：

- 评估和产品行为一致。
- 参数变化能立即反映到 benchmark。

也带来两个代价：

- 评估依赖在线 OpenSearch 和外部 API。
- 评估耗时和失败率受网络、限流、服务状态影响。

因此当前 benchmark 更像“在线回归评估”，不是完全可复现的离线 run-file 评估。后续建议补充固定 run 文件模式，形成：

```text
online benchmark: 验证真实系统
run-file benchmark: 验证固定 ranking
```

## 23. 当前系统优点

第一，结构化程度高。系统不是把全文扔进向量库，而是按案件画像、细争点、焦点、事实、说理、判项、法条组织检索单元。

第二，召回路径多元。BM25、向量、focus、negative、legal、section route 共同工作，降低单一路径漏召风险。

第三，聚合层贴合法律任务。最终按案件排序，并对关键 section、多 route 命中、多 chunk 命中加权。

第四，rerank 是案件级的。它读的是为一个案件组织出的 passage，而不是孤立 chunk，更接近“判断此案是否为类案”的真实语境。

第五，评估闭环完整。系统可以从前端直接跑 CaseLaw-Bench，看到指标、query 明细、弱相关干扰和 rerank delta。

## 24. 当前系统瓶颈

### 24.1 Offset 不精确

当前 chunk 的 `char_start` / `char_end` 不是严格全文 offset。前端可展示命中上下文，但不能保证高亮位置完全等于原文位置。

改进方向：在 `build_full_text()` 后建立 section 到 full_text 的 offset 映射，chunk 切分时记录真实 offset。

### 24.2 Query Profile 是规则型

规则型 profile 可控、便宜、稳定，但覆盖有限。面对更复杂的法律表达、隐含争点、跨法域术语时可能漏抽。

改进方向：保留规则 profile 作为稳定基线，再加入可解释的 LLM query parser，并通过 benchmark A/B 验证。

### 24.3 Section 权重仍是经验值

当前 section 权重来自任务直觉和调参经验。它们合理，但还不是系统学习出来的。

改进方向：使用 grid search 或 learning-to-rank，在验证集上学习 route 权重、section 权重和聚合 bonus。

### 24.4 Rerank Passage 仍可能含噪

如果 passage 放入过多事实细节，reranker 可能更看表面事实；如果放入过少，又可能漏掉关键抗辩。

改进方向：按 query 类型动态分配 section budget。例如发票争议增加 `claims` / `reasoning`，质量争议增加 `facts`，主体责任增加 `case_profile` / `reasoning`。

### 24.5 在线评估可复现性有限

依赖外部 API 的 benchmark 会受限流和服务变化影响。

改进方向：保存每次 benchmark 的完整 ranking、route hit、rerank scores 和配置，支持离线重算指标。

## 25. 后续演进建议

优先级从高到低：

1. 修复真实全文 offset。让证据回溯从“调试高亮”升级为“可审查定位”。

2. 增加 run 文件输出。每次 benchmark 保存完整 ranking，用于复现、对比和置信区间计算。

3. 做 route / section 权重搜索。把当前经验权重变成可回归调参项。

4. 增加 per-query 失败归因页面。把 recall failure、ranking failure、weak interference、rerank regression 直接可视化。

5. 优化 rerank passage。按 query profile 动态选择 section 和 budget。

6. 引入 LLM query parser。专门抽取法律争点、否定事实、抗辩结构、请求权基础，但必须保留规则 parser 作为 fallback。

7. 建立索引版本管理。把 corpus hash、schema version、embedding model、mapping version、ingest time 写入元数据，避免评估结果和索引版本混淆。

8. 扩展数据集。将系统从买卖合同扩展到租赁、借贷、服务合同、建设工程等更多领域。

## 26. 常用命令

重建 benchmark RAG 索引：

```powershell
$env:SILICONFLOW_API_KEY="your SiliconFlow API Key"
$env:OPENSEARCH_PASSWORD="your OpenSearch admin password"
python scripts\rebuild_benchmark_rag_index.py
```

只构建 JSONL 和 mapping：

```powershell
python src\legal_case_rag\data_pipeline\benchmark_dataset_builder.py --overwrite
```

只生成 embedding：

```powershell
python src\legal_case_rag\data_pipeline\embed_chunks.py --resume
```

只入库 OpenSearch：

```powershell
python src\legal_case_rag\data_pipeline\opensearch_ingest.py --delete-existing
```

启动 Web 工作台：

```powershell
python legal_rag_web.py
```

命令行检索：

```powershell
python -m src.legal_case_rag.retrieval.search --query "买受人未支付货款，出卖人要求支付货款和逾期利息是否支持" --mode hybrid --rerank --top-k 8 --show-context
```

## 27. 一句话总结

这套系统的高度在于：它把法律类案检索拆成了“法律结构表达、混合召回、案件级证据聚合、受控精排、评估闭环”五层，而不是把问题简化成普通向量相似度。当前最值得继续投入的是精确证据定位、权重系统化学习、rerank 稳定性和评估可复现性。
