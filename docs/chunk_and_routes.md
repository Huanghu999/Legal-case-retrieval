# Chunk 结构 & 召回 Route 说明

## 1. Chunk 索引字段（OpenSearch 实际存储）

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunk_id` | str | chunk 唯一标识 |
| `doc_id` | str | 所属案例 ID |
| `chunk_text` | str | chunk 正文内容 |
| `embedding_text` | str | 用于向量化的文本（带案由/审级/法院/法律关系等前缀） |
| `section_type` | str | 段落类型（见下方权重表） |
| `section_title` | str | 段落标题 |
| `section_index` | int | 段落在原文中的序号 |
| `chunk_index_in_case` | int | chunk 在整个案例中的序号 |
| `chunk_index_in_section` | int | chunk 在所属段落内的序号 |
| `char_start` / `char_end` | int | 在原文中的字符偏移 |
| `line_start` / `line_end` | int | 在原文中的行号范围 |
| `prev_chunk_id` | str | 前一个 chunk 的 ID（用于上下文拼接） |
| `next_chunk_id` | str | 后一个 chunk 的 ID |
| `chunk_char_len` | int | chunk 字符长度 |
| `chunk_hash` | str | chunk 内容哈希 |
| `full_text_hash` | str | 全文哈希 |
| `case_name` | str | 案件名称 |
| `case_code` | str | 案号 |
| `court_name` | str | 法院名称 |
| `court_region` | str | 法院地区 |
| `reason` | str | 案由 |
| `trial_level` | str | 审级（一审/二审等） |
| `judge_date` | str | 裁判日期 |
| `publish_date` | str | 发布日期 |
| `case_type` | str | 案件类型 |
| `statutes` | list[str] | 关联法条 |
| `section_weight` | float | 段落类型权重 |
| `embedding_model` | str | 使用的 embedding 模型名 |
| `embedding_version` | str | embedding 版本标识 |
| `embedding_dim` | int | 向量维度 |
| `embedding` | list[float] | 向量 |
| `quality_flags` | list[str] | 数据质量标记 |
| `schema_version` | str | 数据 schema 版本 |

### section_type 权重表

| section_type | 权重 | 含义 | 数据来源 |
|--------------|------|------|----------|
| `fine_tags` | 1.60 | 细争点标签 | 主叶子 + 细争点列表（纯分类标签） |
| `fine_rule` | 1.55 | 裁判规则 | 裁判规则争点（一句话规则描述） |
| `focus_tags` | 1.50 | 焦点标签 | 焦点标签列表（粗粒度争点分类） |
| `focus_analysis` | 1.45 | 焦点评析 | 案情核心 + 法律争点 + 裁判要旨 + 焦点原文 |
| `reasoning` | 1.20 | 裁判说理 | 段落.本院认为 |
| `facts` | 1.00 | 案件事实 | 段落.查明事实 |
| `defense` | 0.85 | 抗辩理由 | 段落.诉称中被告辩称部分 |
| `judgment` | 0.75 | 裁判结果 | 段落.裁判结果 |
| `claims` | 0.70 | 诉讼请求 | 段落.诉称中原告主张部分 |
| `statutes` | 0.45 | 法条引用 | 引用法条 |

> **设计原则**：标签归标签、自然语言归自然语言。标签类 chunk（fine_tags/focus_tags）用于精确匹配和同类案件聚合，自然语言类 chunk（fine_rule/focus_analysis）用于语义检索，不互相稀释。

---

## 2. 召回 Route 列表

**查询来源说明**：

| 查询来源 | 构成方式 |
|----------|----------|
| 原始 query | 用户输入原文 |
| 扩展 query | `争议焦点 + 请求类型 + 关键事实 + 法律关系 + 案由`，或 LLM 改写的 expanded_query |
| 法律关系 query | `案由 + 法律关系 + 法条 + 请求类型`，或 LLM 改写的 `legal_issue + main_leaf + focus_labels` |
| 否定 query | `否定事实 + 请求类型 + 法律关系 + 案由` |
| 法条 query | `legal_issue + 法条 + main_leaf`，或规则提取的 `案由 + 法律关系 + 法条` |

| Route 名称 | 检索方式 | 查询来源 | 权重 | 过滤 section_type | 说明 |
|------------|----------|----------|------|-------------------|------|
| `bm25_raw` | BM25 | 原始 query | 1.0 | - | 原始关键词检索 |
| `vector_raw` | 向量 | 原始 query | 0.8 | - | 原始语义检索 |
| `bm25_focus` | BM25 | 扩展 query | 0.95 | - | 焦点扩展关键词检索 |
| `vector_focus` | 向量 | 扩展 query | 1.20 | - | 焦点扩展语义检索（权重最高） |
| `bm25_fine_tags` | BM25 | 法律关系 query | 1.20 | `fine_tags` | 仅搜细争点标签（精确匹配主叶子） |
| `bm25_fine_rule` | BM25 | 法律关系 query | 1.30 | `fine_rule` | 仅搜裁判规则描述 |
| `bm25_focus_tags` | BM25 | 法律关系 query | 1.60 | `focus_tags` | 仅搜焦点标签（route 权重最高） |
| `bm25_focus_analysis` | BM25 | 法律关系 query | 1.10 | `focus_analysis` | 仅搜焦点评析（案情核心+法律争点+裁判要旨） |
| `bm25_reasoning` | BM25 | 法律关系 query | 1.10 | `reasoning` | 仅搜裁判说理段落 |
| `bm25_facts` | BM25 | 法律关系 query | 0.60 | `facts` | 仅搜案件事实段落 |
| `bm25_negative` | BM25 | 否定 query | 1.20 | - | 针对"未交货""未开票"等否定场景 |
| `bm25_legal` | BM25 | 法条 query | 0.80 | - | 法条关键词检索 |

> **融合方式**：各 route 结果通过 RRF（k=10）融合，再叠加 section_type 权重、多源命中 bonus、关键段落 bonus 等得到最终 case_score。
