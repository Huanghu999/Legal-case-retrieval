# MUSER 多视角检索融入现有法律案例 RAG 系统 — 设计方案

## 一、论文核心思想总结

MUSER (Multi-View Prompt Retrieval) 论文的关键创新：

1. **Head-Tail Query Transformation** — 将第三人称叙述转换为第一人称（"买方与卖方…" → "我方与对方…"），使语义更贴近真实当事人视角
2. **五维法律要素分解** — 用 LLM 将用户查询拆解为 5 个法律视角：
   - 程序要件 (Procedure) — 诉讼时效、管辖、诉讼主体资格
   - 请求权基础 (Cause of Action) — 请求权的构成要件、抗辩
   - 法律依据 (Legal Basis) — 具体法条引用
   - 事实认定焦点 (Factual Focus) — 争议焦点、证据认定
   - 当事人 (Parties) — 当事人身份、角色、关系
3. **Multi-View Prompt + 22-shot** — 每个视角独立生成检索查询
4. **Ensemble Retrieval** — 多视角查询结果加权融合

## 二、现有系统架构分析

### 当前 QueryProfile（规则引擎）

现有 `query_profile.py` 使用纯关键词匹配提取法律要素：

| 字段 | 对应 MUSER 视角 | 提取方式 |
|------|----------------|---------|
| `core_reasons` | 请求权基础 (部分) | REASON_TERMS 关键词 |
| `request_types` | 请求权基础 (部分) | REQUEST_TERMS 关键词 |
| `dispute_focus` | 事实认定焦点 (部分) | DISPUTE_TERMS 关键词 |
| `key_facts` | 事实认定焦点 (部分) | REQUEST_TERMS + LEGAL_RELATION_TERMS |
| `negative_facts` | 事实认定焦点 (否定面) | NEGATIVE_TERMS 关键词 |
| `legal_relations` | 请求权基础 (部分) | LEGAL_RELATION_TERMS 关键词 |
| `statutes` | 法律依据 (部分) | 正则提取法条引用 |
| `expected_tendency` | — | 推断裁判倾向 |

### 当前 QueryRoute（检索路由）

| 路由名 | 类型 | 对应 MUSER 视角 |
|--------|------|----------------|
| `bm25_raw` / `vector_raw` | 原始查询 | 全局（相当于 head query） |
| `bm25_focus` / `vector_focus` | 焦点查询 | 事实认定焦点 |
| `bm25_fine_issue` | fine_issue 段落 | 事实认定焦点 |
| `bm25_focus_section` | focus 段落 | 事实认定焦点 |
| `bm25_reasoning` | reasoning 段落 | 法律依据 + 请求权基础 |
| `bm25_facts` | facts 段落 | 事实认定焦点 |
| `bm25_negative` | 否定事实 | 事实认定焦点（否定面） |
| `bm25_legal` | 法律查询 | 请求权基础 + 法律依据 |

### 覆盖缺口分析

| MUSER 视角 | 现有覆盖 | 缺口 |
|-----------|---------|------|
| 程序要件 (Procedure) | ❌ 无 | 完全缺失：诉讼时效、管辖权、诉讼主体资格 |
| 请求权基础 (Cause of Action) | ⚠️ 部分 | `bm25_legal` 覆盖了请求类型和法律关系，但缺乏构成要件拆解 |
| 法律依据 (Legal Basis) | ⚠️ 部分 | `statutes` 字段仅正则提取法条名，缺乏条文内容推理 |
| 事实认定焦点 (Factual Focus) | ✅ 较好 | `bm25_focus` + `bm25_negative` 覆盖了主要焦点 |
| 当事人 (Parties) | ❌ 无 | 完全缺失：当事人身份、代理关系、连带责任等 |

## 三、有机融合方案

### 核心设计原则

1. **增量增强，不破坏现有架构** — LLM 分解作为可选模块，规则引擎始终保留为 fallback
2. **复用 QueryRoute 基础设施** — 多视角查询直接映射为新的路由类型
3. **渐进式集成** — 先加最容易提升的视角，再逐步完善
4. **可评估** — 每一步都可以通过 benchmark 对比效果

### 3.1 增强 QueryProfile：加入 LLM 分解

在现有 `QueryProfile` 基础上，新增一个 LLM 增强层：

```python
@dataclass
class MultiViewProfile:
    """LLM 多视角分解结果"""
    procedure: str = ""           # 程序要件视角查询
    cause_of_action: str = ""     # 请求权基础视角查询
    legal_basis: str = ""         # 法律依据视角查询
    factual_focus: str = ""       # 事实认定焦点视角查询
    parties: str = ""             # 当事人视角查询
    head_query: str = ""          # head-tail 转换后的查询
    raw_decomposition: dict = field(default_factory=dict)  # LLM 原始输出
```

LLM Prompt 设计（利用 Excel 中的 Legal Element Label Schema）：

```
你是一个法律案例检索助手。请将以下用户查询分解为 5 个法律视角的检索查询。

用户查询：{query}

请按以下 JSON 格式输出：
{
  "head_query": "将查询转换为第一人称视角（如果查询是第三人称描述）",
  "procedure": "提取与诉讼程序相关的要素：诉讼时效、管辖权、诉讼主体资格等。如果没有明确程序要素，留空。",
  "cause_of_action": "提取请求权基础：构成要件、抗辩事由、责任类型。",
  "legal_basis": "提取相关法律法规依据，包括具体法条。",
  "factual_focus": "提取事实认定焦点：争议焦点、证据认定、关键事实。",
  "parties": "提取当事人信息：身份角色（买方/卖方、出借人/借款人等）、代理关系、连带责任关系。"
}
```

### 3.2 新增视角路由

在 `build_query_routes()` 中，为 LLM 分解的视角创建新路由：

```python
def build_muser_routes(profile: QueryProfile, mview: MultiViewProfile) -> list[QueryRoute]:
    routes = []

    # 程序要件路由（全新）
    if mview.procedure:
        routes.append(QueryRoute("bm25_procedure", mview.procedure, "bm25", 0.85))
        routes.append(QueryRoute("vector_procedure", mview.procedure, "vector", 0.70))

    # 请求权基础路由（增强现有 bm25_legal）
    if mview.cause_of_action:
        routes.append(QueryRoute("bm25_cause", mview.cause_of_action, "bm25", 1.10))
        routes.append(QueryRoute("vector_cause", mview.cause_of_action, "vector", 1.05))

    # 法律依据路由（增强现有）
    if mview.legal_basis:
        routes.append(QueryRoute("bm25_law", mview.legal_basis, "bm25", 0.90))

    # 事实认定焦点路由（与现有 bm25_focus 互补）
    if mview.factual_focus:
        routes.append(QueryRoute("bm25_factual", mview.factual_focus, "bm25", 1.15))
        routes.append(QueryRoute("vector_factual", mview.factual_focus, "vector", 1.25))

    # 当事人路由（全新）
    if mview.parties:
        routes.append(QueryRoute("bm25_parties", mview.parties, "bm25", 0.75))

    return routes
```

### 3.3 融合策略

多视角路由与现有路由共存，通过 RRF 统一融合：

```
┌─────────────────────────────────────────────────────────────┐
│                    用户查询                                   │
│                         │                                    │
│              ┌──────────┴──────────┐                        │
│              ▼                     ▼                         │
│     规则引擎 QueryProfile    LLM 多视角分解                   │
│              │                     │                         │
│              ▼                     ▼                         │
│     现有路由 (8-10条)       MUSER 路由 (5-10条)              │
│     bm25_raw               bm25_procedure                   │
│     vector_raw             vector_procedure                  │
│     bm25_focus             bm25_cause                       │
│     vector_focus           vector_cause                      │
│     bm25_fine_issue        bm25_law                         │
│     bm25_focus_section     bm25_factual                     │
│     bm25_reasoning         vector_factual                   │
│     bm25_facts             bm25_parties                     │
│     bm25_negative                                           │
│     bm25_legal                                              │
│              │                     │                         │
│              └──────────┬──────────┘                        │
│                         ▼                                    │
│              Case-Level RRF Fusion                           │
│                         │                                    │
│                         ▼                                    │
│              Rerank + Guardrails                              │
│                         │                                    │
│                         ▼                                    │
│                    最终结果                                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Head-Tail 转换

MUSER 的 Head-Tail 转换非常有价值。实现方式：

```python
def head_tail_transform(query: str, llm_client) -> str:
    """将第三人称查询转换为第一人称视角"""
    prompt = f"""请将以下法律问题从第三人称视角转换为第一人称视角（假设你是当事人的律师）。
保持法律要素不变，只改变人称。

第三人称：{query}
第一人称："""
    return llm_client.complete(prompt)
```

这个转换后的查询可以替代现有的 `bm25_raw` / `vector_raw`，或者作为额外的一路召回。

### 3.5 路由权重策略

| 路由类别 | 推荐权重 | 理由 |
|---------|---------|------|
| 原始查询 (raw) | 1.0 | 基准 |
| 事实焦点 (focus/factual) | 1.1-1.25 | 论文证实最有效的视角 |
| 请求权基础 (cause) | 1.0-1.1 | 与现有 legal 路由互补 |
| 法律依据 (law) | 0.8-0.9 | 法条查询粒度太细，权重不宜过高 |
| 程序要件 (procedure) | 0.7-0.85 | 多数查询不涉及程序问题 |
| 当事人 (parties) | 0.6-0.75 | 辅助性视角 |

## 四、实施路径

### Phase 1：LLM 分解 + 事实焦点增强（预期收益最大）

1. 新建 `multi_view.py` 模块，实现 LLM 多视角分解
2. 在 `build_query_routes()` 中集成 `muser_factual` 路由
3. Benchmark 对比：baseline vs +muser_factual
4. 预期：recall@20 提升 3-5%，事实焦点是论文验证最有效的视角

### Phase 2：Head-Tail 转换 + 请求权基础

1. 实现 head-tail 查询转换
2. 添加 `muser_cause` 路由
3. Benchmark 对比

### Phase 3：程序要件 + 当事人视角

1. 添加 `muser_procedure` 和 `muser_parties` 路由
2. 完整 benchmark 评估
3. 路由权重调优（grid search）

### Phase 4：Section-Aware 视角路由

论文发现不同视角与文档不同段落的亲和性不同：

| 视角 | 最相关段落 |
|------|----------|
| 程序要件 | statutes, case_profile |
| 请求权基础 | reasoning, fine_issue |
| 法律依据 | statutes, reasoning |
| 事实认定焦点 | fine_issue, focus, facts |
| 当事人 | case_profile, facts |

可以为每个视角指定 `section_type` 过滤，进一步提升精度。

## 五、关键文件改动清单

| 文件 | 改动内容 |
|------|---------|
| `src/legal_case_rag/retrieval/multi_view.py` | **新建** — LLM 多视角分解模块 |
| `src/legal_case_rag/retrieval/query_profile.py` | 新增 `MultiViewProfile`，`build_query_routes()` 集成多视角路由 |
| `src/legal_case_rag/retrieval/search.py` | `run_search()` 中调用 LLM 分解，传入多视角路由 |
| `src/legal_case_rag/retrieval/constants.py` | 新增多视角路由权重常量 |
| `src/legal_case_rag/retrieval/models.py` | 可选：`ChunkHit` 增加 `perspective` 字段 |
| `src/legal_case_rag/app/benchmark_service.py` | Benchmark 支持多视角方法对比 |
| `scripts/grid_search_hybrid_weights.py` | 扩展为多视角权重 grid search |

## 六、与 Excel Schema 的关系

`Legal Element Label Schema.xlsx` 定义了法律要素的层级标注体系：

- **Legal Fact** sheet — 法律事实的三级分类（如：保证 → 抵押 → 抵押权设立）
- **Dispute Focus** sheet — 争议焦点的三级分类（如：保证 → 保证的实现 → 债权人仅起诉部分共同保证人时保证效力的认定）

这个 Schema 可以直接用作 LLM Prompt 的 few-shot 示例结构，帮助 LLM 更准确地识别查询中的法律要素层级，从而生成更精准的视角查询。

## 七、预期效果

根据论文 Table 4 的消融实验结果：

| 视角组合 | BM25 nDCG@10 提升 |
|---------|-----------------|
| +Factual Focus | +3.2% |
| +Cause of Action | +1.8% |
| +Legal Basis | +1.5% |
| +Procedure | +0.9% |
| +Parties | +0.7% |
| 全部视角 | +5.6% |

考虑到你的系统已经有较强的规则引擎 QueryProfile，实际增量收益可能略低，但预计：
- Phase 1 (事实焦点)：recall@20 提升 2-4%
- Phase 1-2 完整：expected_ndcg@20 提升 3-6%
- 全部视角：expected_ndcg@20 提升 5-8%
