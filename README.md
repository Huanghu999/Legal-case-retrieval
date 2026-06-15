# Legal Case RAG

当前项目只保留基于 CaseLaw-Bench 标注数据集的类案召回链路。

## 数据源

- `data/caselaw-benchmark release/data/corpus.jsonl`
- `data/caselaw-benchmark release/data/queries.jsonl`
- `data/caselaw-benchmark release/data/qrels.jsonl`

旧的 CSV / `rag_dataset` 自造评估链路已经移除。

## 重建索引

```powershell
$env:SILICONFLOW_API_KEY="your SiliconFlow API Key"
$env:OPENSEARCH_PASSWORD="your OpenSearch admin password"
python scripts\rebuild_benchmark_rag_index.py
```

默认索引：

- `caselaw_benchmark_cases_v1`
- `caselaw_benchmark_chunks_v1`

## 启动页面

召回调试与 benchmark 评估：

```powershell
python legal_rag_web.py
```

打开 `http://127.0.0.1:7860/`，页面底部可以运行标注集召回评估。

类案分析问答：

```powershell
$env:MIMO_API_KEY="your Mimo API Key"
python legal_case_advisor_web.py
```

打开 `http://127.0.0.1:7870/`。
