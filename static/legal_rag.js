const defaults = window.LEGAL_RAG_DEFAULTS || {};

const form = document.getElementById("search-form");
const resultsList = document.getElementById("results-list");
const resultsMeta = document.getElementById("results-meta");
const statusBanner = document.getElementById("status-banner");
const searchButton = document.getElementById("search-button");
const drawer = document.getElementById("case-drawer");
const drawerContent = document.getElementById("drawer-content");
const healthOpenSearch = document.getElementById("health-opensearch");
const healthSiliconFlow = document.getElementById("health-siliconflow");
const healthMode = document.getElementById("health-mode");
const benchmarkButton = document.getElementById("benchmark-button");
const benchmarkStatus = document.getElementById("benchmark-status");
const benchmarkSummary = document.getElementById("benchmark-summary");
const benchmarkResults = document.getElementById("benchmark-results");

const state = {
  results: [],
  caseCache: new Map(),
  activeDocId: "",
  activeChunkIndex: 0,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function decorateContext(text) {
  return escapeHtml(text || "")
    .replaceAll("【命中】", '<mark class="hit-mark">')
    .replaceAll("【/命中】", "</mark>");
}

function setStatus(message, kind = "idle") {
  statusBanner.className = `status-banner ${kind}`;
  statusBanner.textContent = message;
}

function setBenchmarkStatus(message, kind = "idle") {
  if (!benchmarkStatus) return;
  benchmarkStatus.className = `status-banner ${kind}`;
  benchmarkStatus.textContent = message;
}

function fillDefaults() {
  for (const [key, value] of Object.entries(defaults)) {
    const field = document.getElementById(key);
    if (!field) continue;
    if (field.type === "checkbox") {
      field.checked = Boolean(value);
    } else {
      field.value = value;
    }
  }
}

function collectPayload() {
  return {
    query: document.getElementById("query").value.trim(),
    mode: document.getElementById("mode").value,
    rerank: document.getElementById("rerank").checked,
    top_k: Number(document.getElementById("top_k").value || 8),
    chunk_top_k: Number(document.getElementById("chunk_top_k").value || 3),
    candidate_size: Number(document.getElementById("candidate_size").value || 80),
    rerank_top_n: Number(document.getElementById("rerank_top_n").value || 30),
    rerank_model_weight: Number(document.getElementById("rerank_model_weight").value || 0.25),
    rerank_rank_safe: defaults.rerank_rank_safe ?? true,
    rerank_max_rank_promotion: defaults.rerank_max_rank_promotion ?? 20,
    reason: document.getElementById("reason").value.trim(),
    trial_level: document.getElementById("trial_level").value.trim(),
    court_name: document.getElementById("court_name").value.trim(),
    section_type: document.getElementById("section_type").value.trim(),
    judge_date_from: document.getElementById("judge_date_from").value,
    judge_date_to: document.getElementById("judge_date_to").value,
    context_window: Number(document.getElementById("context_window").value || 180),
    show_context: document.getElementById("show_context").checked,
  };
}

function renderMetaChip(label, value) {
  if (!value) return "";
  return `<span class="meta-chip">${escapeHtml(label)}：${escapeHtml(value)}</span>`;
}

function renderSourceChips(matchSources) {
  if (!Array.isArray(matchSources) || !matchSources.length) return "";
  return matchSources
    .map((item) => `<span class="source-chip">${escapeHtml(item)}</span>`)
    .join("");
}

function renderChunk(resultIndex, chunk, chunkIndex) {
  const contextHtml = chunk.context_text
    ? `<p class="chunk-context">${decorateContext(chunk.context_text)}</p>`
    : "";

  return `
    <article class="chunk-card">
      <div class="chunk-header">
        <div>
          <div class="chunk-title">${escapeHtml(chunk.section_title || chunk.section_type || "命中片段")}</div>
          <div class="chip-row">${renderSourceChips(chunk.match_sources)}</div>
        </div>
        <div class="chunk-score">chunk score ${Number(chunk.score || 0).toFixed(4)}</div>
      </div>
      <p class="chunk-text">${escapeHtml(chunk.chunk_text || "")}</p>
      ${contextHtml}
      <div class="card-actions">
        <button
          type="button"
          class="ghost-action"
          data-open-case="${resultIndex}"
          data-chunk-index="${chunkIndex}"
        >
          回溯全文
        </button>
      </div>
    </article>
  `;
}

function renderResultCard(result, index) {
  const caseDoc = result.case_doc || {};
  const matchedChunks = Array.isArray(result.matched_chunks) ? result.matched_chunks : [];
  const chunkHtml = matchedChunks.map((chunk, chunkIndex) => renderChunk(index, chunk, chunkIndex)).join("");
  const rerankInfo = result.rerank_score !== undefined && result.rerank_score !== null
    ? `<div class="meta-row">
        ${renderMetaChip("Hybrid原分", Number(result.hybrid_case_score || 0).toFixed(4))}
        ${renderMetaChip("Rerank分", Number(result.rerank_score || 0).toFixed(4))}
        ${renderMetaChip("Rerank权重", Number(result.rerank_model_weight || 0).toFixed(2))}
      </div>`
    : "";

  return `
    <article class="result-card">
      <div class="result-topline">
        <div>
          <div class="result-rank">${index + 1}</div>
          <h3 class="result-title">${escapeHtml(caseDoc.case_name || result.case_name || "未命名案件")}</h3>
          <p class="result-docid">${escapeHtml(result.doc_id || "")}</p>
        </div>
        <div class="score-pill">case score ${Number(result.case_score || 0).toFixed(4)}</div>
      </div>

      <div class="meta-row">
        ${renderMetaChip("案由", caseDoc.reason || result.reason)}
        ${renderMetaChip("审级", caseDoc.trial_level || result.trial_level)}
        ${renderMetaChip("法院", caseDoc.court_name || result.court_name)}
        ${renderMetaChip("裁判日期", caseDoc.judge_date || result.judge_date)}
        ${renderMetaChip("命中块数", result.hit_count)}
      </div>
      ${rerankInfo}

      <div class="chunk-list">
        ${chunkHtml}
      </div>

      <div class="card-actions">
        <button type="button" class="ghost-action" data-open-case="${index}" data-chunk-index="0">
          查看整篇文书
        </button>
      </div>
    </article>
  `;
}

function renderEmpty(message) {
  resultsList.innerHTML = `
    <div class="empty-state">
      <h3>暂无结果</h3>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

function renderResults(data) {
  state.results = data.results || [];
  if (!state.results.length) {
    renderEmpty("这次没有召回到案件，可以尝试放宽过滤条件或切换检索模式。");
    resultsMeta.textContent = `耗时 ${data.duration_ms} ms`;
    return;
  }

  resultsList.innerHTML = state.results
    .map((result, index) => renderResultCard(result, index))
    .join("");

  resultsMeta.textContent = `返回 ${data.result_count} 个案件 · ${data.mode.toUpperCase()} · ${
    data.rerank && data.rerank.enabled ? "Rerank On" : "Rerank Off"
  } · ${data.duration_ms} ms`;
}

function metricText(value) {
  if (value === null || value === undefined) return "n/a";
  return Number(value).toFixed(4);
}

function intText(value) {
  if (value === null || value === undefined) return "0";
  return String(value);
}

function renderMetricCard(label, value, hint = "") {
  return `
    <div class="benchmark-metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(metricText(value))}</strong>
      ${hint ? `<small>${escapeHtml(hint)}</small>` : ""}
    </div>
  `;
}

function renderBenchmarkMethodSummary(methodKey, methodData, isPrimary) {
  const primary = methodData.metrics.overall || {};
  return `
    <section class="benchmark-method-summary ${isPrimary ? "primary" : ""}">
      <div class="benchmark-method-title">
        <div>
          <p class="results-kicker">${isPrimary ? "Primary View" : "Comparison"}</p>
          <h3>${escapeHtml(methodData.label || methodKey)}</h3>
        </div>
        <span>${methodData.settings.rerank ? "Rerank On" : "Rerank Off"}</span>
      </div>
      <div class="benchmark-summary-grid">
        ${renderMetricCard("Expected NDCG@20", primary["expected_ndcg@20"])}
        ${renderMetricCard("Expected NDCG@10 / @50", primary["expected_ndcg@10"], `@50 ${metricText(primary["expected_ndcg@50"])}`)}
        ${renderMetricCard("Recall@20", primary["recall@20"])}
        ${renderMetricCard("NDCG@10", primary["ndcg@10"])}
        ${renderMetricCard("MRR", primary.mrr)}
        ${renderMetricCard("MAP", primary.map)}
        ${renderMetricCard("Hit@5 / Hit@10", primary["hit@5"], `Hit@10 ${metricText(primary["hit@10"])}`)}
        ${renderMetricCard("Recall@50 / @100", primary["recall@50"], `@100 ${metricText(primary["recall@100"])}`)}
      </div>
      <div class="benchmark-compare-line">
        <span>有效 query ${intText(primary.queries_with_positive)} / ${intText(primary.queries)}</span>
      </div>
    </section>
  `;
}

function renderBenchmarkSummary(data) {
  const methodEntries = Object.entries(data.methods || {});
  const primaryMethod = data.settings.primary_method || methodEntries[0]?.[0] || "";
  benchmarkSummary.innerHTML = `
    ${methodEntries
      .map(([methodKey, methodData]) => renderBenchmarkMethodSummary(methodKey, methodData, methodKey === primaryMethod))
      .join("")}
    <p class="benchmark-note">
      当前索引：${escapeHtml(data.settings.case_index)} / ${escapeHtml(data.settings.chunk_index)}
      · TopK ${escapeHtml(data.settings.top_k)} · 候选 ${escapeHtml(data.settings.candidate_size)}
      · Rerank TopN ${escapeHtml(data.settings.rerank_top_n)}
      · Rerank权重 ${escapeHtml(Number(data.settings.rerank_model_weight || 0).toFixed(2))}
      · Rank-safe ${data.settings.rerank_rank_safe ? "On" : "Off"} / +${escapeHtml(data.settings.rerank_max_rank_promotion || 0)}
      · 限流 ${escapeHtml(data.settings.rerank_min_interval_ms)}ms / 重试 ${escapeHtml(data.settings.rerank_max_retries)}
      · 展示 Top ${escapeHtml(data.settings.display_top_n)} · ${data.duration_ms} ms
    </p>
  `;
}

function gradeBadge(item) {
  const grade = Number(item.grade || 0);
  const cls = grade >= 2 ? "positive" : grade === 1 ? "weak" : "zero";
  const label = item.is_anchor ? `锚点 G${grade}` : item.is_judged ? `G${grade}` : "未标注";
  return `<span class="grade-badge ${cls}">${escapeHtml(label)}</span>`;
}

function hitClass(item) {
  if (item.is_positive) return "positive";
  if (item.is_weak) return "weak";
  if (item.is_anchor) return "anchor";
  return "";
}

function renderMissedPositive(row) {
  const missed = row.missed_positive_doc_ids || [];
  if (!missed.length) return "";
  return `
    <div class="benchmark-missed">
      未召回正例：${missed.slice(0, 5).map((docId) => `<code>${escapeHtml(docId)}</code>`).join(" ")}
      ${missed.length > 5 ? `等 ${missed.length} 个` : ""}
    </div>
  `;
}

function failureLabel(value) {
  const labels = {
    no_positive: "无正例",
    recall_failure: "召回失败：Top100无正例",
    ranking_failure: "排序失败：Top100有正例但Top20无正例",
    hit_top20: "Top20命中",
  };
  return labels[value] || value || "未知";
}

function renderBenchmarkQuery(row, methodLabel) {
  const metric = row.metrics || {};
  const topHtml = (row.top_results || [])
    .map((item) => {
      const firstChunk = Array.isArray(item.matched_chunks) && item.matched_chunks.length
        ? `<p class="benchmark-chunk">${escapeHtml(item.matched_chunks[0].chunk_text || "")}</p>`
        : "";
      return `
        <div class="benchmark-hit ${hitClass(item)}">
          <div class="benchmark-hit-main">
            <span class="hit-rank">${item.rank}</span>
            ${gradeBadge(item)}
            <span class="hit-title">${escapeHtml(item.case_name || item.doc_id)}</span>
          </div>
          <div class="hit-docid">
            ${escapeHtml(item.doc_id)}
            ${item.case_score !== null && item.case_score !== undefined ? ` · score ${Number(item.case_score).toFixed(4)}` : ""}
          </div>
          ${firstChunk}
        </div>
      `;
    })
    .join("");

  return `
    <article class="benchmark-query-card">
      <div class="benchmark-query-head">
        <div>
          <div class="benchmark-query-id">${escapeHtml(methodLabel)} · ${escapeHtml(row.query_id)} · ${escapeHtml(row.difficulty || "")} · ${
            row.trap ? "陷阱" : "常规"
          }</div>
          <h3>${escapeHtml(row.query_text || "")}</h3>
          <p>${escapeHtml(row.main_leaf || "")}</p>
          <p class="anchor-line">来源案件：${escapeHtml(row.query_source_doc || "")}</p>
        </div>
        <div class="benchmark-query-score">
          <strong>${metricText(metric["recall@20"])}</strong>
          <span>Recall@20</span>
        </div>
      </div>
      <div class="benchmark-mini-metrics">
        <span>正例 ${escapeHtml(metric.positive_count ?? 0)}</span>
        <span>Hit@5 ${escapeHtml(metricText(metric["hit@5"]))}</span>
        <span>Hit@10 ${escapeHtml(metricText(metric["hit@10"]))}</span>
        <span>MRR ${escapeHtml(metricText(metric.mrr))}</span>
        <span>MAP ${escapeHtml(metricText(metric.map))}</span>
        <span>Expected NDCG@20 ${escapeHtml(metricText(metric["expected_ndcg@20"]))}</span>
        <span>NDCG@10 ${escapeHtml(metricText(metric["ndcg@10"]))}</span>
        <span>首正例 ${escapeHtml(row.first_positive_rank ?? "n/a")}</span>
        <span>Top20弱相关 ${escapeHtml(row.weak_top20_count ?? 0)}</span>
        <span>${escapeHtml(failureLabel(row.failure_type))}</span>
      </div>
      ${renderMissedPositive(row)}
      <div class="benchmark-hit-list">${topHtml}</div>
    </article>
  `;
}

function renderComparisonDelta(data) {
  const rows = data.comparison?.queries || [];
  if (!rows.length) return "";
  const sorted = [...rows]
    .filter((row) => row["delta_expected_ndcg@20"] !== null || row["delta_ndcg@10"] !== null || row.delta_mrr !== null)
    .sort((a, b) => Number(a["delta_expected_ndcg@20"] || 0) - Number(b["delta_expected_ndcg@20"] || 0))
    .slice(0, 8);
  const rowHtml = sorted.map((row) => `
    <div class="benchmark-delta-row">
      <strong>${escapeHtml(row.query_id)}</strong>
      <span>Expected NDCG@20 ${metricText(row["delta_expected_ndcg@20"])}</span>
      <span>NDCG ${metricText(row["delta_ndcg@10"])}</span>
      <span>MRR ${metricText(row.delta_mrr)}</span>
      <span>Recall@20 ${metricText(row["delta_recall@20"])}</span>
      <span>首正例 ${escapeHtml(row.hybrid_first_positive_rank ?? "n/a")} → ${escapeHtml(row.rerank_first_positive_rank ?? "n/a")}</span>
      <span>弱相关 ${escapeHtml(row.hybrid_weak_top20_count ?? 0)} → ${escapeHtml(row.rerank_weak_top20_count ?? 0)}</span>
    </div>
  `).join("");
  return `
    <section class="benchmark-delta-panel">
      <h3>Hybrid → Rerank 伤害最大的 query（paired-only）</h3>
      <p>共同成功 query：${escapeHtml(data.comparison.shared_query_count || 0)}。用于判断 rerank 是否把弱相关案件推到前排。</p>
      ${rowHtml}
    </section>
  `;
}

function renderBenchmark(data) {
  renderBenchmarkSummary(data);
  const methodEntries = Object.entries(data.methods || {});
  const primaryMethod = data.settings.primary_method || methodEntries[0]?.[0] || "";
  const primaryData = data.methods?.[primaryMethod] || {};
  const errorHtml = Array.isArray(data.errors) && data.errors.length
    ? `<div class="benchmark-error-note">有 ${data.errors.length} 条检索失败，已跳过失败项。首条：${escapeHtml(data.errors[0].method || "")} / ${escapeHtml(data.errors[0].query_id || "")}：${escapeHtml(data.errors[0].error || "")}</div>`
    : "";
  const comparisonHtml = methodEntries
    .filter(([methodKey]) => methodKey !== primaryMethod)
    .map(([methodKey, methodData]) => {
      const overall = methodData.metrics.overall || {};
      return `
        <div class="benchmark-compact-method">
          <strong>${escapeHtml(methodData.label || methodKey)}</strong>
          <span>Expected NDCG@20 ${metricText(overall["expected_ndcg@20"])}</span>
          <span>Recall@20 ${metricText(overall["recall@20"])}</span>
          <span>NDCG@10 ${metricText(overall["ndcg@10"])}</span>
          <span>MAP ${metricText(overall.map)}</span>
        </div>
      `;
    })
    .join("");
  benchmarkResults.innerHTML = `
    ${errorHtml}
    ${renderComparisonDelta(data)}
    ${comparisonHtml ? `<div class="benchmark-method-strip">${comparisonHtml}</div>` : ""}
    ${(primaryData.queries || [])
      .map((row) => renderBenchmarkQuery(row, primaryData.label || primaryMethod))
      .join("")}
  `;
}

async function runBenchmark() {
  const limit = Number(document.getElementById("benchmark-limit").value || 58);
  const topK = Number(document.getElementById("benchmark-top-k").value || 100);
  const candidateSize = Number(document.getElementById("benchmark-candidate-size").value || 300);
  const rerankTopN = Number(document.getElementById("benchmark-rerank-top-n").value || 100);
  const rerankModelWeight = Number(document.getElementById("benchmark-rerank-model-weight").value || 0.25);
  const rerankRankSafe = document.getElementById("benchmark-rerank-rank-safe").value !== "false";
  const rerankMaxRankPromotion = Number(document.getElementById("benchmark-rerank-max-rank-promotion").value || 20);
  const rerankMinIntervalMs = Number(document.getElementById("benchmark-rerank-min-interval-ms").value || 1200);
  const rerankMaxRetries = Number(document.getElementById("benchmark-rerank-max-retries").value || 3);
  benchmarkButton.disabled = true;
  benchmarkSummary.innerHTML = "";
  benchmarkResults.innerHTML = "";
  setBenchmarkStatus("正在逐条运行 benchmark 召回评估，这会真实调用当前 OpenSearch / embedding / rerank 链路。", "loading");

  try {
    const response = await fetch("/api/benchmark/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        methods: ["hybrid", "hybrid_rerank"],
        top_k: topK,
        candidate_size: Math.max(candidateSize, topK),
        rerank_top_n: rerankTopN,
        rerank_model_weight: rerankModelWeight,
        rerank_rank_safe: rerankRankSafe,
        rerank_max_rank_promotion: rerankMaxRankPromotion,
        rerank_min_interval_ms: rerankMinIntervalMs,
        rerank_max_retries: rerankMaxRetries,
        display_top_n: Math.min(topK, 20),
        limit,
      }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "评估失败");
    }
    renderBenchmark(data);
    setBenchmarkStatus(`评估完成：${data.settings.limit} 条 query，耗时 ${data.duration_ms} ms。`, "success");
  } catch (error) {
    setBenchmarkStatus(error.message || "评估失败，请检查 OpenSearch / SiliconFlow 配置。", "error");
  } finally {
    benchmarkButton.disabled = false;
  }
}

async function fetchHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    healthOpenSearch.textContent = data.has_opensearch_password ? "已配置" : "缺少密码";
    healthSiliconFlow.textContent = data.has_siliconflow_key ? "已配置" : "缺少 Key";
    healthMode.textContent = `${data.defaults.mode} + ${data.defaults.rerank ? "Rerank" : "Recall Only"}`;
  } catch (error) {
    healthOpenSearch.textContent = "不可用";
    healthSiliconFlow.textContent = "不可用";
    healthMode.textContent = "未知";
  }
}

async function runSearch() {
  const payload = collectPayload();
  if (!payload.query) {
    setStatus("先输入检索问题，再开始召回。", "error");
    return;
  }

  searchButton.disabled = true;
  setStatus("正在执行召回、向量检索和精排，请稍候。", "loading");
  resultsMeta.textContent = "运行中";

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "检索失败。");
    }
    renderResults(data);
    setStatus(
      `检索完成，返回 ${data.result_count} 个案件，用时 ${data.duration_ms} ms。`,
      "success",
    );
  } catch (error) {
    renderEmpty("当前检索失败，请检查 OpenSearch 或 SiliconFlow 配置。");
    setStatus(error.message || "检索失败。", "error");
    resultsMeta.textContent = "失败";
  } finally {
    searchButton.disabled = false;
  }
}

function renderFullText(fullText, chunk) {
  if (!fullText) {
    return `<div class="full-text">全文为空。</div>`;
  }

  const start = Number.isInteger(chunk?.char_start) ? chunk.char_start : -1;
  const end = Number.isInteger(chunk?.char_end) ? chunk.char_end : -1;
  if (start < 0 || end <= start || end > fullText.length) {
    return `<div class="full-text">${escapeHtml(fullText)}</div>`;
  }

  const prefix = escapeHtml(fullText.slice(0, start));
  const hit = escapeHtml(fullText.slice(start, end));
  const suffix = escapeHtml(fullText.slice(end));
  return `<div class="full-text">${prefix}<mark id="drawer-hit">${hit}</mark>${suffix}</div>`;
}

function renderLitigants(list) {
  if (!Array.isArray(list) || !list.length) return "未提供";
  return list
    .map((item) => {
      if (item && typeof item === "object") {
        return `${item.角色 || item.role || "当事人"}：${item.名称 || item.name || ""}`;
      }
      return String(item);
    })
    .join("；");
}

function renderStatutes(list) {
  if (!Array.isArray(list) || !list.length) return "未提供";
  return list.map((item) => String(item)).join("；");
}

function renderDrawer(caseResult, caseData, chunkIndex) {
  const chunks = Array.isArray(caseResult.matched_chunks) ? caseResult.matched_chunks : [];
  const safeChunkIndex = Math.min(Math.max(chunkIndex, 0), Math.max(chunks.length - 1, 0));
  const activeChunk = chunks[safeChunkIndex] || null;
  const evidenceNav = chunks
    .map((chunk, index) => {
      const activeClass = index === safeChunkIndex ? "active" : "";
      const label = chunk.section_title || chunk.section_type || `证据 ${index + 1}`;
      return `
        <button
          type="button"
          class="evidence-chip ${activeClass}"
          data-switch-chunk="${index}"
        >
          ${escapeHtml(label)}
        </button>
      `;
    })
    .join("");

  drawerContent.innerHTML = `
    <div class="drawer-header">
      <p class="results-kicker">Original Case View</p>
      <h3>${escapeHtml(caseData.case_name || caseResult.case_name || "")}</h3>
      <div class="drawer-meta">
        ${renderMetaChip("案号", caseData.doc_id || caseResult.doc_id)}
        ${renderMetaChip("案由", caseData.reason || caseResult.reason)}
        ${renderMetaChip("审级", caseData.trial_level || caseResult.trial_level)}
        ${renderMetaChip("法院", caseData.court_name || caseResult.court_name)}
        ${renderMetaChip("裁判日期", caseData.judge_date || caseResult.judge_date)}
      </div>
    </div>

    <div class="drawer-columns">
      <section class="drawer-section">
        <h4>案件信息</h4>
        <p><strong>当事人：</strong>${escapeHtml(renderLitigants(caseData.litigants))}</p>
        <p><strong>引用法条：</strong>${escapeHtml(renderStatutes(caseData.statutes))}</p>
      </section>

      <section class="drawer-section">
        <h4>命中证据</h4>
        <div class="evidence-nav">${evidenceNav || "暂无命中证据"}</div>
        ${
          activeChunk
            ? `<p class="chunk-context">${decorateContext(activeChunk.context_text || activeChunk.chunk_text || "")}</p>`
            : ""
        }
      </section>
    </div>

    <section class="drawer-section">
      <h4>原文回溯</h4>
      ${renderFullText(caseData.full_text || "", activeChunk)}
    </section>
  `;

  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");

  const hit = document.getElementById("drawer-hit");
  if (hit) {
    hit.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

async function openCase(resultIndex, chunkIndex = 0) {
  const caseResult = state.results[resultIndex];
  if (!caseResult) return;
  const docId = caseResult.doc_id;
  state.activeDocId = docId;
  state.activeChunkIndex = chunkIndex;

  let caseData = state.caseCache.get(docId);
  if (!caseData) {
    drawerContent.innerHTML = `<div class="empty-state"><h3>正在载入全文</h3><p>稍等一下，正在从案件索引中读取原文。</p></div>`;
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");

    const response = await fetch(`/api/cases/${encodeURIComponent(docId)}`);
    const data = await response.json();
    if (!response.ok || !data.ok) {
      drawerContent.innerHTML = `<div class="empty-state"><h3>载入失败</h3><p>${escapeHtml(data.error || "无法读取案件全文。")}</p></div>`;
      return;
    }
    caseData = data.case;
    state.caseCache.set(docId, caseData);
  }

  renderDrawer(caseResult, caseData, chunkIndex);
}

function closeDrawer() {
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runSearch();
});

if (benchmarkButton) {
  benchmarkButton.addEventListener("click", runBenchmark);
}

document.addEventListener("click", async (event) => {
  const exampleChip = event.target.closest(".example-chip");
  if (exampleChip) {
    document.getElementById("query").value = exampleChip.dataset.query || "";
    return;
  }

  const openButton = event.target.closest("[data-open-case]");
  if (openButton) {
    const resultIndex = Number(openButton.dataset.openCase);
    const chunkIndex = Number(openButton.dataset.chunkIndex || 0);
    await openCase(resultIndex, chunkIndex);
    return;
  }

  const switchButton = event.target.closest("[data-switch-chunk]");
  if (switchButton && drawer.classList.contains("open")) {
    const resultIndex = state.results.findIndex((item) => item.doc_id === state.activeDocId);
    if (resultIndex >= 0) {
      const chunkIndex = Number(switchButton.dataset.switchChunk || 0);
      state.activeChunkIndex = chunkIndex;
      const caseData = state.caseCache.get(state.activeDocId);
      renderDrawer(state.results[resultIndex], caseData, chunkIndex);
    }
    return;
  }

  if (event.target.closest("[data-close-drawer='true']")) {
    closeDrawer();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && drawer.classList.contains("open")) {
    closeDrawer();
  }
});

fillDefaults();
fetchHealth();
