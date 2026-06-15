const defaults = window.CASE_ADVISOR_DEFAULTS || {};

const form = document.getElementById("advisor-form");
const analyzeButton = document.getElementById("analyze-button");
const statusBanner = document.getElementById("status-banner");
const runMeta = document.getElementById("run-meta");
const analysisOutput = document.getElementById("analysis-output");
const hoverCard = document.getElementById("case-hover-card");
const drawer = document.getElementById("case-drawer");
const drawerContent = document.getElementById("drawer-content");
const healthRetrieval = document.getElementById("health-retrieval");
const healthMimo = document.getElementById("health-mimo");

const HOVER_SECTION_SPECS = [
  {
    key: "reasoning",
    title: "本院认为",
    missingText: "该文书未识别到“本院认为”部分。",
  },
  {
    key: "judgment",
    title: "裁判结果",
    missingText: "该文书未识别到“裁判结果”部分。",
  },
];

const state = {
  answerPayload: null,
  caseCache: new Map(),
  caseLinkHoverController: null,
  hoverTimer: null,
  hideHoverTimer: null,
  pendingHoverDocId: "",
  pendingHoverAnchor: null,
  currentHoverDocId: "",
  currentHoverAnchor: null,
  hoverPointer: null,
  hoverLocked: false,
  activeHoverSectionKey: "reasoning",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setStatus(message, kind = "idle") {
  statusBanner.className = `status-banner ${kind}`;
  statusBanner.textContent = message;
}

function fillDefaults() {
  Object.entries(defaults).forEach(([key, value]) => {
    const field = document.getElementById(key);
    if (!field) return;
    if (field.type === "checkbox") {
      field.checked = Boolean(value);
    } else {
      field.value = value;
    }
  });
}

function syncQueryProfileControls() {
  const profileToggle = document.getElementById("query_profile");
  const boostToggle = document.getElementById("query_profile_boost");
  if (!profileToggle || !boostToggle) return;
  boostToggle.disabled = !profileToggle.checked;
  boostToggle.closest(".toggle-wrap")?.classList.toggle("is-disabled", !profileToggle.checked);
}

function collectPayload() {
  return {
    query: document.getElementById("query").value.trim(),
    mode: document.getElementById("mode").value,
    rerank: document.getElementById("rerank").checked,
    query_profile: document.getElementById("query_profile").checked,
    query_profile_boost: document.getElementById("query_profile_boost").checked,
    top_k: Number(document.getElementById("top_k").value || defaults.top_k || 8),
    chunk_top_k: Number(document.getElementById("chunk_top_k").value || defaults.chunk_top_k || 3),
    candidate_size: Number(document.getElementById("candidate_size").value || defaults.candidate_size || 80),
    rerank_top_n: Number(document.getElementById("rerank_top_n").value || defaults.rerank_top_n || 20),
    reason: document.getElementById("reason").value.trim(),
    trial_level: document.getElementById("trial_level").value.trim(),
    court_name: document.getElementById("court_name").value.trim(),
    judge_date_from: document.getElementById("judge_date_from").value,
    judge_date_to: document.getElementById("judge_date_to").value,
  };
}

function renderCaseLink(docId, caseMap) {
  const caseInfo = caseMap?.[docId];
  const caseName = caseInfo?.case_name || docId;
  return `
    <a
      href="javascript:void(0)"
      class="case-link"
      data-doc-id="${escapeHtml(docId)}"
      title="${escapeHtml(caseName)}"
    >
      ${escapeHtml(docId)}
    </a>
  `;
}

function renderViewpoint(viewpoint, index, caseMap) {
  const supportingList = (viewpoint.supporting_cases || [])
    .map((item) => {
      return `
        <div class="case-reference-item">
          ${renderCaseLink(item.doc_id, caseMap)}
          <span class="case-reason">${escapeHtml(item.reason || "")}</span>
        </div>
      `;
    })
    .join("");

  const paragraphs = (viewpoint.analysis || "")
    .split(/\n+/)
    .filter(Boolean)
    .map((text) => `<p>${escapeHtml(text)}</p>`)
    .join("");

  return `
    <article class="viewpoint-card">
      <h3>观点${index + 1}：${escapeHtml((viewpoint.title || "").replace(/^观点[一二三四五六七八九十0-9]+[:：]?\s*/, ""))}</h3>
      ${paragraphs || `<p>${escapeHtml(viewpoint.analysis || "")}</p>`}
      <div class="reference-block">
        <p class="reference-title">参考案例：</p>
        <div class="case-reference-list">${supportingList}</div>
      </div>
    </article>
  `;
}

function renderTagList(items, emptyText = "未识别") {
  const values = Array.isArray(items) ? items.filter(Boolean) : [];
  if (!values.length) {
    return `<span class="profile-empty">${escapeHtml(emptyText)}</span>`;
  }
  return values.map((item) => `<span class="profile-tag">${escapeHtml(item)}</span>`).join("");
}

function renderQueryProfilePanel(payload) {
  const profile = payload.query_profile || {};
  const routes = Array.isArray(payload.query_routes) ? payload.query_routes : [];
  if (!Object.keys(profile).length && !routes.length) {
    return "";
  }

  const routeItems = routes
    .map((route) => {
      return `
        <div class="route-row">
          <div>
            <strong>${escapeHtml(route.name || "")}</strong>
            <span>${escapeHtml(route.type || "")} · weight ${escapeHtml(route.weight ?? "")}</span>
          </div>
          <em>${escapeHtml(route.hit_count ?? 0)} hits</em>
        </div>
      `;
    })
    .join("");

  return `
    <section class="query-profile-panel">
      <div class="profile-header">
        <div>
          <p class="eyebrow">Retrieval Trace</p>
          <h3>Query 要素解析</h3>
        </div>
        <div class="profile-switch-state">
          <span>${payload.query_profile_boost ? "要素加权已开启" : "仅多路召回"}</span>
        </div>
      </div>
      <div class="profile-grid">
        <div class="profile-item">
          <span>争议焦点候选</span>
          <div>${renderTagList(profile.dispute_focus)}</div>
        </div>
        <div class="profile-item">
          <span>否定事实</span>
          <div>${renderTagList(profile.negative_facts)}</div>
        </div>
        <div class="profile-item">
          <span>请求类型</span>
          <div>${renderTagList(profile.request_types)}</div>
        </div>
        <div class="profile-item">
          <span>案由/法律关系</span>
          <div>${renderTagList([...(profile.core_reasons || []), ...(profile.legal_relations || [])])}</div>
        </div>
      </div>
      <details class="route-details">
        <summary>召回路线 ${routes.length} 路</summary>
        <div class="route-list">${routeItems || `<div class="route-row"><span>暂无路线信息</span></div>`}</div>
      </details>
    </section>
  `;
}

function renderAnswer(payload) {
  state.answerPayload = payload;
  const answer = payload.answer || {};
  const caseMap = payload.source_cases || {};

  const headlineStats = `
    <section class="summary-card summary-card-metrics">
      <div class="summary-main">
        <h3>${escapeHtml(answer.title || "类案分析")}</h3>
        <p>${escapeHtml(answer.summary || "")}</p>
      </div>
      <div class="summary-side">
        <div class="summary-metric">
          <span>观点数</span>
          <strong>${(answer.viewpoints || []).length}</strong>
        </div>
        <div class="summary-metric">
          <span>引用案例</span>
          <strong>${(payload.retrieval?.cited_doc_ids || []).length}</strong>
        </div>
      </div>
    </section>
  `;

  const viewpointsBlock = (answer.viewpoints || [])
    .map((item, index) => renderViewpoint(item, index, caseMap))
    .join("");

  const noticeBlock = answer.notice
    ? `<section class="notice-card">${escapeHtml(answer.notice)}</section>`
    : "";

  analysisOutput.innerHTML = `${headlineStats}${renderQueryProfilePanel(payload)}${viewpointsBlock}${noticeBlock}`;
  bindCaseLinkHoverHandlers();
}

function renderPlaceholder(message) {
  detachCaseLinkHoverHandlers();
  hideHoverCard();
  analysisOutput.innerHTML = `
    <div class="placeholder-card">
      <h3>暂无结果</h3>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

function renderStreamingDraft(text) {
  detachCaseLinkHoverHandlers();
  hideHoverCard();
  analysisOutput.innerHTML = `
    <section class="summary-card">
      <h3>正在生成分析</h3>
      <p class="stream-hint">模型正在流式生成可阅读的分析草稿，结构化观点会在完成后自动整理。</p>
      <pre class="stream-output">${escapeHtml(text || "正在等待模型输出...")}</pre>
    </section>
  `;
}

function inferHoverSectionKey(section) {
  const directKey = String(section?.section_key || "").trim();
  if (directKey === "reasoning" || directKey === "judgment") {
    return directKey;
  }
  const title = String(section?.section_title || "");
  if (title.includes("本院认为")) return "reasoning";
  if (title.includes("裁判结果") || title.includes("判决如下")) return "judgment";
  return "";
}

function normalizePreviewSections(caseInfo) {
  const rawSections = Array.isArray(caseInfo?.preview_sections) ? caseInfo.preview_sections : [];
  const sectionMap = new Map();

  rawSections.forEach((section) => {
    const key = inferHoverSectionKey(section);
    if (!key || sectionMap.has(key)) return;
    sectionMap.set(key, section);
  });

  return HOVER_SECTION_SPECS.map((spec) => {
    const section = sectionMap.get(spec.key) || {};
    const chunkText = String(section.chunk_text || "");
    const contextText = String(section.context_text || chunkText || spec.missingText);
    return {
      section_key: spec.key,
      section_title: spec.title,
      chunk_text: chunkText,
      context_text: contextText,
      char_start: section.char_start ?? null,
      char_end: section.char_end ?? null,
      available: section.available ?? Boolean(chunkText.trim()),
    };
  });
}

function getDefaultHoverSectionKey(sections) {
  return sections.find((item) => item.section_key === "reasoning")?.section_key
    || sections[0]?.section_key
    || "reasoning";
}

function storeHoverPointer(event) {
  if (typeof event?.clientX !== "number" || typeof event?.clientY !== "number") {
    return;
  }
  state.hoverPointer = {
    clientX: event.clientX,
    clientY: event.clientY,
  };
}

function isPointerOnCaseLink(caseLink, pointer = state.hoverPointer) {
  if (!caseLink || !pointer) {
    return false;
  }
  if (!caseLink.matches(":hover")) {
    return false;
  }
  const hitElement = document.elementFromPoint(pointer.clientX, pointer.clientY);
  return Boolean(hitElement?.closest?.(".case-link") === caseLink);
}

function detachCaseLinkHoverHandlers() {
  if (state.caseLinkHoverController) {
    state.caseLinkHoverController.abort();
    state.caseLinkHoverController = null;
  }
}

function bindCaseLinkHoverHandlers() {
  detachCaseLinkHoverHandlers();
  const caseLinks = analysisOutput.querySelectorAll(".case-link");
  if (!caseLinks.length) {
    return;
  }

  const controller = new AbortController();
  const { signal } = controller;
  state.caseLinkHoverController = controller;

  caseLinks.forEach((caseLink) => {
    caseLink.addEventListener(
      "mouseenter",
      (event) => {
        storeHoverPointer(event);
        clearTimeout(state.hoverTimer);
        clearTimeout(state.hideHoverTimer);
        state.pendingHoverDocId = caseLink.dataset.docId || "";
        state.pendingHoverAnchor = caseLink;
        state.hoverTimer = setTimeout(() => {
          if (state.pendingHoverAnchor !== caseLink) {
            return;
          }
          if (!isPointerOnCaseLink(caseLink, state.hoverPointer)) {
            return;
          }
          showHoverCard(caseLink.dataset.docId, caseLink, state.hoverPointer);
        }, 120);
      },
      { signal },
    );

    caseLink.addEventListener(
      "mousemove",
      (event) => {
        storeHoverPointer(event);
      },
      { signal },
    );

    caseLink.addEventListener(
      "mouseleave",
      (event) => {
        clearTimeout(state.hoverTimer);
        if (state.pendingHoverAnchor === caseLink) {
          state.pendingHoverDocId = "";
          state.pendingHoverAnchor = null;
        }
        const toElement = event.relatedTarget;
        if (toElement && (toElement.closest?.("#case-hover-card") || toElement.closest?.(".case-link"))) {
          return;
        }
        scheduleHideHoverCard();
      },
      { signal },
    );
  });
}

function cleanStreamingPreview(rawText) {
  const text = String(rawText || "");
  const markers = ["整体判断：", "整体判断", "观点一：", "观点1：", "观点一", "观点1"];
  let start = -1;
  for (const marker of markers) {
    const index = text.indexOf(marker);
    if (index >= 0 && (start < 0 || index < start)) {
      start = index;
    }
  }
  if (start >= 0) {
    return text.slice(start);
  }
  return "";
}

async function fetchHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    healthRetrieval.textContent = data.has_opensearch_password && data.has_siliconflow_key ? "已就绪" : "待配置";
    healthMimo.textContent = data.has_mimo_key ? "已就绪" : "待配置";
  } catch (error) {
    healthRetrieval.textContent = "不可用";
    healthMimo.textContent = "不可用";
  }
}

async function analyzeQuestion() {
  const payload = collectPayload();
  if (!payload.query) {
    setStatus("先输入一个需要分析的问题。", "error");
    return;
  }

  analyzeButton.disabled = true;
  setStatus("正在召回案例并生成观点式回答，请稍候。", "loading");
  runMeta.textContent = "生成中";
  renderStreamingDraft("");

  try {
    const response = await fetch("/api/analyze/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok || !response.body) {
      throw new Error("流式生成启动失败。");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let draftText = "";
    let shownText = "";
    let finalPayload = null;
    let finalDuration = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const rawEvent of events) {
        const line = rawEvent
          .split("\n")
          .find((item) => item.startsWith("data: "));
        if (!line) continue;
        const eventPayload = JSON.parse(line.slice(6));
        if (eventPayload.type === "retrieval") {
          runMeta.textContent = `已召回 ${eventPayload.result_count} 个案件，正在生成回答`;
        } else if (eventPayload.type === "token") {
          draftText += eventPayload.text || "";
          shownText = cleanStreamingPreview(draftText);
          renderStreamingDraft(shownText);
        } else if (eventPayload.type === "result") {
          finalPayload = eventPayload.payload;
          finalDuration = eventPayload.duration_ms;
        } else if (eventPayload.type === "error") {
          throw new Error(eventPayload.error || "生成失败。");
        }
      }
    }

    if (!finalPayload) {
      throw new Error("未收到最终结果。");
    }

    renderAnswer(finalPayload);
    const citedCount = (finalPayload.retrieval?.cited_doc_ids || []).length;
    runMeta.textContent = `${finalDuration || 0} ms · 引用案例 ${citedCount} 个`;
    setStatus(
      `已生成类案分析，基于 ${finalPayload.retrieval?.result_count || 0} 个召回案件组织观点。`,
      "success",
    );
  } catch (error) {
    renderPlaceholder("当前无法完成类案分析，请检查 OpenSearch、SiliconFlow 或 Mimo 配置。");
    runMeta.textContent = "失败";
    setStatus(error.message || "生成失败。", "error");
  } finally {
    analyzeButton.disabled = false;
  }
}

function renderHoverCard(caseInfo) {
  const sections = normalizePreviewSections(caseInfo);
  if (!sections.some((item) => item.section_key === state.activeHoverSectionKey)) {
    state.activeHoverSectionKey = getDefaultHoverSectionKey(sections);
  }
  const activeSection = sections.find((item) => item.section_key === state.activeHoverSectionKey) || sections[0];
  const tabHtml = sections
    .map((item, index) => {
      const activeClass = item.section_key === state.activeHoverSectionKey ? "active" : "";
      return `
        <button
          type="button"
          class="hover-tab ${activeClass}"
          data-hover-section-key="${escapeHtml(item.section_key || "")}"
        >
          ${escapeHtml(item.section_title || `片段${index + 1}`)}
        </button>
      `;
    })
    .join("");

  hoverCard.innerHTML = `
    <div class="hover-header">
      <div>
        <h4 class="hover-title">${escapeHtml(caseInfo.case_name || "")}</h4>
        <div class="hover-meta">
          <span class="hover-chip">${escapeHtml(caseInfo.doc_id || "")}</span>
          <span class="hover-chip">${escapeHtml(caseInfo.court_name || "")}</span>
          <span class="hover-chip">${escapeHtml(caseInfo.reason || "")}</span>
        </div>
      </div>
      <button type="button" class="hover-expand-button" data-expand-doc-id="${escapeHtml(caseInfo.doc_id || "")}" title="展开完整文书">
        ↗
      </button>
    </div>
    <div class="hover-tabs">${tabHtml}</div>
    <div class="hover-body">
      ${
        activeSection
          ? `
            <div class="hover-section">
              <div class="hover-section-title">${escapeHtml(activeSection.section_title || "")}</div>
              <div class="hover-section-text">${escapeHtml(activeSection.context_text || activeSection.chunk_text || "")}</div>
            </div>
          `
          : `<div class="hover-section"><div class="hover-section-text">暂无可展示内容</div></div>`
      }
    </div>
  `;
}

function positionHoverCard(pointer = state.hoverPointer) {
  if (state.hoverLocked || !pointer) return;
  const margin = 18;
  const cardWidth = hoverCard.offsetWidth || 420;
  const cardHeight = hoverCard.offsetHeight || 240;
  let left = pointer.clientX - Math.floor(cardWidth / 2);
  let top = pointer.clientY + 18;

  if (left + cardWidth + margin > window.innerWidth) {
    left = window.innerWidth - cardWidth - margin;
  }
  if (top + cardHeight + margin > window.innerHeight) {
    top = window.innerHeight - cardHeight - margin;
  }

  hoverCard.style.left = `${Math.max(margin, left)}px`;
  hoverCard.style.top = `${Math.max(margin, top)}px`;
}

function showHoverCard(docId, anchor, pointer = state.hoverPointer) {
  const caseMap = state.answerPayload?.source_cases || {};
  const caseInfo = caseMap[docId];
  if (!caseInfo) return;
  clearTimeout(state.hideHoverTimer);
  clearTimeout(state.hoverTimer);
  state.pendingHoverDocId = "";
  state.pendingHoverAnchor = null;
  state.currentHoverDocId = docId;
  state.currentHoverAnchor = anchor || null;
  const sections = normalizePreviewSections(caseInfo);
  state.activeHoverSectionKey = getDefaultHoverSectionKey(sections);
  renderHoverCard(caseInfo);
  hoverCard.classList.remove("hidden");
  positionHoverCard(pointer);
}

function hideHoverCard() {
  clearTimeout(state.hideHoverTimer);
  clearTimeout(state.hoverTimer);
  state.pendingHoverDocId = "";
  state.pendingHoverAnchor = null;
  hoverCard.classList.add("hidden");
  state.currentHoverDocId = "";
  state.currentHoverAnchor = null;
  state.hoverLocked = false;
}

function scheduleHideHoverCard() {
  clearTimeout(state.hideHoverTimer);
  state.hideHoverTimer = setTimeout(() => {
    hideHoverCard();
  }, 40);
}

async function fetchCaseDetail(docId) {
  if (state.caseCache.has(docId)) {
    return state.caseCache.get(docId);
  }
  const response = await fetch(`/api/cases/${encodeURIComponent(docId)}`);
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "无法获取案件详情。");
  }
  state.caseCache.set(docId, data.case);
  return data.case;
}

function renderDrawer(caseData) {
  const sections = caseData.sections || [];
  const navLinks = sections
    .map((section, index) => {
      return `
        <button type="button" class="drawer-nav-link ${index === 0 ? "active" : ""}" data-scroll-section="${escapeHtml(section.id)}">
          ${escapeHtml(section.title || section.id || "")}
        </button>
      `;
    })
    .join("");

  const sectionBlocks = sections
    .map((section) => {
      return `
        <section class="section-card" id="section-${escapeHtml(section.id)}">
          <h3>${escapeHtml(section.title || section.id || "")}</h3>
          <pre>${escapeHtml(section.content || "")}</pre>
        </section>
      `;
    })
    .join("");

  drawerContent.innerHTML = `
    <header class="drawer-header">
      <p class="eyebrow">Case Detail</p>
      <h2>${escapeHtml(caseData.case_name || "")}</h2>
    </header>

    <div class="drawer-metadata">
      <div class="drawer-meta-item"><strong>审理法院：</strong>${escapeHtml(caseData.court_name || "")}</div>
      <div class="drawer-meta-item"><strong>审理程序：</strong>${escapeHtml(caseData.trial_level || "")}</div>
      <div class="drawer-meta-item"><strong>审结日期：</strong>${escapeHtml(caseData.judge_date || "")}</div>
      <div class="drawer-meta-item"><strong>案号：</strong>${escapeHtml(caseData.doc_id || "")}</div>
      <div class="drawer-meta-item"><strong>案由：</strong>${escapeHtml(caseData.reason || "")}</div>
      <div class="drawer-meta-item"><strong>发布日期：</strong>${escapeHtml(caseData.publish_date || "")}</div>
    </div>

    <div class="drawer-layout">
      <aside class="drawer-nav">
        <div class="drawer-nav-list">${navLinks}</div>
      </aside>
      <main class="drawer-main">${sectionBlocks}</main>
    </div>
  `;

  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
}

async function openCaseDrawer(docId) {
  try {
    const caseData = await fetchCaseDetail(docId);
    renderDrawer(caseData);
  } catch (error) {
    drawerContent.innerHTML = `
      <div class="placeholder-card">
        <h3>案件详情载入失败</h3>
        <p>${escapeHtml(error.message || "无法读取案件全文。")}</p>
      </div>
    `;
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
  }
}

function closeDrawer() {
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await analyzeQuestion();
});

document.getElementById("query_profile")?.addEventListener("change", syncQueryProfileControls);

document.addEventListener("click", async (event) => {
  const exampleButton = event.target.closest(".example-button");
  if (exampleButton) {
    document.getElementById("query").value = exampleButton.dataset.query || "";
    return;
  }

  const caseLink = event.target.closest(".case-link");
  if (caseLink) {
    event.preventDefault();
    return;
  }

  const expandButton = event.target.closest("[data-expand-doc-id]");
  if (expandButton) {
    await openCaseDrawer(expandButton.dataset.expandDocId);
    hideHoverCard();
    return;
  }

  const hoverTab = event.target.closest("[data-hover-section-key]");
  if (hoverTab) {
    const caseMap = state.answerPayload?.source_cases || {};
    const caseInfo = caseMap[state.currentHoverDocId];
    if (caseInfo) {
      state.activeHoverSectionKey = hoverTab.dataset.hoverSectionKey || getDefaultHoverSectionKey(normalizePreviewSections(caseInfo));
      renderHoverCard(caseInfo);
    }
    return;
  }

  const closeTrigger = event.target.closest("[data-close-drawer='true']");
  if (closeTrigger) {
    closeDrawer();
    return;
  }

  const navButton = event.target.closest("[data-scroll-section]");
  if (navButton) {
    const sectionId = navButton.dataset.scrollSection;
    const target = document.getElementById(`section-${sectionId}`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      drawerContent.querySelectorAll(".drawer-nav-link").forEach((item) => {
        item.classList.toggle("active", item === navButton);
      });
    }
  }
});

document.addEventListener("mousemove", (event) => {
  storeHoverPointer(event);
  if (!hoverCard.classList.contains("hidden") && !state.hoverLocked) {
    positionHoverCard();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeDrawer();
    hideHoverCard();
  }
});

hoverCard.addEventListener("mouseenter", () => {
  clearTimeout(state.hideHoverTimer);
  state.hoverLocked = true;
});

hoverCard.addEventListener("mouseleave", () => {
  state.hoverLocked = false;
  const hitElement = state.hoverPointer
    ? document.elementFromPoint(state.hoverPointer.clientX, state.hoverPointer.clientY)
    : null;
  if (hitElement?.closest?.(".case-link")) {
    return;
  }
  scheduleHideHoverCard();
});

fillDefaults();
syncQueryProfileControls();
fetchHealth();
