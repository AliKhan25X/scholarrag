const state = {
  documents: [],
  chunks: [],
  chunkCount: 0,
  history: [],
  backend: false,
  avgConfidence: 0,
  currentAnswer: null,
  dashboardMetric: "confidence",
  dashboardRisk: "all",
  selectedDocumentIds: [],
  benchmarkResults: [],
};

const apiBase = `${window.location.origin}/api`;

const stopWords = new Set(
  "a an and are as at be by for from has have how in into is it its of on or our that the their this to was were what when where which who why with your".split(
    " "
  )
);

const sampleDocs = [
  {
    name: "retrieval_augmented_generation_notes.txt",
    text:
      "Retrieval augmented generation improves factual grounding by searching a knowledge base before generation. A RAG system usually has document ingestion, text chunking, embedding generation, vector search, prompt construction, answer generation, and citation display. The main contribution is that the language model answers from retrieved evidence instead of relying only on memorized parameters. Strong RAG systems add reranking, hybrid search, metadata filtering, and evaluation for faithfulness.",
  },
  {
    name: "llm_evaluation_playbook.txt",
    text:
      "LLM evaluation measures whether a model answer is relevant, faithful, concise, and supported by citations. Production teams track latency, cost, token usage, retrieval precision, citation coverage, and hallucination risk. A practical dashboard should show question history, retrieved chunks, model output, evaluator scores, and failure categories. Faithfulness means the answer is supported by the retrieved context.",
  },
  {
    name: "phd_research_assistant_architecture.txt",
    text:
      "A PhD research assistant can read papers, summarize methods, compare contributions, and find gaps. The system should preserve page numbers and section names so every answer has a source. The best user experience includes multi-document chat, export to markdown, document filters, highlighted citations, and a research trend dashboard. Privacy can be improved by using local embeddings and local LLMs.",
  },
];

const fileInput = document.querySelector("#fileInput");
const loadSampleButton = document.querySelector("#loadSampleButton");
const askButton = document.querySelector("#askButton");
const exampleButton = document.querySelector("#exampleButton");
const resetButton = document.querySelector("#resetButton");
const questionInput = document.querySelector("#questionInput");
const providerSelect = document.querySelector("#providerSelect");
const answerBox = document.querySelector("#answerBox");
const evidenceList = document.querySelector("#evidenceList");
const documentList = document.querySelector("#documentList");
const selectAllDocsButton = document.querySelector("#selectAllDocsButton");
const historyList = document.querySelector("#historyList");
const exportAnswerButton = document.querySelector("#exportAnswerButton");
const exportSessionButton = document.querySelector("#exportSessionButton");
const printButton = document.querySelector("#printButton");
const exportPreview = document.querySelector("#exportPreview");
const exportStatus = document.querySelector("#exportStatus");
const trendChart = document.querySelector("#trendChart");
const riskBreakdown = document.querySelector("#riskBreakdown");
const reviewList = document.querySelector("#reviewList");
const dashboardInsight = document.querySelector("#dashboardInsight");
const citationMap = document.querySelector("#citationMap");
const runBenchmarkButton = document.querySelector("#runBenchmarkButton");
const benchmarkList = document.querySelector("#benchmarkList");
const benchmarkSummary = document.querySelector("#benchmarkSummary");

const benchmarkQuestions = [
  {
    question: "What are the main stages in a RAG system?",
    expectedSource: "retrieval_augmented_generation_notes.txt",
    expectedTerms: ["ingestion", "chunking", "retrieval", "citation"],
  },
  {
    question: "Which quality metrics should LLM teams track?",
    expectedSource: "llm_evaluation_playbook.txt",
    expectedTerms: ["latency", "cost", "faithful", "citation"],
  },
  {
    question: "What should a PhD research assistant preserve for citations?",
    expectedSource: "phd_research_assistant_architecture.txt",
    expectedTerms: ["page", "section", "source"],
  },
];

async function apiFetch(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "API request failed");
  return payload;
}

async function detectBackend() {
  try {
    const health = await apiFetch("/health");
    state.backend = Boolean(health.ok);
    await syncBackendState();
    document.querySelector("#retrievalStatus").textContent = "Backend";
  } catch {
    state.backend = false;
  }
}

async function syncBackendState() {
  const snapshot = await apiFetch("/documents");
  state.documents = snapshot.documents || [];
  state.chunks = [];
  state.chunkCount = snapshot.chunks || 0;
  state.history = snapshot.history || [];
  state.avgConfidence = snapshot.avg_confidence || 0;
  syncSelectedDocuments();
  render();
}

async function fileToBase64(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const batchSize = 0x8000;
  for (let i = 0; i < bytes.length; i += batchSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + batchSize));
  }
  return btoa(binary);
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2 && !stopWords.has(token));
}

function splitSentences(text) {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function chunkText(text, documentId, documentName) {
  const sentences = splitSentences(text);
  const chunks = [];
  let current = [];
  let currentLength = 0;

  sentences.forEach((sentence) => {
    const sentenceLength = tokenize(sentence).length;
    if (currentLength + sentenceLength > 95 && current.length) {
      chunks.push(current.join(" "));
      current = [];
      currentLength = 0;
    }
    current.push(sentence);
    currentLength += sentenceLength;
  });

  if (current.length) chunks.push(current.join(" "));

  return chunks.map((chunk, index) => ({
    id: `${documentId}-${index + 1}`,
    documentId,
    documentName,
    page: Math.floor(index / 3) + 1,
    index: index + 1,
    text: chunk,
    terms: termFrequency(tokenize(chunk)),
  }));
}

function termFrequency(tokens) {
  const counts = {};
  tokens.forEach((token) => {
    counts[token] = (counts[token] || 0) + 1;
  });
  return counts;
}

function addDocument(name, text) {
  const id = `doc-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const cleanedText = text.replace(/\r/g, "").trim();
  if (!cleanedText) return;

  const document = {
    id,
    name,
    chars: cleanedText.length,
    uploadedAt: new Date().toISOString(),
  };
  const chunks = chunkText(cleanedText, id, name);
  state.documents.push(document);
  state.selectedDocumentIds = Array.from(new Set([...state.selectedDocumentIds, id]));
  state.chunks.push(...chunks);
  state.chunkCount = state.chunks.length;
  syncSelectedDocuments();
  render();
}

function inverseDocumentFrequency(term) {
  const docsWithTerm = state.chunks.filter((chunk) => chunk.terms[term]).length;
  return Math.log((state.chunks.length + 1) / (docsWithTerm + 1)) + 1;
}

function scoreChunk(queryTokens, chunk) {
  let score = 0;
  queryTokens.forEach((term) => {
    if (chunk.terms[term]) {
      score += chunk.terms[term] * inverseDocumentFrequency(term);
    }
  });
  return score;
}

function retrieve(question, limit = 4) {
  const queryTokens = tokenize(question);
  const allowedDocuments = new Set(state.selectedDocumentIds);
  return state.chunks
    .filter((chunk) => !allowedDocuments.size || allowedDocuments.has(chunk.documentId))
    .map((chunk) => ({
      ...chunk,
      score: scoreChunk(queryTokens, chunk),
    }))
    .filter((chunk) => chunk.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

function buildAnswer(question, evidence) {
  if (!state.chunks.length) {
    return "No documents are indexed yet. Upload research notes or load the sample papers first.";
  }
  if (!evidence.length) {
    return "I could not find enough grounded evidence in the indexed documents. Try asking with terms that appear in the papers.";
  }

  const queryTokens = new Set(tokenize(question));
  const selectedSentences = [];
  evidence.forEach((chunk) => {
    splitSentences(chunk.text).forEach((sentence) => {
      const sentenceTokens = tokenize(sentence);
      const overlap = sentenceTokens.filter((token) => queryTokens.has(token)).length;
      if (overlap > 0) {
        selectedSentences.push({ sentence, overlap, chunk });
      }
    });
  });

  const best = selectedSentences
    .sort((a, b) => b.overlap - a.overlap)
    .slice(0, 4)
    .map((item) => item.sentence);

  const answerSentences = best.length ? best : evidence.slice(0, 2).map((chunk) => chunk.text);
  return answerSentences.join(" ");
}

function evaluate(question, answer, evidence, latency) {
  const queryTokens = tokenize(question);
  const answerTokens = tokenize(answer);
  const evidenceText = evidence.map((chunk) => chunk.text).join(" ");
  const evidenceTokens = new Set(tokenize(evidenceText));
  const queryMatches = queryTokens.filter((token) => answerTokens.includes(token)).length;
  const supportedTokens = answerTokens.filter((token) => evidenceTokens.has(token)).length;

  const relevance = Math.min(98, Math.round((queryMatches / Math.max(queryTokens.length, 1)) * 100));
  const faithfulness = Math.min(99, Math.round((supportedTokens / Math.max(answerTokens.length, 1)) * 100));
  const citation = evidence.length ? Math.min(96, 60 + evidence.length * 9) : 0;
  const risk = faithfulness > 80 ? "Low" : faithfulness > 55 ? "Medium" : "High";

  return { relevance, faithfulness, citation, risk, latency };
}

function highlightTerms(text, question) {
  const terms = Array.from(new Set(tokenize(question))).slice(0, 8);
  let highlighted = escapeHtml(text);
  terms.forEach((term) => {
    const pattern = new RegExp(`\\b(${term})\\b`, "gi");
    highlighted = highlighted.replace(pattern, "<mark>$1</mark>");
  });
  return highlighted;
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (char) => {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return map[char];
  });
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  if (state.backend) {
    try {
      const result = await apiFetch("/ask", {
        method: "POST",
        body: JSON.stringify({ question, provider: providerSelect.value, documentIds: state.selectedDocumentIds }),
      });
      const metrics = {
        ...result.metrics,
        provider: result.provider,
        providerWarning: result.providerWarning,
      };
      renderAnswer(question, result.answer, result.evidence, metrics);
      await syncBackendState();
      renderAnswer(question, result.answer, result.evidence, metrics);
    } catch (error) {
      answerBox.textContent = error.message;
    }
    return;
  }

  const startedAt = performance.now();
  const evidence = retrieve(question);
  const answer = buildAnswer(question, evidence);
  const latency = Math.round(performance.now() - startedAt);
  const metrics = evaluate(question, answer, evidence, latency);
  state.history.push({ question, answer, evidence, metrics, createdAt: new Date().toISOString() });
  renderAnswer(question, answer, evidence, metrics);
  renderHistory();
  renderEvaluationDashboard();
}

function renderAnswer(question, answer, evidence, metrics) {
  state.currentAnswer = { question, answer, evidence, metrics, createdAt: new Date().toISOString() };
  const sources = evidence
    .map(
      (chunk, index) =>
        `<li>[${index + 1}] ${escapeHtml(chunk.documentName)} - page ${chunk.page}, chunk ${chunk.index}</li>`
    )
    .join("");

  answerBox.innerHTML = `
    <div>${highlightTerms(answer, question)}</div>
    ${metrics.provider ? `<p><strong>Provider:</strong> ${escapeHtml(metrics.provider)}${metrics.providerWarning ? ` - ${escapeHtml(metrics.providerWarning)}` : ""}</p>` : ""}
    <div class="sources">
      <strong>Sources</strong>
      <ol>${sources || "<li>No supporting source found.</li>"}</ol>
    </div>
  `;

  evidenceList.innerHTML = evidence
    .map(
      (chunk, index) => `
      <div class="evidence" data-evidence-index="${index + 1}">
        <div class="evidence-top">
          <span>[${index + 1}] ${escapeHtml(chunk.documentName)}</span>
          <span>${formatScores(chunk)}</span>
        </div>
        <p>${highlightTerms(chunk.text, question)}</p>
      </div>`
    )
    .join("");

  document.querySelector("#retrievalStatus").textContent = evidence.length ? "Retrieved" : "No match";
  document.querySelector("#latencyMetric").textContent = `${metrics.latency} ms`;
  document.querySelector("#relevanceScore").textContent = `${metrics.relevance}%`;
  document.querySelector("#faithfulnessScore").textContent = `${metrics.faithfulness}%`;
  document.querySelector("#citationScore").textContent = `${metrics.citation}%`;
  document.querySelector("#riskScore").textContent = metrics.risk;
  setExportAvailability();
  renderCitationExplorer(evidence);
  renderStats();
}

function formatScores(chunk) {
  const hybrid = Number(chunk.score || 0).toFixed(2);
  if (chunk.vectorScore == null || chunk.keywordScore == null) {
    return `score ${hybrid}`;
  }
  return `hybrid ${hybrid} | vector ${Number(chunk.vectorScore).toFixed(2)}`;
}

function renderStats() {
  document.querySelector("#docCount").textContent = String(state.documents.length);
  document.querySelector("#chunkCount").textContent = String(state.backend ? state.chunkCount : state.chunks.length);
  const avgConfidence = state.backend
    ? state.avgConfidence
    : state.history.length
    ? Math.round(
        state.history.reduce((sum, item) => sum + (item.metrics.relevance + item.metrics.faithfulness) / 2, 0) /
          state.history.length
      )
    : 0;
  document.querySelector("#confidenceMetric").textContent = `${avgConfidence}%`;
}

function syncSelectedDocuments() {
  const availableIds = state.documents.map((document) => document.id);
  const available = new Set(availableIds);
  state.selectedDocumentIds = state.selectedDocumentIds.filter((id) => available.has(id));
  if (!state.selectedDocumentIds.length && availableIds.length) {
    state.selectedDocumentIds = [...availableIds];
  }
}

function renderDocuments() {
  syncSelectedDocuments();
  const selected = new Set(state.selectedDocumentIds);
  documentList.innerHTML = state.documents.length
    ? state.documents
        .map(
          (document) => `
      <label class="doc-item selectable-doc">
        <input type="checkbox" data-document-id="${escapeHtml(document.id)}" ${selected.has(document.id) ? "checked" : ""} />
        <span>
          <strong>${escapeHtml(document.name)}</strong>
          <small>${document.chars.toLocaleString()} chars</small>
        </span>
      </label>`
        )
        .join("")
    : `<div class="doc-item"><span>No documents indexed yet.</span></div>`;
  selectAllDocsButton.textContent =
    state.documents.length && state.selectedDocumentIds.length < state.documents.length ? "All" : "All on";
}

function renderCitationExplorer(evidence = []) {
  const grouped = evidence.reduce((groups, chunk, index) => {
    const key = chunk.documentId || chunk.documentName;
    if (!groups[key]) {
      groups[key] = { documentName: chunk.documentName, chunks: [] };
    }
    groups[key].chunks.push({ ...chunk, sourceIndex: index + 1 });
    return groups;
  }, {});
  const groups = Object.values(grouped);
  document.querySelector("#citationStatus").textContent = groups.length ? `${groups.length} sources` : "No sources";
  citationMap.innerHTML = groups.length
    ? groups
        .map(
          (group) => `
      <article class="citation-card">
        <div class="citation-head">
          <strong>${escapeHtml(group.documentName)}</strong>
          <span>${group.chunks.length} chunks</span>
        </div>
        ${group.chunks
          .map(
            (chunk) => `
          <button class="citation-hit" data-source-index="${chunk.sourceIndex}">
            <span>[${chunk.sourceIndex}] page ${chunk.page}, chunk ${chunk.index}</span>
            <small>${escapeHtml(chunk.text.slice(0, 180))}${chunk.text.length > 180 ? "..." : ""}</small>
          </button>`
          )
          .join("")}
      </article>`
        )
        .join("")
    : `<div class="dashboard-empty">Ask a question to inspect source coverage.</div>`;
}

function renderHistory() {
  document.querySelector("#historyCount").textContent = `${state.history.length} saved`;
  setExportAvailability();

  historyList.innerHTML = state.history.length
    ? state.history
        .slice()
        .reverse()
        .map((item, index) => {
          const displayIndex = state.history.length - index;
          const metrics = item.metrics || {};
          return `
      <button class="history-item" data-history-index="${displayIndex - 1}">
        <span>Q${displayIndex}</span>
        <strong>${escapeHtml(item.question || "Untitled question")}</strong>
        <small>${metrics.relevance ?? 0}% relevance | ${metrics.faithfulness ?? 0}% faithful</small>
      </button>`;
        })
        .join("")
    : `<div class="history-empty">Ask a question to build a reusable research session.</div>`;
}

function render() {
  renderStats();
  renderDocuments();
  renderHistory();
  renderEvaluationDashboard();
  renderBenchmarkResults();
}

function renderEvaluationDashboard() {
  const summary = buildEvaluationSummary();
  const metricLabel = metricLabels[state.dashboardMetric];
  const filteredLabel = state.dashboardRisk === "all" ? "all risks" : `${state.dashboardRisk} risk`;
  document.querySelector("#dashboardStatus").textContent = summary.count ? `${summary.filteredCount}/${summary.count} runs` : "No runs";
  document.querySelector("#avgRelevanceMetric").textContent = `${summary.avgRelevance}%`;
  document.querySelector("#avgFaithfulnessMetric").textContent = `${summary.avgFaithfulness}%`;
  document.querySelector("#avgCitationMetric").textContent = `${summary.avgCitation}%`;
  document.querySelector("#highRiskMetric").textContent = String(summary.risk.high);
  document.querySelector("#trendTitle").textContent = `${metricLabel} trend`;
  document.querySelector("#trendSubtitle").textContent = `${filteredLabel} | latest 10 runs`;
  renderDashboardToggles();

  trendChart.innerHTML = summary.trend.length
    ? summary.trend
        .map(
          (item) => `
      <button class="trend-run ${item.risk}" data-history-index="${item.historyIndex}" title="${escapeHtml(item.question)}">
        <div class="trend-bars" aria-label="Q${item.index} ${metricLabel}">
          <span class="bar selected ${state.dashboardMetric}" style="height: ${item.selectedMetric}%"></span>
        </div>
        <strong>${item.selectedMetric}%</strong>
        <small>Q${item.index}</small>
      </button>`
        )
        .join("")
    : `<div class="dashboard-empty">No runs match this filter.</div>`;

  const total = Math.max(summary.count, 1);
  riskBreakdown.innerHTML = ["low", "medium", "high"]
    .map((risk) => {
      const count = summary.risk[risk];
      const width = Math.round((count / total) * 100);
      return `
      <button class="risk-row ${risk} ${state.dashboardRisk === risk ? "active" : ""}" data-risk="${risk}">
        <span>${risk}</span>
        <div><strong style="width: ${width}%"></strong></div>
        <b>${count}</b>
      </button>`;
    })
    .join("");

  dashboardInsight.innerHTML = `
    <strong>${summary.filteredCount ? `${summary.filteredCount} runs selected` : "No runs selected"}</strong>
    <span>${summary.insight}</span>
  `;

  reviewList.innerHTML = summary.needsReview.length
    ? summary.needsReview
        .map(
          (item) => `
      <button class="review-item" data-history-index="${item.historyIndex}">
        <span>${item.confidence}%</span>
        <strong>${escapeHtml(item.question)}</strong>
        <small>${item.reason}</small>
      </button>`
        )
        .join("")
    : `<div class="dashboard-empty">No weak answers detected yet.</div>`;
}

const metricLabels = {
  confidence: "Confidence",
  relevance: "Relevance",
  faithfulness: "Faithfulness",
  citation: "Citation",
};

function renderDashboardToggles() {
  document.querySelectorAll("[data-metric]").forEach((button) => {
    button.classList.toggle("active", button.dataset.metric === state.dashboardMetric);
  });
  document.querySelectorAll("[data-risk]").forEach((button) => {
    button.classList.toggle("active", button.dataset.risk === state.dashboardRisk);
  });
}

function buildEvaluationSummary() {
  const runs = state.history.map((item, index) => {
    const metrics = item.metrics || {};
    const relevance = Number(metrics.relevance || 0);
    const faithfulness = Number(metrics.faithfulness || 0);
    const citation = Number(metrics.citation || 0);
    const risk = String(metrics.risk || "High").toLowerCase();
    const confidence = Math.round((relevance + faithfulness + citation) / 3);
    return {
      index: index + 1,
      historyIndex: index,
      question: item.question || "Untitled question",
      relevance,
      faithfulness,
      citation,
      risk: ["low", "medium", "high"].includes(risk) ? risk : "high",
      confidence,
    };
  });

  const count = runs.length;
  const filteredRuns =
    state.dashboardRisk === "all" ? runs : runs.filter((item) => item.risk === state.dashboardRisk);
  const average = (field) =>
    filteredRuns.length ? Math.round(filteredRuns.reduce((sum, item) => sum + item[field], 0) / filteredRuns.length) : 0;
  const risk = {
    low: runs.filter((item) => item.risk === "low").length,
    medium: runs.filter((item) => item.risk === "medium").length,
    high: runs.filter((item) => item.risk === "high").length,
  };

  const needsReview = filteredRuns
    .filter((item) => item.confidence < 75 || item.risk !== "low")
    .sort((left, right) => left.confidence - right.confidence)
    .slice(0, 4)
    .map((item) => ({
      ...item,
      reason:
        item.risk === "high"
          ? "high hallucination risk"
          : item.risk === "medium"
          ? "medium risk"
          : "low confidence",
    }));

  const trend = filteredRuns.slice(-10).map((item) => ({
    ...item,
    selectedMetric: item[state.dashboardMetric],
  }));
  const insight = buildDashboardInsight(filteredRuns, state.dashboardMetric);

  return {
    count,
    filteredCount: filteredRuns.length,
    avgRelevance: average("relevance"),
    avgFaithfulness: average("faithfulness"),
    avgCitation: average("citation"),
    risk,
    trend,
    needsReview,
    insight,
  };
}

function buildDashboardInsight(runs, metric) {
  if (!runs.length) return "Change the risk filter or run more questions to populate this view.";
  const values = runs.map((item) => item[metric]);
  const average = Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
  const best = runs.reduce((left, right) => (right[metric] > left[metric] ? right : left), runs[0]);
  const weakest = runs.reduce((left, right) => (right[metric] < left[metric] ? right : left), runs[0]);
  return `${metricLabels[metric]} average is ${average}%. Best run is Q${best.index} at ${best[metric]}%; weakest run is Q${weakest.index} at ${weakest[metric]}%.`;
}

async function runBenchmarkSuite() {
  runBenchmarkButton.disabled = true;
  benchmarkSummary.textContent = "Running benchmark questions...";
  if (!state.documents.length) {
    await loadSampleDocuments();
  }
  state.selectedDocumentIds = state.documents.map((document) => document.id);
  renderDocuments();

  const results = [];
  for (const testCase of benchmarkQuestions) {
    const result = state.backend
      ? await runBackendBenchmarkCase(testCase)
      : runLocalBenchmarkCase(testCase);
    results.push(scoreBenchmarkCase(testCase, result));
  }
  state.benchmarkResults = results;
  renderBenchmarkResults();
  renderHistory();
  renderEvaluationDashboard();
  runBenchmarkButton.disabled = false;
}

async function loadSampleDocuments() {
  if (state.backend) {
    await apiFetch("/load-samples", { method: "POST", body: "{}" });
    await syncBackendState();
  } else {
    sampleDocs.forEach((doc) => addDocument(doc.name, doc.text));
  }
}

async function runBackendBenchmarkCase(testCase) {
  const result = await apiFetch("/ask", {
    method: "POST",
    body: JSON.stringify({
      question: testCase.question,
      provider: "template-llm",
      documentIds: state.documents.map((document) => document.id),
    }),
  });
  const metrics = {
    ...result.metrics,
    provider: result.provider,
    providerWarning: result.providerWarning,
  };
  renderAnswer(testCase.question, result.answer, result.evidence, metrics);
  await syncBackendState();
  return result;
}

function runLocalBenchmarkCase(testCase) {
  const startedAt = performance.now();
  const evidence = retrieve(testCase.question);
  const answer = buildAnswer(testCase.question, evidence);
  const metrics = evaluate(testCase.question, answer, evidence, Math.round(performance.now() - startedAt));
  const result = {
    question: testCase.question,
    answer,
    evidence,
    metrics,
    provider: "extractive",
    createdAt: new Date().toISOString(),
  };
  state.history.push(result);
  renderAnswer(testCase.question, answer, evidence, metrics);
  return result;
}

function scoreBenchmarkCase(testCase, result) {
  const answerText = `${result.answer || ""} ${result.evidence?.map((item) => item.text).join(" ") || ""}`.toLowerCase();
  const sourceHit = (result.evidence || []).some((item) => item.documentName === testCase.expectedSource);
  const matchedTerms = testCase.expectedTerms.filter((term) => answerText.includes(term.toLowerCase()));
  const termCoverage = Math.round((matchedTerms.length / testCase.expectedTerms.length) * 100);
  const faithfulness = Number(result.metrics?.faithfulness || 0);
  const passed = sourceHit && termCoverage >= 50 && faithfulness >= 55;
  return {
    question: testCase.question,
    expectedSource: testCase.expectedSource,
    matchedTerms,
    termCoverage,
    faithfulness,
    sourceHit,
    passed,
  };
}

function renderBenchmarkResults() {
  const total = state.benchmarkResults.length;
  const passed = state.benchmarkResults.filter((result) => result.passed).length;
  document.querySelector("#benchmarkStatus").textContent = total ? `${passed}/${total} pass` : "Not run";
  benchmarkSummary.textContent = total
    ? `${Math.round((passed / total) * 100)}% pass rate across source, term, and faithfulness checks.`
    : "Load sample papers, then run grounded retrieval checks.";
  benchmarkList.innerHTML = total
    ? state.benchmarkResults
        .map(
          (result, index) => `
      <article class="benchmark-item ${result.passed ? "pass" : "fail"}">
        <div>
          <span>${result.passed ? "Pass" : "Review"}</span>
          <strong>Q${index + 1}. ${escapeHtml(result.question)}</strong>
        </div>
        <dl>
          <dt>Source</dt><dd>${result.sourceHit ? "hit" : "miss"}</dd>
          <dt>Terms</dt><dd>${result.termCoverage}%</dd>
          <dt>Faithful</dt><dd>${result.faithfulness}%</dd>
        </dl>
        <small>Expected: ${escapeHtml(result.expectedSource)} | matched: ${escapeHtml(result.matchedTerms.join(", ") || "none")}</small>
      </article>`
        )
        .join("")
    : `<div class="dashboard-empty">No benchmark run yet.</div>`;
}

function setExportAvailability() {
  const hasAnswer = Boolean(state.currentAnswer);
  const hasSession = state.history.length > 0;
  exportAnswerButton.disabled = !hasAnswer;
  exportSessionButton.disabled = !hasSession;
  printButton.disabled = !hasAnswer;
}

function exportLatestAnswer() {
  if (!state.currentAnswer) return;
  const content = buildAnswerMarkdown(state.currentAnswer);
  setExportPreview(content, "Answer ready");
  downloadMarkdown("scholarrag-answer.md", content);
}

function exportSession() {
  if (!state.history.length) return;
  const content = [
    "# ScholarRAG Research Session",
    "",
    `Exported: ${new Date().toLocaleString()}`,
    "",
    "## Indexed Documents",
    "",
    ...state.documents.map((document) => `- ${document.name} (${document.chars.toLocaleString()} chars)`),
    "",
    "## Questions",
    "",
    ...state.history.map((item, index) => buildAnswerMarkdown(item, index + 1)),
  ].join("\n");
  setExportPreview(content, "Session ready");
  downloadMarkdown("scholarrag-session.md", content);
}

function buildAnswerMarkdown(item, index = null) {
  const metrics = item.metrics || {};
  const evidence = item.evidence || [];
  const heading = index ? `## ${index}. ${item.question}` : `# ${item.question}`;
  const provider = item.provider || metrics.provider || "extractive";
  return [
    heading,
    "",
    `Provider: ${provider}`,
    `Relevance: ${metrics.relevance ?? 0}%`,
    `Faithfulness: ${metrics.faithfulness ?? 0}%`,
    `Citation coverage: ${metrics.citation ?? 0}%`,
    `Hallucination risk: ${metrics.risk || "Unknown"}`,
    "",
    "### Answer",
    "",
    item.answer || "",
    "",
    "### Sources",
    "",
    ...(evidence.length
      ? evidence.map(
          (chunk, sourceIndex) =>
            `${sourceIndex + 1}. ${chunk.documentName} - page ${chunk.page}, chunk ${chunk.index}`
        )
      : ["No supporting source found."]),
    "",
    "### Retrieved Evidence",
    "",
    ...(evidence.length
      ? evidence.map(
          (chunk, sourceIndex) =>
            `**[${sourceIndex + 1}] ${chunk.documentName}**\n\n${chunk.text}`
        )
      : ["No retrieved evidence."]),
    "",
  ].join("\n");
}

function downloadMarkdown(fileName, content) {
  if (typeof Blob === "undefined" || typeof URL === "undefined" || !URL.createObjectURL) {
    exportStatus.textContent = "Preview only";
    return;
  }
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function setExportPreview(content, status) {
  exportPreview.value = content;
  exportStatus.textContent = status;
}

fileInput.addEventListener("change", async (event) => {
  const files = Array.from(event.target.files || []);
  for (const file of files) {
    if (state.backend) {
      try {
        const contentBase64 = await fileToBase64(file);
        await apiFetch("/upload", {
          method: "POST",
          body: JSON.stringify({ name: file.name, contentBase64 }),
        });
      } catch (error) {
        answerBox.textContent = error.message;
      }
    } else {
      if (file.name.toLowerCase().endsWith(".pdf")) {
        answerBox.textContent = "PDF upload needs backend mode. Start server.py, then reload this page.";
      } else {
        const text = await file.text();
        addDocument(file.name, text);
      }
    }
  }
  if (state.backend) await syncBackendState();
  fileInput.value = "";
});

loadSampleButton.addEventListener("click", async () => {
  if (state.backend) {
    await apiFetch("/load-samples", { method: "POST", body: "{}" });
    await syncBackendState();
  } else {
    sampleDocs.forEach((doc) => addDocument(doc.name, doc.text));
  }
});

askButton.addEventListener("click", askQuestion);

exampleButton.addEventListener("click", () => {
  questionInput.value = "What makes a RAG system faithful and useful for PhD research?";
  askQuestion();
});

resetButton.addEventListener("click", async () => {
  if (state.backend) {
    await apiFetch("/reset", { method: "POST", body: "{}" });
  }
  state.documents = [];
  state.chunks = [];
  state.chunkCount = 0;
  state.history = [];
  state.avgConfidence = 0;
  state.currentAnswer = null;
  state.selectedDocumentIds = [];
  state.benchmarkResults = [];
  questionInput.value = "";
  evidenceList.innerHTML = "";
  renderCitationExplorer();
  answerBox.textContent = "Upload documents or load samples, then ask a question. Answers will cite the chunks used.";
  document.querySelector("#retrievalStatus").textContent = "Idle";
  document.querySelector("#latencyMetric").textContent = "0 ms";
  document.querySelector("#relevanceScore").textContent = "0%";
  document.querySelector("#faithfulnessScore").textContent = "0%";
  document.querySelector("#citationScore").textContent = "0%";
  document.querySelector("#riskScore").textContent = "Low";
  render();
});

runBenchmarkButton.addEventListener("click", () => {
  runBenchmarkSuite().catch((error) => {
    benchmarkSummary.textContent = error.message;
    runBenchmarkButton.disabled = false;
  });
});

documentList.addEventListener("change", (event) => {
  const checkbox = event.target.closest("[data-document-id]");
  if (!checkbox) return;
  const documentId = checkbox.dataset.documentId;
  if (checkbox.checked) {
    state.selectedDocumentIds = Array.from(new Set([...state.selectedDocumentIds, documentId]));
  } else {
    state.selectedDocumentIds = state.selectedDocumentIds.filter((id) => id !== documentId);
  }
  renderDocuments();
});

selectAllDocsButton.addEventListener("click", () => {
  state.selectedDocumentIds = state.documents.map((document) => document.id);
  renderDocuments();
});

historyList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-history-index]");
  if (!item) return;
  const historyItem = state.history[Number(item.dataset.historyIndex)];
  if (!historyItem) return;
  questionInput.value = historyItem.question;
  const metrics = {
    ...historyItem.metrics,
    provider: historyItem.provider || historyItem.metrics?.provider,
    providerWarning: historyItem.providerWarning || historyItem.metrics?.providerWarning,
  };
  renderAnswer(historyItem.question, historyItem.answer, historyItem.evidence || [], metrics);
});

citationMap.addEventListener("click", (event) => {
  const item = event.target.closest("[data-source-index]");
  if (!item) return;
  const sourceIndex = Number(item.dataset.sourceIndex);
  const evidenceItem = document.querySelector(`[data-evidence-index="${sourceIndex}"]`);
  if (evidenceItem) {
    evidenceItem.scrollIntoView({ behavior: "smooth", block: "center" });
    evidenceItem.classList.add("evidence-focus");
    window.setTimeout(() => evidenceItem.classList.remove("evidence-focus"), 1200);
  }
});

reviewList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-history-index]");
  if (!item) return;
  const historyItem = state.history[Number(item.dataset.historyIndex)];
  if (!historyItem) return;
  questionInput.value = historyItem.question;
  const metrics = {
    ...historyItem.metrics,
    provider: historyItem.provider || historyItem.metrics?.provider,
    providerWarning: historyItem.providerWarning || historyItem.metrics?.providerWarning,
  };
  renderAnswer(historyItem.question, historyItem.answer, historyItem.evidence || [], metrics);
});

trendChart.addEventListener("click", (event) => {
  const item = event.target.closest("[data-history-index]");
  if (!item) return;
  const historyItem = state.history[Number(item.dataset.historyIndex)];
  if (!historyItem) return;
  questionInput.value = historyItem.question;
  const metrics = {
    ...historyItem.metrics,
    provider: historyItem.provider || historyItem.metrics?.provider,
    providerWarning: historyItem.providerWarning || historyItem.metrics?.providerWarning,
  };
  renderAnswer(historyItem.question, historyItem.answer, historyItem.evidence || [], metrics);
});

riskBreakdown.addEventListener("click", (event) => {
  const item = event.target.closest("[data-risk]");
  if (!item) return;
  state.dashboardRisk = state.dashboardRisk === item.dataset.risk ? "all" : item.dataset.risk;
  renderEvaluationDashboard();
});

document.querySelectorAll(".metric-toggle").forEach((button) => {
  button.addEventListener("click", () => {
    state.dashboardMetric = button.dataset.metric;
    renderEvaluationDashboard();
  });
});

document.querySelectorAll(".risk-toggle").forEach((button) => {
  button.addEventListener("click", () => {
    state.dashboardRisk = button.dataset.risk;
    renderEvaluationDashboard();
  });
});

exportAnswerButton.addEventListener("click", exportLatestAnswer);
exportSessionButton.addEventListener("click", exportSession);
printButton.addEventListener("click", () => window.print());

detectBackend().then(render);
