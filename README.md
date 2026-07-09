# ScholarRAG

ScholarRAG is a local-first RAG research assistant MVP. It lets a user upload text-based research notes, indexes the content into chunks, retrieves relevant evidence for a question, and returns a grounded answer with citations and quality metrics.

## Current MVP

- Upload `.pdf`, `.txt`, `.md`, `.csv`, and `.json` research notes.
- Split documents into citation-ready chunks.
- Retrieve relevant context using a lightweight TF-IDF style local retriever.
- Generate extractive grounded answers.
- Show source citations, evidence chunks, relevance, faithfulness, citation coverage, latency, and hallucination risk.
- Optional backend mode adds persistent indexing and real PDF text extraction through `pypdf`.
- Backend retrieval now uses hybrid scoring: keyword relevance plus local hashing embeddings and cosine similarity.
- Answer generation supports provider modes: extractive, local grounded template, and optional Ollama.
- Session history lets users revisit previous questions and export answers or full research sessions to Markdown.
- Evaluation dashboard tracks relevance, faithfulness, citation coverage, hallucination risk, and weak answers.
- Document filters and citation explorer make retrieval source coverage inspectable.
- Benchmark suite runs regression questions against expected sources, terms, and faithfulness thresholds.

## Why this project is strong

This is not just a chatbot UI. It demonstrates the full RAG architecture:

```text
Document upload -> extraction -> chunking -> embedding/retrieval -> grounded answer -> citations -> evaluation
```

The current app runs without paid APIs so the architecture is visible and easy to demo. The next production step is to replace local TF-IDF retrieval with embeddings and a vector database.

## Run locally

### Browser-only mode

From this folder:

```bash
python -m http.server 8770
```

Open:

```text
http://127.0.0.1:8770/index.html
```

### Backend mode with PDF extraction

Use the bundled Codex Python runtime or any Python environment with `pypdf` installed:

```bash
python server.py
```

Open:

```text
http://127.0.0.1:8780/index.html
```

Backend API:

```text
GET  /api/health
GET  /api/documents
POST /api/upload
POST /api/load-samples
POST /api/ask
POST /api/reset
```

Smoke test:

```bash
python tests/smoke_test.py
```

## Retrieval Mode

The backend uses a dependency-light semantic retrieval layer:

```text
text -> token features -> hashing embedding -> cosine similarity
question -> hashing embedding -> hybrid retrieval
hybrid score = keyword score + vector similarity
```

This is designed as a local stand-in for production embeddings. The next upgrade can swap this layer with BGE/E5/OpenAI embeddings and Qdrant/Chroma without changing the UI contract.

## Answer Providers

```text
extractive     -> safest default; composes an answer from retrieved sentences
template-llm   -> local grounded LLM-style answer without external calls
ollama         -> optional local LLM at http://127.0.0.1:11434
```

If Ollama is unavailable, ScholarRAG falls back to the local grounded template provider and shows a warning.

## Research Session Export

ScholarRAG keeps the latest questions in session history and can export:

```text
single answer -> question, answer, provider, metrics, citations, retrieved evidence
full session  -> indexed documents plus every saved Q/A turn
Print/PDF     -> browser print flow for a clean report-style output
```

## Evaluation Dashboard

The monitoring panel turns every Q/A run into a lightweight RAG evaluation record:

```text
quality trend   -> relevance, faithfulness, citation bars across latest runs
risk breakdown  -> low, medium, high hallucination-risk counts
needs review    -> lowest-confidence or non-low-risk answers
interaction     -> metric toggles, risk filters, clickable runs, and dashboard insights
```

This makes the project feel closer to a real AI system where teams track answer quality, not just chatbot output.

## Document Filters and Citation Explorer

Users can select which indexed documents should participate in retrieval before asking a question. After an answer is generated, the citation explorer groups retrieved chunks by source document so reviewers can inspect coverage and jump back to the matching evidence cards.

## Benchmark Suite

The app includes a small regression suite for the sample research corpus:

```text
source hit      -> retrieved evidence includes the expected document
term coverage   -> answer/evidence includes expected grounding terms
faithfulness    -> evaluator score stays above a minimum threshold
```

This gives the project a practical quality gate for future retrieval, reranking, or model-provider changes.

## Production Upgrade Plan

1. Add a FastAPI backend.
2. Add PDF extraction with PyMuPDF or pdfplumber.
3. Replace local hashing embeddings with BGE, E5, sentence-transformers, or OpenAI embeddings.
4. Store vectors in ChromaDB or Qdrant instead of `data/index.json`.
5. Add reranking and hybrid search.
6. Add local LLM support through Ollama plus cloud model support.
7. Add persistent document storage and user accounts.
8. Add eval datasets and regression tests for RAG quality.
