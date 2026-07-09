# ScholarRAG Architecture

## Goal

Build a research assistant that answers questions from uploaded documents and always shows the evidence used.

## MVP Architecture

```text
Browser UI
  |
  | upload text documents
  v
Document Indexer
  |
  | clean text
  v
Chunker
  |
  | 1 chunk ~= several sentences
  v
Local Retriever
  |
  | TF-IDF style score
  v
Evidence Builder
  |
  | top ranked chunks
  v
Extractive Answer Generator
  |
  | cited answer
  v
Evaluation Dashboard
```

## Backend Mode Added

```text
Browser UI
  |
  | JSON API
  v
Stdlib Python HTTP Server
  |
  | upload .pdf/.txt/.md/.csv/.json
  v
PDF/Text Extractor
  |
  | pypdf for PDFs
  v
Persistent RAG Engine
  |
  | local hashing embeddings + data/index.json
  v
Ask API + Evidence + Metrics
```

## Retrieval Layer

Current backend retrieval is hybrid:

```text
Query
  |
  | tokenize
  | hashing embedding
  v
Hybrid Retriever
  |-- keyword TF-IDF score
  |-- vector cosine score
  v
Ranked evidence chunks
```

This approximates semantic retrieval without downloading models. It keeps the portfolio deployable while showing the same shape as a production vector-search system.

## Answer Provider Layer

```text
Retrieved evidence
  |
  | provider = extractive | template-llm | ollama
  v
Grounded answer generator
  |
  | provider metadata + warning
  v
Answer + citations + evaluation
```

The provider contract is intentionally small so OpenAI-compatible APIs, Ollama, vLLM, or Hugging Face inference can be plugged in later.

## Production Architecture

```text
Frontend: Next.js
Backend: FastAPI
Document parsing: PyMuPDF / pdfplumber
Chunking: semantic text splitter
Embeddings: BGE / E5 / OpenAI embeddings
Vector DB: Qdrant / ChromaDB
LLM: Ollama local models / OpenAI / Hugging Face
Evaluation: faithfulness, answer relevance, citation coverage, latency
Deployment: Docker + Render/Railway/Fly.io
```

## Data Model

```text
documents
- id
- filename
- uploaded_at
- total_pages
- status

chunks
- id
- document_id
- chunk_index
- page_number
- text
- embedding_id

queries
- id
- question
- answer
- sources
- relevance_score
- faithfulness_score
- latency_ms
- created_at
```

## Key Portfolio Features

- Multi-document research chat
- Evidence-first answer generation
- Citation display
- RAG quality dashboard
- Local-first privacy story
- Clear upgrade path to real embeddings and vector DB
