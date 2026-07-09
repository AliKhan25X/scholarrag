import base64
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.llm_providers import generate_answer


STOP_WORDS = set(
    "a an and are as at be by for from has have how in into is it its of on or our that the their this to was were what when where which who why with your".split()
)
EMBEDDING_DIMENSIONS = 256
HYBRID_KEYWORD_WEIGHT = 0.58
HYBRID_VECTOR_WEIGHT = 0.42


@dataclass
class Document:
    id: str
    name: str
    chars: int
    uploaded_at: float


@dataclass
class Chunk:
    id: str
    document_id: str
    document_name: str
    page: int
    index: int
    text: str
    terms: dict
    embedding: list[float] | None = None


class ScholarRAGEngine:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "index.json"
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.history: list[dict] = []
        self.load()

    def load(self):
        if not self.index_path.exists():
            return
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.documents = [Document(**item) for item in data.get("documents", [])]
        self.chunks = [Chunk(**item) for item in data.get("chunks", [])]
        changed = False
        for chunk in self.chunks:
            if not chunk.embedding:
                chunk.embedding = embed_text(chunk.text)
                changed = True
        self.history = data.get("history", [])
        if changed:
            self.save()

    def save(self):
        data = {
            "documents": [asdict(doc) for doc in self.documents],
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "history": self.history[-100:],
        }
        self.index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def reset(self):
        self.documents = []
        self.chunks = []
        self.history = []
        self.save()

    def add_file(self, name: str, content_base64: str):
        raw = base64.b64decode(content_base64)
        text = extract_text(name, raw)
        return self.add_document(name, text)

    def add_document(self, name: str, text: str):
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            raise ValueError(f"No extractable text found in {name}")
        document_id = f"doc-{int(time.time() * 1000)}-{len(self.documents) + 1}"
        document = Document(document_id, name, len(cleaned), time.time())
        chunks = chunk_text(cleaned, document_id, name)
        self.documents.append(document)
        self.chunks.extend(chunks)
        self.save()
        return {"document": asdict(document), "chunks_added": len(chunks)}

    def load_samples(self):
        samples = [
            (
                "retrieval_augmented_generation_notes.txt",
                "Retrieval augmented generation improves factual grounding by searching a knowledge base before generation. A RAG system usually has document ingestion, text chunking, embedding generation, vector search, prompt construction, answer generation, and citation display. The main contribution is that the language model answers from retrieved evidence instead of relying only on memorized parameters. Strong RAG systems add reranking, hybrid search, metadata filtering, and evaluation for faithfulness.",
            ),
            (
                "llm_evaluation_playbook.txt",
                "LLM evaluation measures whether a model answer is relevant, faithful, concise, and supported by citations. Production teams track latency, cost, token usage, retrieval precision, citation coverage, and hallucination risk. A practical dashboard should show question history, retrieved chunks, model output, evaluator scores, and failure categories. Faithfulness means the answer is supported by the retrieved context.",
            ),
            (
                "phd_research_assistant_architecture.txt",
                "A PhD research assistant can read papers, summarize methods, compare contributions, and find gaps. The system should preserve page numbers and section names so every answer has a source. The best user experience includes multi-document chat, export to markdown, document filters, highlighted citations, and a research trend dashboard. Privacy can be improved by using local embeddings and local LLMs.",
            ),
        ]
        added = []
        existing = {doc.name for doc in self.documents}
        for name, text in samples:
            if name not in existing:
                added.append(self.add_document(name, text))
        return {"added": len(added), "documents": self.snapshot()["documents"]}

    def ask(self, question: str, provider: str = "extractive", document_ids: list[str] | None = None):
        started = time.perf_counter()
        evidence = self.retrieve(question, document_ids=document_ids)
        provider_result = generate_answer(question, evidence, provider)
        answer = provider_result.text or build_answer(question, evidence, bool(self.chunks))
        latency = round((time.perf_counter() - started) * 1000)
        metrics = evaluate(question, answer, evidence, latency)
        result = {
            "question": question,
            "answer": answer,
            "evidence": [chunk_to_public(chunk) for chunk in evidence],
            "metrics": metrics,
            "provider": provider_result.get("provider", "extractive"),
            "providerMode": provider_result.get("mode", "extractive"),
            "providerWarning": provider_result.get("warning", ""),
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.history.append(result)
        self.save()
        return result

    def retrieve(self, question: str, limit: int = 4, document_ids: list[str] | None = None):
        query_tokens = tokenize(question)
        query_embedding = embed_text(question)
        allowed_documents = set(document_ids or [])
        ranked = []
        for chunk in self.chunks:
            if allowed_documents and chunk.document_id not in allowed_documents:
                continue
            keyword_score = 0.0
            for term in query_tokens:
                if term in chunk.terms:
                    keyword_score += chunk.terms[term] * self.idf(term)
            vector_score = cosine_similarity(query_embedding, chunk.embedding or embed_text(chunk.text))
            score = HYBRID_KEYWORD_WEIGHT * normalize_keyword_score(keyword_score) + HYBRID_VECTOR_WEIGHT * vector_score
            if score > 0.08:
                ranked.append((score, keyword_score, vector_score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        output = []
        for score, keyword_score, vector_score, chunk in ranked[:limit]:
            public_chunk = Chunk(**asdict(chunk))
            public_chunk.terms = dict(public_chunk.terms)
            public_chunk.terms["_score"] = score
            public_chunk.terms["_keyword_score"] = keyword_score
            public_chunk.terms["_vector_score"] = vector_score
            output.append(public_chunk)
        return output

    def idf(self, term: str):
        docs_with_term = sum(1 for chunk in self.chunks if term in chunk.terms)
        return math.log((len(self.chunks) + 1) / (docs_with_term + 1)) + 1

    def snapshot(self):
        avg_confidence = 0
        if self.history:
            avg_confidence = round(
                sum((item["metrics"]["relevance"] + item["metrics"]["faithfulness"]) / 2 for item in self.history)
                / len(self.history)
            )
        return {
            "documents": [asdict(doc) for doc in self.documents],
            "chunks": len(self.chunks),
            "history": self.history[-20:],
            "avg_confidence": avg_confidence,
            "retrieval_mode": "hybrid-keyword-vector",
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        }


def extract_text(name: str, raw: bytes):
    lower = name.lower()
    if lower.endswith(".pdf") or raw.startswith(b"%PDF"):
        return extract_pdf_text(raw)
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unsupported file encoding for {name}")


def extract_pdf_text(raw: bytes):
    try:
        from pypdf import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(raw))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[Page {index}] {page_text}")
        text = "\n".join(pages).strip()
        if text:
            return text
    except Exception as exc:
        raise ValueError(f"PDF text extraction failed: {exc}") from exc
    raise ValueError("PDF has no extractable text")


def tokenize(text: str):
    return [
        token
        for token in re.sub(r"[^a-z0-9\s-]", " ", text.lower()).split()
        if len(token) > 2 and token not in STOP_WORDS
    ]


def term_frequency(tokens):
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def stable_hash(text: str):
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def token_features(tokens: list[str]):
    for token in tokens:
        yield token
        if len(token) > 4:
            yield token[:4]
            yield token[-4:]
        for index in range(max(0, len(token) - 2)):
            yield token[index : index + 3]


def embed_text(text: str, dimensions: int = EMBEDDING_DIMENSIONS):
    vector = [0.0] * dimensions
    for feature in token_features(tokenize(text)):
        hashed = stable_hash(feature)
        index = hashed % dimensions
        sign = 1.0 if hashed & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def cosine_similarity(left: list[float], right: list[float]):
    if not left or not right:
        return 0.0
    return max(0.0, sum(a * b for a, b in zip(left, right)))


def normalize_keyword_score(score: float):
    return score / (score + 3.0) if score > 0 else 0.0


def split_sentences(text: str):
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]


def chunk_text(text: str, document_id: str, document_name: str):
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(tokenize(sentence))
        if current and current_length + sentence_length > 110:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        current.append(sentence)
        current_length += sentence_length
    if current:
        chunks.append(" ".join(current))
    return [
        Chunk(
            id=f"{document_id}-{index + 1}",
            document_id=document_id,
            document_name=document_name,
            page=index // 3 + 1,
            index=index + 1,
            text=chunk,
            terms=term_frequency(tokenize(chunk)),
            embedding=embed_text(chunk),
        )
        for index, chunk in enumerate(chunks)
    ]


def build_answer(question: str, evidence: list[Chunk], has_documents: bool):
    if not has_documents:
        return "No documents are indexed yet. Upload research notes or load the sample papers first."
    if not evidence:
        return "I could not find enough grounded evidence in the indexed documents. Try asking with terms that appear in the papers."
    query_terms = set(tokenize(question))
    selected = []
    for chunk in evidence:
        for sentence in split_sentences(chunk.text):
            overlap = sum(1 for token in tokenize(sentence) if token in query_terms)
            if overlap:
                selected.append((overlap, sentence))
    selected.sort(reverse=True, key=lambda item: item[0])
    sentences = [sentence for _, sentence in selected[:4]] or [chunk.text for chunk in evidence[:2]]
    return " ".join(sentences)


def evaluate(question: str, answer: str, evidence: list[Chunk], latency: int):
    query_tokens = tokenize(question)
    answer_tokens = tokenize(answer)
    evidence_tokens = set(tokenize(" ".join(chunk.text for chunk in evidence)))
    query_matches = sum(1 for token in query_tokens if token in answer_tokens)
    supported = sum(1 for token in answer_tokens if token in evidence_tokens)
    relevance = min(98, round((query_matches / max(len(query_tokens), 1)) * 100))
    faithfulness = min(99, round((supported / max(len(answer_tokens), 1)) * 100))
    citation = min(96, 60 + len(evidence) * 9) if evidence else 0
    risk = "Low" if faithfulness > 80 else "Medium" if faithfulness > 55 else "High"
    return {
        "relevance": relevance,
        "faithfulness": faithfulness,
        "citation": citation,
        "risk": risk,
        "latency": latency,
    }


def chunk_to_public(chunk: Chunk):
    return {
        "id": chunk.id,
        "documentId": chunk.document_id,
        "documentName": chunk.document_name,
        "page": chunk.page,
        "index": chunk.index,
        "text": chunk.text,
        "score": float(chunk.terms.get("_score", 0.0)),
        "keywordScore": float(chunk.terms.get("_keyword_score", 0.0)),
        "vectorScore": float(chunk.terms.get("_vector_score", 0.0)),
    }
