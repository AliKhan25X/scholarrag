import json
import urllib.error
import urllib.request


class LLMResult(dict):
    @property
    def text(self):
        return self["text"]


def generate_answer(question: str, evidence: list, provider: str = "extractive"):
    provider = (provider or "extractive").lower()
    context = build_context(evidence)
    if provider == "template-llm":
        return template_llm_answer(question, evidence)
    if provider == "ollama":
        try:
            return ollama_answer(question, context)
        except Exception as exc:
            fallback = template_llm_answer(question, evidence)
            fallback["provider"] = "template-llm"
            fallback["warning"] = f"Ollama unavailable, used local fallback: {exc}"
            return fallback
    return LLMResult({"text": "", "provider": "extractive", "mode": "extractive"})


def build_context(evidence: list):
    lines = []
    for index, chunk in enumerate(evidence, start=1):
        lines.append(f"[{index}] {chunk.document_name} page {chunk.page}: {chunk.text}")
    return "\n".join(lines)


def template_llm_answer(question: str, evidence: list):
    if not evidence:
        return LLMResult(
            {
                "text": "I could not find grounded context for this question. Add more documents or ask using terms present in the indexed sources.",
                "provider": "template-llm",
                "mode": "local-grounded-template",
            }
        )

    opening = "Based on the retrieved research context, "
    focus = infer_focus(question)
    bullets = []
    for index, chunk in enumerate(evidence[:3], start=1):
        sentence = first_relevant_sentence(chunk.text, question)
        bullets.append(f"{index}. {sentence} [source {index}]")
    answer = opening + focus + "\n\n" + "\n".join(bullets)
    answer += "\n\nThe answer is grounded only in the retrieved chunks, so unsupported claims should be treated as unknown."
    return LLMResult({"text": answer, "provider": "template-llm", "mode": "local-grounded-template"})


def infer_focus(question: str):
    lowered = question.lower()
    if "faithful" in lowered or "hallucination" in lowered:
        return "faithfulness comes from grounding the answer in retrieved evidence and showing citations."
    if "architecture" in lowered or "system" in lowered:
        return "the system is best understood as an ingestion, retrieval, answer-generation, and evaluation pipeline."
    if "improve" in lowered or "benefit" in lowered:
        return "the main benefit is better factual grounding and easier verification through source-linked evidence."
    return "the most relevant points are the following."


def first_relevant_sentence(text: str, question: str):
    question_terms = {term for term in tokenize(question) if len(term) > 3}
    sentences = [item.strip() for item in text.replace("\n", " ").split(".") if item.strip()]
    if not sentences:
        return text[:240]
    ranked = []
    for sentence in sentences:
        overlap = sum(1 for term in tokenize(sentence) if term in question_terms)
        ranked.append((overlap, sentence))
    ranked.sort(reverse=True, key=lambda item: item[0])
    return ranked[0][1].strip() + "."


def tokenize(text: str):
    return [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split()]


def ollama_answer(question: str, context: str):
    prompt = (
        "You are ScholarRAG, a careful research assistant. Answer only from the provided context. "
        "Cite sources like [1], [2]. If the context is insufficient, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    payload = json.dumps(
        {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama is not reachable at 127.0.0.1:11434") from exc
    text = (data.get("response") or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return LLMResult({"text": text, "provider": "ollama", "mode": "local-llm"})
