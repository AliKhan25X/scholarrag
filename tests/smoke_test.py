import base64
import sys
import tempfile
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.rag_engine import ScholarRAGEngine


def make_pdf_bytes():
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 760, "ScholarRAG uses retrieval augmented generation for faithful research answers.")
    pdf.drawString(72, 736, "The system stores citations, retrieved chunks, latency, and hallucination risk.")
    pdf.save()
    return buffer.getvalue()


def test_pdf_ingestion_and_question_answering():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ScholarRAGEngine(tmpdir)
        pdf_b64 = base64.b64encode(make_pdf_bytes()).decode("ascii")
        result = engine.add_file("scholarrag_test.pdf", pdf_b64)
        assert result["chunks_added"] >= 1
        second = engine.add_document(
            "unrelated_notes.txt",
            "Graph neural networks can model molecules, social networks, and citation graphs.",
        )

        answer = engine.ask("What does ScholarRAG use for faithful research answers?")
        assert answer["evidence"]
        assert "retrieval" in answer["answer"].lower()
        assert answer["evidence"][0]["vectorScore"] > 0
        assert answer["provider"] == "extractive"
        assert answer["createdAt"]
        assert answer["metrics"]["faithfulness"] > 50

        llm_answer = engine.ask("Why is citation tracking useful?", provider="template-llm")
        assert llm_answer["provider"] == "template-llm"
        assert "[source" in llm_answer["answer"]

        filtered_answer = engine.ask(
            "What can graph neural networks model?",
            document_ids=[second["document"]["id"]],
        )
        assert filtered_answer["evidence"]
        assert filtered_answer["evidence"][0]["documentName"] == "unrelated_notes.txt"
        assert len(engine.snapshot()["history"]) == 3


if __name__ == "__main__":
    test_pdf_ingestion_and_question_answering()
    print("ScholarRAG smoke test passed")
