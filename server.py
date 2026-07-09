from __future__ import annotations

import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from backend.rag_engine import ScholarRAGEngine


ROOT = Path(__file__).resolve().parent
ENGINE = ScholarRAGEngine(ROOT / "data")


class ScholarRAGHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self.write_json({"ok": True, "mode": "backend", "pdf": True})
        if path == "/api/documents":
            return self.write_json(ENGINE.snapshot())
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/reset":
                ENGINE.reset()
                return self.write_json(ENGINE.snapshot())
            if path == "/api/load-samples":
                return self.write_json(ENGINE.load_samples())
            if path == "/api/upload":
                payload = self.read_json()
                result = ENGINE.add_file(payload["name"], payload["contentBase64"])
                return self.write_json(result)
            if path == "/api/ask":
                payload = self.read_json()
                question = (payload.get("question") or "").strip()
                if not question:
                    raise ValueError("Question is required")
                return self.write_json(
                    ENGINE.ask(question, payload.get("provider", "extractive"), payload.get("documentIds") or [])
                )
            self.send_error(404, "Unknown API endpoint")
        except Exception as exc:
            self.write_json({"error": str(exc)}, status=400)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def write_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path):
        if path.endswith(".js"):
            return "text/javascript"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8780), ScholarRAGHandler)
    print("ScholarRAG backend running at http://127.0.0.1:8780")
    server.serve_forever()


if __name__ == "__main__":
    main()
