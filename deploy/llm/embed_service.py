#!/usr/bin/env python3
"""Plenarsuche, query embeddings on the GPU machine: jina-embeddings-v4, loopback only.

The public VM sends a query and receives its 2048-d vector through the reverse tunnel, so a search no
longer depends on an external embedding API. Same model and settings as the local backend in
backend/main.py (retrieval adapter, query prompt, fp16, L2-normalised). Standard library HTTP server.

  POST /embed   {"query": "..."}   ->  {"vector": [...], "dim": 2048}
  GET  /health                     ->  {"ok": true}
Every request needs `Authorization: Bearer <key>` (PLENARSUCHE_KEY_FILE, default
~/.config/secrets/plenarsuche_llm.key).
"""
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = os.environ.get("PARLIAMENT_EMBED_MODEL", "jinaai/jina-embeddings-v4")
PORT = int(os.environ.get("PORT", "18101"))
MAX_CHARS = 2000
KEY = Path(os.environ.get("PLENARSUCHE_KEY_FILE", Path.home() / ".config/secrets/plenarsuche_llm.key")).read_text().strip()

model = SentenceTransformer(MODEL, trust_remote_code=True, device="cuda")
model.max_seq_length = 512
model.half()                       # the document vectors were produced in fp16
lock = threading.Lock()            # one query at a time on the GPU; a query takes tens of milliseconds


def encode(q: str) -> np.ndarray:
    with lock:
        v = model.encode([q], task="retrieval", prompt_name="query", convert_to_numpy=True,
                         normalize_embeddings=True, batch_size=1, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)[0]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):     # never log queries
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self):
        got = self.headers.get("Authorization", "")
        return hmac.compare_digest(got.encode(), f"Bearer {KEY}".encode())

    def do_GET(self):
        if not self._authorised():
            return self._send(401, {"error": "unauthorised"})
        self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorised():
            return self._send(401, {"error": "unauthorised"})
        if self.path != "/embed":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 8192:
                return self._send(413, {"error": "too large"})
            q = str(json.loads(self.rfile.read(n) or b"{}").get("query", "")).strip()[:MAX_CHARS]
            if not q:
                return self._send(422, {"error": "empty query"})
            v = encode(q)
            self._send(200, {"vector": v.tolist(), "dim": int(v.shape[0])})
        except Exception as e:
            self._send(500, {"error": type(e).__name__})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"embed service on 127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()
