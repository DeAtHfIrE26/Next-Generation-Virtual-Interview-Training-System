"""The model downloader resumes after a dropped connection instead of failing the install."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from interview_core.realtime.assets import fetch

BODY = bytes(range(256)) * 4096  # 1 MiB


class FlakyRangeServer(BaseHTTPRequestHandler):
    drops = 2  # the first two responses are cut off part-way

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.removeprefix("bytes=").split("-")[0])
        body = BODY[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if FlakyRangeServer.drops:
            FlakyRangeServer.drops -= 1
            self.wfile.write(body[: len(body) // 3])
            self.wfile.flush()
            self.connection.close()  # short read on the client
            return
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_fetch_resumes_after_dropped_connections(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FlakyRangeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        dest = tmp_path / "model.bin"
        fetch(f"http://127.0.0.1:{srv.server_port}/model.bin", dest, backoff_s=0.01)
        assert dest.read_bytes() == BODY
        assert FlakyRangeServer.drops == 0
    finally:
        srv.shutdown()
