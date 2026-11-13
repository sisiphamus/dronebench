"""Live dashboard server (stdlib only). Read-only over a workspace folder.

    dronebench serve WS [--port 8765]

  GET /                       the dashboard
  GET /vendor/...             three.js (vendored, works offline)
  GET /api/state              meta + every run record (poll this; carries a version string)
  GET /api/step/N             that step's full report + ops
  GET /api/step/N/model.glb   that step's 3D model (built once, cached in the step folder)

Everything the dashboard shows comes from files the CLI wrote, so an agent can iterate in one
terminal while this runs in another. Binds to 127.0.0.1 only.
"""
from __future__ import annotations

import json
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import workspace as W

VIEWER = Path(__file__).resolve().parent / "viewer"
_glb_lock = threading.Lock()


def _mtime(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


def state(ws: Path) -> dict:
    m = W.meta(ws)
    runs = W._runs(ws)
    version = f"{_mtime(ws / 'workspace.json')}-{_mtime(ws / 'runs.jsonl')}"
    return {"version": version, "meta": m, "runs": runs, "name": ws.name}


def step_payload(ws: Path, n: int) -> dict | None:
    rep = W._read_json(ws / "steps" / str(n) / "report.json")
    ops = W._read_json(ws / "steps" / str(n) / "ops.json", [])
    if rep is None:
        return {"step": n, "report": None, "ops": ops, "has_design": (ws / "steps" / str(n) / "design").is_dir()}
    return {"step": n, "report": rep, "ops": ops, "has_design": (ws / "steps" / str(n) / "design").is_dir()}


def step_glb(ws: Path, n: int) -> bytes | None:
    sdir = ws / "steps" / str(n)
    cache = sdir / "model.glb"
    if cache.exists():
        return cache.read_bytes()
    if not (sdir / "design").is_dir():
        return None
    with _glb_lock:
        if cache.exists():
            return cache.read_bytes()
        from .ingest import load_design
        from .viewer import design_glb
        glb, _ids, _notes = design_glb(load_design(sdir / "design"))
        if not glb:
            return None
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(glb)
        tmp.replace(cache)
        return glb


def make_handler(ws: Path):
    class H(BaseHTTPRequestHandler):
        server_version = "DroneBench"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, (VIEWER / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
                if path.startswith("/vendor/"):
                    rel = path[len("/vendor/"):]
                    f = (VIEWER / "vendor" / rel).resolve()
                    if (VIEWER / "vendor").resolve() not in f.parents or not f.is_file():
                        return self._send(404, b"not found", "text/plain")
                    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                    if f.suffix == ".js":
                        ctype = "text/javascript"
                    return self._send(200, f.read_bytes(), ctype, "public, max-age=86400")
                if path == "/api/state":
                    return self._json(state(ws))
                mm = re.fullmatch(r"/api/step/(\d+)(/model\.glb)?", path)
                if mm:
                    n = int(mm.group(1))
                    if mm.group(2):
                        glb = step_glb(ws, n)
                        if glb is None:
                            return self._send(404, b"no model for this step", "text/plain")
                        return self._send(200, glb, "model/gltf-binary", "public, max-age=3600")
                    return self._json(step_payload(ws, n))
                return self._send(404, b"not found", "text/plain")
            except W.WorkspaceError as e:
                return self._json({"error": str(e)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                return None
            except Exception as e:  # never kill the server on one bad request
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return H


def serve(ws, port: int = 8765, open_browser: bool = False) -> None:
    ws = Path(ws).resolve()
    W.meta(ws)  # fail early if this is not a workspace
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(ws))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"DroneBench dashboard for {ws.name}: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
