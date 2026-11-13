"""Workspace loop: init -> try (accept / reject) -> build -> status -> dashboard API."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from dronebench import server, workspace as W


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    p = tmp_path_factory.mktemp("ws") / "w"
    W.init(p, "falcon_v2", steps=8)
    return p


def test_init_layout(ws):
    m = W.meta(ws)
    assert m["fixture"] == "falcon_v2" and m["steps"] == 0 and m["baseline"] > 0
    for f in ("TASK.md", "design/bom.yaml", "runs.jsonl", "steps/0/report.json", "params.yaml"):
        assert (ws / f).exists(), f
    assert "dronebench try" in (ws / "TASK.md").read_text()


def test_invalid_ops_rejected(ws):
    o = W.try_step(ws, [{"op": "remove", "id": "no_such_part"}], "typo")
    assert not o["accepted"] and o["errors"]
    assert W.meta(ws)["current_step"] == 0


def test_gate_failure_rejected_with_reason(ws):
    o = W.try_step(ws, [{"op": "remove", "id": "main_spar"}], "break it")
    assert not o["accepted"] and o["reason"] == "gate failed"
    assert any(f.startswith("spar") for f in o["failures"])


def test_improvement_accepted_and_design_advances(ws):
    base = W.meta(ws)["best"]
    o = W.try_step(ws, [{"op": "remove", "id": "ballast_nose"}], "waste")
    assert o["accepted"] and o["delta"] > 0
    m = W.meta(ws)
    assert m["best"] > base and m["current_step"] == o["step"]
    rep = W.current_report(ws)   # accepted steps carry the full removal test
    assert any(p["necessity"] == "unnecessary" for p in rep["parts"])
    assert not (ws / "design" / "cad" / "ballast_nose.stl").exists()


def test_regression_rejected_unless_keep(ws):
    heavy = [{"op": "swap", "id": i, "entry": {"part_number": "APC 9x6E"}} for i in ("prop_L", "prop_R")]
    o = W.try_step(ws, heavy, "heavier")
    assert not o["accepted"] and (o["reason"] or "").startswith("regression")
    o2 = W.try_step(ws, heavy, "heavier, keep", keep=True)
    assert o2["accepted"] and o2["status"] == "kept"


def test_build_unknown_param(ws):
    o = W.build_step(ws, {"wing.nope": 1}, "bad knob")
    assert not o["accepted"] and "unknown parameter" in o["errors"][0]


def test_dashboard_api(ws):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(ws))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        st = json.load(urllib.request.urlopen(base + "/api/state"))
        assert st["runs"] and st["meta"]["fixture"] == "falcon_v2"
        assert b"DroneBench" in urllib.request.urlopen(base + "/").read()
        glb = urllib.request.urlopen(base + "/api/step/0/model.glb").read()
        assert glb[:4] == b"glTF"
        assert json.load(urllib.request.urlopen(base + "/api/step/0"))["report"]["parts"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/vendor/../../pyproject.toml")
    finally:
        httpd.shutdown()
