"""Workspace loop: init -> try (accept / reject) -> build -> status -> dashboard API."""
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from dronebench import server, workspace as W


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    p = tmp_path_factory.mktemp("ws") / "w"
    W.init(p, "falcon_v2_padded", steps=8)
    return p


def test_init_layout(ws):
    m = W.meta(ws)
    assert m["fixture"] == "falcon_v2_padded" and m["steps"] == 0 and m["baseline"] > 0
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


def test_build_works_on_a_variant_fixture(tmp_path):
    """falcon_v2_padded has no builder file of its own; it names the one it shares in params.yaml."""
    p = tmp_path / "w"
    W.init(p, "falcon_v2_padded", steps=4, necessity=False)
    o = W.build_step(p, {"wing.span_mm": 2150}, "wider", necessity=False)
    assert not o.get("errors"), o
    assert o["step"] == 1 and (p / "steps" / "1" / "design" / "bom.yaml").exists()


def test_init_can_start_from_chosen_numbers(tmp_path):
    """A new design is the builder run on your own numbers, not a fixed starting aircraft."""
    p = tmp_path / "w"
    m = W.init(p, "falcon_v2", steps=4, necessity=False, sets={"wing.span_mm": 2000, "htail.area_m2": 0.08})
    import yaml
    params = yaml.safe_load((p / "params.yaml").read_text())
    assert params["wing"]["span_mm"] == 2000 and params["htail"]["area_m2"] == 0.08
    assert yaml.safe_load((p / "design" / "params.yaml").read_text())["wing"]["span_mm"] == 2000
    assert m["start_params"] == {"wing.span_mm": 2000, "htail.area_m2": 0.08}
    assert (p / "design" / "cad").is_dir() and m["baseline"] >= 0


def test_force_only_replaces_a_dronebench_workspace(tmp_path):
    """`init . --force` in a folder of your own files used to delete the folder."""
    mine = tmp_path / "my_project"
    mine.mkdir()
    (mine / "notes.txt").write_text("keep me")
    with pytest.raises(W.WorkspaceError, match="not a DroneBench workspace"):
        W.init(mine, "trainer", steps=2, necessity=False, fidelity="analytic", force=True)
    assert (mine / "notes.txt").read_text() == "keep me"
    W.init(tmp_path / "w", "trainer", steps=2, necessity=False, fidelity="analytic")
    W.init(tmp_path / "w", "trainer", steps=2, necessity=False, fidelity="analytic", force=True)   # a real workspace: fine


def test_init_rejects_numbers_for_a_design_without_a_builder(tmp_path):
    with pytest.raises(W.WorkspaceError, match="no parameters to set"):
        W.init(tmp_path / "w", "trainer", steps=2, necessity=False, sets={"wing.span_mm": 1000})


def test_dashboard_api(ws):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(ws))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        st = json.load(urllib.request.urlopen(base + "/api/state"))
        assert st["runs"] and st["meta"]["fixture"] == "falcon_v2_padded"
        assert b"DroneBench" in urllib.request.urlopen(base + "/").read()
        glb = urllib.request.urlopen(base + "/api/step/0/model.glb").read()
        assert glb[:4] == b"glTF"
        assert json.load(urllib.request.urlopen(base + "/api/step/0"))["report"]["parts"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/vendor/../../pyproject.toml")
    finally:
        httpd.shutdown()


def test_dashboard_json_has_no_bare_nan():
    """Python writes NaN as a bare token that the browser's JSON.parse rejects (it blanked the dashboard once)."""
    out = server.dumps({"a": float("nan"), "b": [float("inf"), 1.5], "c": {"d": None}})
    assert json.loads(out) == {"a": None, "b": [None, 1.5], "c": {"d": None}}


def test_any_command_can_be_the_agent(tmp_path):
    """Your own agent is just a command run inside the workspace; it learns where it is from the environment."""
    from dronebench import agents
    p = tmp_path / "w"
    W.init(p, "trainer", steps=2, necessity=False, fidelity="analytic")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, pathlib, shutil\n"
        "pathlib.Path('seen.txt').write_text(os.environ['DRONEBENCH_WORKSPACE'] + '|' + os.environ['DRONEBENCH_AGENT'] + '|'"
        " + os.environ['DRONEBENCH_STEPS'] + '|' + str(pathlib.Path('TASK.md').exists()) + '|' + str(bool(shutil.which('dronebench'))))\n")
    rc = agents.run_command(p, f"{sys.executable} {probe}", label="my-agent", say=lambda *_: None)
    assert rc == 0
    assert (p / "seen.txt").read_text() == f"{p}|my-agent|2|True|True"


def test_a_failing_agent_command_reports_its_exit_code(tmp_path):
    from dronebench import agents
    p = tmp_path / "w"
    W.init(p, "trainer", steps=2, necessity=False, fidelity="analytic")
    assert agents.run_command(p, f"{sys.executable} -c 'raise SystemExit(7)'", say=lambda *_: None) == 7
