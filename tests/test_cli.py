import json
import sys
import types

import pytest

from dronebench import cli
from dronebench.runs import append_run, load_runs, next_step, record_from_report
from test_viewer import fake_design, load_sample


@pytest.fixture
def fake_eval(monkeypatch):
    calls = []

    def _eval(design_dir, preset=None, necessity=True):
        calls.append((str(design_dir), preset, necessity))
        rep = load_sample()
        if preset:
            rep["preset"] = rep["score"]["preset"] = preset
        return rep

    monkeypatch.setattr(cli, "_evaluate", _eval)
    return calls


@pytest.fixture
def design(tmp_path):
    d = tmp_path / "trainer_demo"
    d.mkdir()
    return d


# ---------- parsing ----------

def test_parser_score_flags():
    a = cli.build_parser().parse_args(["score", "d", "--preset", "range", "--no-necessity", "--summary",
                                       "--runs", "r.jsonl", "--label", "claude", "--step", "3"])
    assert (a.cmd, a.design, a.preset, a.no_necessity, a.summary, a.runs, a.label, a.step) == \
        ("score", "d", "range", True, True, "r.jsonl", "claude", 3)


def test_parser_view_two_designs():
    a = cli.build_parser().parse_args(["view", "a", "b", "--out", "v.html", "--open"])
    assert a.designs == ["a", "b"] and a.out == "v.html" and a.open


def test_usage_errors_exit_2(design, capsys):
    assert cli.main([]) == 2
    assert cli.main(["score", str(design), "--preset", "nope"]) == 2
    assert cli.main(["score", "/no/such/design"]) == 2
    assert "design folder not found" in capsys.readouterr().err
    assert cli.main(["baseline", str(design)]) == 2  # --steps required


def test_internal_error_exit_1(design, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("physics exploded")
    monkeypatch.setattr(cli, "_evaluate", boom)
    assert cli.main(["score", str(design)]) == 1
    err = capsys.readouterr().err
    assert "physics exploded" in err and "Traceback" not in err
    with pytest.raises(RuntimeError):
        cli.main(["--debug", "score", str(design)])


# ---------- commands ----------

def test_score_json_and_runs(design, tmp_path, fake_eval, capsys):
    runs = tmp_path / "runs.jsonl"
    out = tmp_path / "report.json"
    assert cli.main(["score", str(design), "--runs", str(runs), "--out", str(out)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["score"]["value"] == 41.6 and json.loads(out.read_text())["design"] == "trainer_demo"
    assert cli.main(["score", str(design), "--runs", str(runs), "--no-necessity", "--preset", "range"]) == 0
    capsys.readouterr()
    recs = load_runs(runs)
    assert [r["step"] for r in recs] == [0, 1]
    assert fake_eval[-1] == (str(design), "range", False)
    assert recs[0]["gate_pass"] is True and recs[0]["failures"] == [] and recs[0]["mass_g"] == 2224.8


def test_score_summary_shows_failures(design, monkeypatch, capsys):
    rep = load_sample()
    rep["gate"]["pass"] = False
    rep["gate"]["checks"][1].update({"value": 0.31, "pass": False})
    monkeypatch.setattr(cli, "_evaluate", lambda *a, **k: rep)
    assert cli.main(["score", str(design), "--summary"]) == 0
    out = capsys.readouterr().out
    assert "FAIL" in out and "static_margin: 0.31 vs 0.05..0.2" in out and "Wh/km" in out


def test_parts_table(design, fake_eval, capsys):
    assert cli.main(["parts", str(design), "--table"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["id", "type", "function", "necessity", "mass_g", "material", "margin"]
    assert "ballast_nose" in out and "unnecessary" in out
    assert cli.main(["parts", str(design)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 20


def test_view_command(design, tmp_path, fake_eval, monkeypatch, capsys):
    fake_ingest = types.ModuleType("dronebench.ingest")
    fake_ingest.load_design = lambda d: fake_design()
    monkeypatch.setitem(sys.modules, "dronebench.ingest", fake_ingest)
    out = tmp_path / "v.html"
    assert cli.main(["view", str(design), str(design), "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip() == str(out)
    assert 'id="glb-1"' in out.read_text()
    assert all(c[2] is True for c in fake_eval)  # necessity always on for view


def test_edit_command(design, tmp_path, monkeypatch, capsys):
    fake = types.ModuleType("dronebench.edits")
    fake.apply_edits = lambda d, ops, out: (out, [] if ops[0]["id"] != "bad" else ["unknown part id: bad"])
    monkeypatch.setitem(sys.modules, "dronebench.edits", fake)
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps([{"op": "remove", "id": "ballast_nose"}]))
    assert cli.main(["edit", str(design), "--ops", str(ops), "--out", str(tmp_path / "new")]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "new")
    ops.write_text(json.dumps([{"op": "remove", "id": "bad"}]))
    assert cli.main(["edit", str(design), "--ops", str(ops), "--out", str(tmp_path / "new")]) == 1
    assert "unknown part id" in capsys.readouterr().err
    ops.write_text("{not json")
    assert cli.main(["edit", str(design), "--ops", str(ops), "--out", str(tmp_path / "n2")]) == 2


def test_baseline_command(design, tmp_path, monkeypatch, capsys):
    got = {}
    fake = types.ModuleType("dronebench.baseline")
    fake.random_search = lambda *a: got.setdefault("args", a) and {"best": 44.0}
    monkeypatch.setitem(sys.modules, "dronebench.baseline", fake)
    assert cli.main(["baseline", str(design), "--steps", "5", "--seed", "7", "--runs", "r.jsonl"]) == 0
    assert got["args"][1:4] == (5, None, 7) and got["args"][5] == "r.jsonl"


# ---------- runs ----------

def test_runs_append_load(tmp_path):
    p = tmp_path / "sub" / "runs.jsonl"
    assert load_runs(p) == [] and next_step(p) == 0
    rec = record_from_report(load_sample(), ops=[{"op": "remove", "id": "gps_2"}], agent="claude")
    r0 = append_run(p, rec)
    r1 = append_run(p, dict(rec))
    append_run(p, {"agent": "baseline", "score": 40.0})
    append_run(p, {"agent": "claude", "score": 1.0, "step": 10})
    with p.open("a") as f:
        f.write("garbage line\n")
    recs = load_runs(p)
    assert (r0["step"], r1["step"]) == (0, 1)
    assert [r["step"] for r in recs] == [0, 1, 0, 10]
    assert recs[0]["ops"][0]["id"] == "gps_2" and recs[0]["metrics"]["wh_per_km"] == 1.52
    assert {"timestamp", "design", "preset", "score", "gate_pass", "failures", "mass_g"} <= set(recs[0])
    assert next_step(p, "claude") == 11
