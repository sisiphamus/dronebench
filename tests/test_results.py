"""A finished run exports to a result file that anyone can replay: the score is re-flown, not trusted."""
import copy
import json

import pytest

from dronebench import cli, results, workspace as W

NUDGE = [{"op": "move", "ids": ["battery"], "offset_m": [-0.004, 0, 0]}]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    p = tmp_path_factory.mktemp("res") / "w"
    W.init(p, "trainer", steps=4, necessity=False, fidelity="analytic")
    W.try_step(p, [{"op": "remove", "id": "no_such_part"}], "typo", agent="mine", necessity=False)
    W.try_step(p, NUDGE, "battery back a little", agent="mine", keep=True, necessity=False)
    W.try_step(p, NUDGE, "again", agent="mine", keep=True, necessity=False)
    return p


def test_export_summarises_the_run(run):
    r = results.export_result(run)
    assert r["format"] == "dronebench-result/1"
    assert (r["fixture"], r["preset"], r["fidelity"], r["agent"]) == ("trainer", W.meta(run)["preset"], "analytic", "mine")
    assert r["baseline"] == pytest.approx(W.meta(run)["baseline"], abs=1e-3)
    assert (r["attempts"], r["accepted"], r["budget"]) == (3, 2, 4)
    assert [s["accepted"] for s in r["steps"]] == [False, True, True]
    assert r["steps"][1]["ops"] == NUDGE and r["steps"][1]["note"] == "battery back a little"
    assert r["best"] == pytest.approx(max([r["baseline"]] + [s["score"] for s in r["steps"] if s["accepted"]]), abs=1e-3)


def test_replay_reproduces_the_scores(run):
    v = results.verify_result(results.export_result(run))
    assert v["ok"], v
    assert v["replayed"] == 2 and not v["mismatches"]


def test_replay_catches_an_inflated_score(run):
    r = results.export_result(run)
    r["steps"][2]["score"] += 10
    r["best"] += 10
    v = results.verify_result(r)
    assert not v["ok"]
    assert {m["what"] for m in v["mismatches"]} >= {"step 3 score", "best"}


def test_replay_catches_edits_that_would_not_have_been_accepted(run):
    r = results.export_result(run)
    r["steps"][1]["ops"] = [{"op": "move", "ids": ["battery"], "offset_m": [0.05, 0, 0]}]   # CG far aft: fails the gate
    v = results.verify_result(r)
    assert not v["ok"]
    assert v["mismatches"][0]["what"] == "step 2 accepted" and "static_margin" in v["mismatches"][0]["why"]


def test_replay_rejects_a_file_in_the_wrong_format():
    with pytest.raises(results.ResultError, match="format"):
        results.verify_result({"format": "something-else"})


def test_cli_result_and_verify_round_trip(run, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert cli.main(["result", str(run), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["agent"] == "mine"
    capsys.readouterr()
    assert cli.main(["verify", str(out)]) == 0
    assert "VERIFIED" in capsys.readouterr().out
    bad = json.loads(out.read_text())
    bad["best"] += 5
    out.write_text(json.dumps(bad))
    assert cli.main(["verify", str(out)]) == 3
    assert "NOT VERIFIED" in capsys.readouterr().out
