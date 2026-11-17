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


def test_small_inflation_is_caught(run):
    r = results.export_result(run)
    r["steps"][2]["score"] *= 1.004
    r["best"] = max(r["best"], r["steps"][2]["score"])
    assert not results.verify_result(r)["ok"]


def test_budget_smaller_than_the_attempts_is_refused(run):
    r = results.export_result(run)
    r["budget"] = 1
    with pytest.raises(results.ResultError, match="budget"):
        results.verify_result(r)


@pytest.mark.parametrize("drop", ["baseline", "steps", "fixture", "budget"])
def test_malformed_files_are_refused_cleanly(run, drop):
    r = results.export_result(run)
    del r[drop]
    with pytest.raises(results.ResultError, match="malformed"):
        results.verify_result(r)


def test_a_step_without_a_score_is_malformed(run):
    r = results.export_result(run)
    del r["steps"][0]["score"]
    with pytest.raises(results.ResultError, match="malformed"):
        results.verify_result(r)


def test_a_fixture_cannot_be_a_path(run, tmp_path):
    r = results.export_result(run)
    r["fixture"] = str(tmp_path)
    with pytest.raises(results.ResultError, match="not a built-in"):
        results.verify_result(r)


def _standard_result(**over):
    r = {"format": results.FORMAT, "fixture": "falcon_v2", "preset": "endurance", "fidelity": "flown", "start_params": {},
         "budget": 30, "baseline": 1.0, "best": 1.0, "steps": []}
    r.update(over)
    return r


def test_the_standard_task_is_the_plain_falcon_flown_within_30_steps():
    assert results.standard_problems(_standard_result()) == []
    for over, word in (({"fixture": "trainer"}, "fixture"), ({"start_params": {"wing.span_mm": 2200}}, "own numbers"),
                       ({"preset": "range"}, "preset"), ({"fidelity": "analytic"}, "fidelity"), ({"budget": 31}, "budget"),
                       ({"steps": [{"step": 1, "accepted": True, "score": 2.0, "ops": [{"op": "add", "cad_file": "/etc/passwd"}]}]},
                        "cad_file")):
        probs = results.standard_problems(_standard_result(**over))
        assert probs and word in " ".join(probs), (over, probs)


def test_verify_standard_stops_before_flying_anything(run):
    r = results.export_result(run)          # a trainer run: not the standard task
    v = results.verify_result(r, standard=True)
    assert not v["ok"] and v["replayed"] == 0 and "fixture" in v["mismatches"][0]["why"]


def test_build_and_try_steps_replay_together(tmp_path):
    p = tmp_path / "w"
    W.init(p, "falcon_v2", steps=4, necessity=False, fidelity="analytic")
    W.build_step(p, {"wing.span_mm": 2150}, "wider", agent="mix", keep=True, necessity=False)
    W.try_step(p, NUDGE, "battery", agent="mix", keep=True, necessity=False)
    r = results.export_result(p)
    assert [s["kind"] for s in r["steps"]] == ["build", "try"]
    v = results.verify_result(r)
    assert v["ok"] and v["replayed"] == 2, v


def test_a_design_folder_result_needs_that_folder(tmp_path):
    import shutil
    src = tmp_path / "mine"
    shutil.copytree("fixtures/trainer", src)
    p = tmp_path / "w"
    W.init(p, str(src), steps=3, necessity=False, fidelity="analytic")
    W.try_step(p, NUDGE, "x", agent="mine", keep=True, necessity=False)
    r = results.export_result(p)
    with pytest.raises(results.ResultError, match="not a built-in"):
        results.verify_result(r)
    assert results.verify_result(r, design=str(src))["ok"]
    other = tmp_path / "other"
    shutil.copytree("fixtures/trainer", other)
    import yaml
    bom = yaml.safe_load((other / "bom.yaml").read_text())
    bom["parts"][0]["part_number"] = "something else"
    (other / "bom.yaml").write_text(yaml.safe_dump(bom))
    with pytest.raises(results.ResultError, match="not the one"):
        results.verify_result(r, design=str(other))


def test_standard_cannot_be_combined_with_your_own_design(run, tmp_path):
    r = results.export_result(run)
    with pytest.raises(results.ResultError, match="standard"):
        results.verify_result(r, design="fixtures/trainer", standard=True)


def test_attempt_numbers_must_be_complete_and_counts_are_recomputed(run):
    r = results.export_result(run)
    r["steps"] = [s for s in r["steps"] if s["accepted"]]       # rejected steps left out
    r["attempts"] = 2
    with pytest.raises(results.ResultError, match="steps must be numbered"):
        results.verify_result(r)
    r = results.export_result(run)
    r["attempts"], r["accepted"] = 99, 99
    v = results.verify_result(r)
    assert v["ok"]                                              # the claim is recomputed, not trusted
    assert results.recount(r)["attempts"] == 3 and results.recount(r)["accepted"] == 2


@pytest.mark.parametrize("field,value", [
    ("budget", float("nan")), ("budget", float("inf")), ("start_params", "wing.span_mm=2"), ("start_params", ["a"]),
    ("preset", "nonsense"), ("fidelity", "dreamed"), ("baseline", float("nan"))])
def test_odd_field_values_are_refused_cleanly(run, field, value):
    r = results.export_result(run)
    r[field] = value
    with pytest.raises(results.ResultError):
        results.verify_result(r)


def test_odd_step_values_are_refused_cleanly(run):
    for key, value in (("accepted", "false"), ("ops", ["abc"]), ("score", float("nan")), ("step", "x")):
        r = results.export_result(run)
        r["steps"][1][key] = value
        with pytest.raises(results.ResultError):
            results.verify_result(r)


def test_edits_that_read_local_files_are_flagged(run):
    r = results.export_result(run)
    assert results.uses_local_files(r) == []
    r["steps"][1]["ops"] = [{"op": "add", "entry": {"id": "x"}, "cad_file": "/home/me/part.stl"}]
    assert results.uses_local_files(r) == [2]


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
