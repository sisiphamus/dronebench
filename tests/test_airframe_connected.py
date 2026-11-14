from dronebench.edits import apply_edits
from dronebench.pipeline import evaluate


def _check(report):
    return {c["name"]: c for c in report["gate"]["checks"]}["airframe_connected"]


def test_baseline_is_one_piece():
    assert _check(evaluate("fixtures/falcon_v2", necessity=False))["pass"]


def _gate_names_failed(tmp_path, ops):
    out, errs = apply_edits("fixtures/falcon_v2", ops, tmp_path / "d")
    assert not errs
    return {c["name"] for c in evaluate(out, necessity=False)["gate"]["checks"] if not c["pass"]}


def test_cannot_delete_the_harness(tmp_path):
    failed = _gate_names_failed(tmp_path, [{"op": "remove", "id": "wiring_power"}, {"op": "remove", "id": "wiring_signal"}])
    assert "harness" in failed


def test_cannot_delete_the_rudder(tmp_path):
    ids = ["rudder", "servo_rudder", "horn_rudder", "link_rudder", "hinge_rudder"]
    failed = _gate_names_failed(tmp_path, [{"op": "remove", "id": i} for i in ids])
    assert "control_chain" in failed


def test_removing_a_nacelle_strands_the_motor(tmp_path):
    out, errs = apply_edits("fixtures/falcon_v2", [{"op": "remove", "id": "nacelle_L"}], tmp_path / "d")
    assert not errs
    chk = _check(evaluate(out, necessity=False))
    assert not chk["pass"] and "motor_L" in chk["detail"]


def test_cannot_delete_a_fuselage_section(tmp_path):
    failed = _gate_names_failed(tmp_path, [{"op": "remove", "id": "fuse2"}])
    assert "fuselage_continuous" in failed
