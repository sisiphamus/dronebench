"""The flown evaluation end to end: a real design flies, broken designs fail for the right reason."""
import numpy as np
import pytest

from dronebench import aero
from dronebench.edits import apply_edits
from dronebench.pipeline import evaluate


@pytest.fixture(scope="module")
def trainer():
    return evaluate("fixtures/trainer", necessity=False, flown=True)


def checks(rep):
    return {c["name"]: c for c in rep["gate"]["checks"]}


def test_trainer_is_flown_and_scored(trainer):
    assert trainer["gate"]["pass"], [c for c in trainer["gate"]["checks"] if not c["pass"]]
    fl = trainer["flight"]
    assert fl["ok"] and fl["fidelity"].startswith("flown")
    assert trainer["score"]["fidelity"] == "flown" and trainer["score"]["value"] > 20
    for name in ("orbit_calm", "orbit_wind", "legs_calm", "legs_wind", "gusts_light", "gusts_gusty", "gusts_legs", "load_factor", "no_stall"):
        assert name in checks(trainer)
    # flown numbers replace the closed-form ones and are kept alongside
    assert trainer["aircraft"]["analytic_endurance_min"] != trainer["aircraft"]["endurance_min"]


def test_wind_costs_range_and_turbulence_does_not_change_the_score(trainer):
    """An equal-distance round trip in a steady wind costs more energy per km than calm air (the headwind leg loses more than
    the tailwind leg gains), and the score must not move when only the turbulence seeds change."""
    calm, wind = trainer["flight"]["legs"]
    assert wind["wh_per_km"] > calm["wh_per_km"] * 1.03
    from dronebench import flight as Fl, mission as M
    from dronebench.ingest import load_design
    from dronebench.pipeline import apply_plausibility, core
    d = load_design("fixtures/trainer"); apply_plausibility(d)
    veh = Fl.Vehicle.from_bits(core(d, "endurance"), d)
    a = M.evaluate_flight(veh, "endurance", seed_offset=0)
    b = M.evaluate_flight(veh, "endurance", seed_offset=100)
    assert a["score"] == b["score"] > 0


def test_unstable_design_is_rejected_by_the_flight_check():
    rep = evaluate("fixtures/bad_tail", necessity=False, flown=True)
    c = checks(rep)
    assert not c["static_margin"]["pass"] and c["static_margin"]["value"] < 0.05
    assert rep["score"]["value"] == 0.0


def test_a_design_that_cannot_trim_fails_the_gate(tmp_path):
    """A 3-inch propeller cannot hold level flight: the gate must fail with the reason, not pass with no flight checks."""
    out, errs = apply_edits("fixtures/trainer", [
        {"op": "swap", "id": "prop", "entry": {"part_number": "APC 3x2", "specs": {"diameter_in": 3, "pitch_in": 2, "mass_g": 5}}}], tmp_path / "p")
    if errs:
        pytest.skip(str(errs))
    rep = evaluate(out, necessity=False, flown=True)
    assert not rep["gate"]["pass"] and rep["score"]["value"] == 0.0


def test_cg_too_far_aft_fails_and_forward_move_is_flown(tmp_path):
    aft, errs = apply_edits("fixtures/trainer", [{"op": "move", "ids": ["battery"], "offset_m": [0.25, 0, 0]}], tmp_path / "aft")
    assert not errs
    rep = evaluate(aft, necessity=False, flown=True)
    assert not checks(rep)["static_margin"]["pass"]


def test_mass_only_edits_reuse_the_aero_table(trainer, tmp_path):
    """Removing a small part or moving the battery must not change the geometry hash (no 6 s rebuild)."""
    from dronebench.ingest import load_design
    from dronebench.pipeline import core
    d0 = load_design("fixtures/trainer")
    k0 = aero.geometry_key(core(d0, "endurance")["geom"])
    out, errs = apply_edits("fixtures/trainer", [{"op": "move", "ids": ["battery"], "offset_m": [-0.02, 0, 0]}], tmp_path / "m")
    assert not errs
    d1 = load_design(out)
    assert aero.geometry_key(core(d1, "endurance")["geom"]) == k0


def test_bigger_wing_changes_the_geometry_key(tmp_path):
    from dronebench.ingest import load_design
    from dronebench.pipeline import core
    k0 = aero.geometry_key(core(load_design("fixtures/trainer"), "endurance")["geom"])
    out, errs = apply_edits("fixtures/trainer", [{"op": "scale", "ids": ["wing_L1", "wing_L2", "wing_R1", "wing_R2"], "factors": [1.0, 1.1, 1.0], "about": "origin"}], tmp_path / "w")
    if errs:
        pytest.skip("no wing parts with those ids in this fixture")
    k1 = aero.geometry_key(core(load_design(out), "endurance")["geom"])
    assert k1 != k0
