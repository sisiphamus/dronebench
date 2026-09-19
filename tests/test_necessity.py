"""Necessity verdict logic with a mocked evaluator (no geometry/physics needed)."""
from __future__ import annotations

import time

import pytest

from dronebench.gate import evaluate_gate
from dronebench.necessity import candidates, caution_moves, joint_rule, margins, removal_test
from dronebench.assemble import LinkIndex
from dronebench.score import compute_score
from dronebench.functions import describe

from test_gate import MASS, make_design, make_geom, make_pres


def fake_evaluator(var, removed, base, preset):
    """Physics stand-in: each removal nudges the fake result in a known way."""
    kw = {}
    if "main_spar" in removed:
        kw["spar_sf"] = float("nan")
    if "ballast" in removed:
        kw["endurance_min"] = 33.0          # lighter -> better
    if "gps_2" in removed:
        kw["endurance_min"] = 30.2
    if "fairing" in removed:
        kw["climb_ms"] = 2.8                # climb into its caution band (< 3 m/s) -> helpful
    if "gps" in removed:
        kw["spar_sf"] = 2.2                 # big drop but still outside the caution band -> unnecessary
    if "rx" in removed:
        pass                                # control_chain fails by itself
    if "fuselage" in removed:
        kw["endurance_min"] = 29.0          # score drops, margins fine -> helpful
    pres = make_pres(**kw)
    gate = evaluate_gate(var, None, make_geom(), pres, mass=MASS)
    return {"gate": gate, "pres": pres, "pin": None, "score": compute_score(preset, pres, gate)}


def run():
    d = make_design()
    pres = make_pres()
    gate = evaluate_gate(d, None, make_geom(), pres, mass=MASS)
    assert gate.passed
    base = {"graph": None, "geom": make_geom(), "pres": pres, "pin": None, "gate": gate,
            "score": compute_score("endurance", pres, gate)}
    return d, removal_test(d, "endurance", base, evaluator=fake_evaluator)


def test_candidates_group_fasteners():
    keys = [k for k, _ in candidates(make_design())]
    assert "group:motor_mount" in keys and "screws_motor" not in keys and "screw_hatch" in keys


def test_verdicts():
    d, res = run()
    assert res["main_spar"].verdict == "required" and "spar" in res["main_spar"].reason
    assert res["rx"].verdict == "required" and "control_chain" in res["rx"].reason
    assert res["motor"].verdict == "required" and "power_chain" in res["motor"].reason
    assert res["servo_elev"].verdict == "required"
    assert res["camera"].verdict == "required" and "payload" in res["camera"].reason
    assert res["ballast"].verdict == "unnecessary" and res["ballast"].score_delta == pytest.approx(3.0)
    assert res["gps_2"].verdict == "unnecessary"
    assert res["fairing"].verdict == "helpful" and "climb" in res["fairing"].reason
    assert res["fairing"].margins_after["climb"] == pytest.approx(2.8)
    assert res["fairing"].margins_before["climb"] == pytest.approx(4.0)
    assert res["fuselage"].verdict == "helpful" and "score" in res["fuselage"].reason


def test_fastener_rules():
    d, res = run()
    r = res["group:motor_mount"]
    assert r.verdict == "required" and "joint motor_mount unfastened: motor would detach" == r.reason
    # the hatch screw holds only a cosmetic fairing -> evaluated normally -> unnecessary
    assert res["screw_hatch"].verdict == "unnecessary"


def test_joint_rule_with_second_fastener():
    d = make_design()
    from dronebench.model import Part
    d.parts["screws_motor_b"] = Part(id="screws_motor_b", type="fastener", group="motor_mount_b", parent="motor")
    assert joint_rule(d, ["screws_motor"], LinkIndex(d)) is None


def test_margins_and_caution_bands():
    m = margins(make_pres())
    assert m["static_margin"] == pytest.approx(0.1) and m["climb"] == 4.0 and m["tail_vh"] == 0.5
    assert caution_moves({"climb": 4.0}, {"climb": 3.2}) == {}               # dropped 20 %, still > 3
    assert "climb" in caution_moves({"climb": 4.0}, {"climb": 2.9})          # moved into < 3
    assert "static_margin" in caution_moves({"static_margin": 0.12}, {"static_margin": 0.07})
    assert "static_margin" in caution_moves({"static_margin": 0.15}, {"static_margin": 0.19})
    assert caution_moves({"static_margin": 0.19}, {"static_margin": 0.12}) == {}   # moved out: fine
    assert "spar_sf" in caution_moves({"spar_sf": 1.9}, {"spar_sf": 1.6})    # already in band, >5 % worse
    assert caution_moves({"spar_sf": 1.9}, {"spar_sf": 1.88}) == {}


def test_drop_outside_band_is_unnecessary():
    d, res = run()
    r = res["gps"]
    assert r.verdict == "unnecessary" and "no check near its limit" in r.reason
    assert r.margins_before["spar_sf"] == 3.0 and r.margins_after["spar_sf"] == 2.2


def test_functions_describe():
    d = make_design()
    L = LinkIndex(d)
    fn, why = describe(d.parts["gps_2"], d, None, links=L)
    assert "duplicate of gps" in why
    assert describe(d.parts["gps"], d, None, links=L)[1].startswith("position fix")
    fn, why = describe(d.parts["motor"], d, None, links=L)
    assert fn == "propulsion" and "drives prop" in why and "esc" in why
    fn, why = describe(d.parts["screws_motor"], d, None, links=L)
    assert fn == "fastening" and "4× M3x8" in why and "motor_mount" in why
    fn, why = describe(d.parts["ballast"], d, None, links=L)
    assert fn == "ballast" and "80 g" in why and "forward" in why
    fn, why = describe(d.parts["servo_elev"], d, None, links=L)
    assert "moves elevator" in why and "fc" in why
    for p in d.parts.values():
        fn, why = describe(p, d, None, links=L)
        assert fn and why


def test_speed_real_evaluator_shape():
    """removal_test on the mock is fast; the real-pipeline timing is measured in the integration smoke."""
    t = time.perf_counter()
    run()
    assert time.perf_counter() - t < 5


def test_load_path_rule():
    import networkx as nx
    from dronebench.necessity import airframe_root, load_path_rule, structure_adjacency, _reach
    d = make_design()
    G = nx.Graph()
    for a, b in (("fuselage", "wing_L"), ("fuselage", "wing_R"), ("fuselage", "main_spar"),
                 ("fuselage", "fairing"), ("fairing", "htail"), ("htail", "vtail"), ("fuselage", "gps")):
        G.add_edge(a, b, kind="physical", links=[{"kind": "physical", "gap_mm": 0.0}])
    links = LinkIndex(d, G)
    adj = structure_adjacency(d, links)
    root = airframe_root(d)
    attached = _reach(adj, root, set())
    why = load_path_rule(d, ["fairing"], adj, root, attached)
    assert why and "htail" in why and "vtail" in why
    assert load_path_rule(d, ["gps"], adj, root, attached) is None


def test_base_already_failing_counts_only_new_failures():
    from dronebench.necessity import judge
    d = make_design()
    bad = make_pres(climb_ms=1.0)
    base = {"pres": bad, "pin": None, "gate": evaluate_gate(d, None, make_geom(), bad, mass=MASS), "score": (0.0, "min")}
    del d.parts["rx"]
    res = {"pres": bad, "pin": None, "gate": evaluate_gate(d, None, make_geom(), bad, mass=MASS), "score": (0.0, "min")}
    r = judge("rx", ["rx"], base, res)
    assert r.verdict == "required" and "control_chain" in r.reason and "base already fails climb" in r.reason
    d2 = make_design()
    del d2.parts["gps_2"]
    res = {"pres": bad, "pin": None, "gate": evaluate_gate(d2, None, make_geom(), bad, mass=MASS), "score": (0.0, "min")}
    assert judge("gps_2", ["gps_2"], base, res).verdict == "unnecessary"
