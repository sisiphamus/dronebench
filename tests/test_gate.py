"""Gate + score logic on a synthetic in-memory design and a fake PhysicsResult (no CAD, no physics)."""
from __future__ import annotations

import dataclasses
import math

import networkx as nx
import numpy as np
import pytest

from dronebench.gate import evaluate_gate
from dronebench.model import (AeroGeometry, Design, MassProps, Mission, Part, PhysicsResult, SurfaceGeom)
from dronebench.score import compute_score


def _p(pid, type_, cg=(0.3, 0, 0), **kw) -> Part:
    p = Part(id=pid, type=type_, **kw)
    p.cg_m = np.array(cg, float)
    p.mass_g = kw.get("specs", {}).get("mass_g", 10.0)
    return p


def make_design() -> Design:
    parts = [
        _p("fuselage", "fuselage", role="fuselage"),
        _p("wing_L", "wing", (0.3, -0.3, 0), role="wing"), _p("wing_R", "wing", (0.3, 0.3, 0), role="wing"),
        _p("htail", "tail", (0.9, 0, 0), role="htail"), _p("vtail", "tail", (0.9, 0, 0.1), role="vtail"),
        _p("aileron_L", "surface", (0.35, -0.4, 0), role="aileron", parent="wing_L"),
        _p("aileron_R", "surface", (0.35, 0.4, 0), role="aileron", parent="wing_R"),
        _p("elevator", "surface", (0.95, 0, 0), role="elevator", parent="htail"),
        _p("rudder", "surface", (0.95, 0, 0.1), role="rudder", parent="vtail"),
        _p("servo_ail_L", "servo", connects=["aileron_L", "fc"], specs={"torque_kgcm": 2.0, "mass_g": 9}),
        _p("servo_ail_R", "servo", connects=["aileron_R", "fc"], specs={"torque_kgcm": 2.0, "mass_g": 9}),
        _p("servo_elev", "servo", connects=["elevator", "fc"], specs={"torque_kgcm": 2.0, "mass_g": 9}),
        _p("servo_rud", "servo", connects=["rudder", "fc"], specs={"torque_kgcm": 2.0, "mass_g": 9}),
        _p("fc", "fc", connects=["rx", "gps", "gps_2"], part_number="Matek F405-Wing"),
        _p("rx", "rx"),
        _p("gps", "gps", part_number="M8N"), _p("gps_2", "gps", part_number="M8N"),
        _p("motor", "motor", (0.02, 0, 0), connects=["esc", "prop"],
           specs={"kv": 1000, "rm_ohm": 0.05, "i0_a": 1.0, "imax_a": 30, "mass_g": 70}),
        _p("prop", "prop", (0.0, 0, 0), part_number="APC 9x6E"),
        _p("esc", "esc", connects=["battery"], specs={"imax_a": 40, "mass_g": 30}),
        _p("battery", "battery", (0.2, 0, 0), specs={"cells_s": 3, "capacity_mah": 2200, "c_rating": 25,
                                                     "mass_g": 180}),
        _p("main_spar", "spar", (0.3, 0, 0), role="main_spar",
           specs={"outer_d_mm": 10, "inner_d_mm": 8, "length_m": 1.2}),
        _p("camera", "camera", (0.1, 0, 0), locked=True),
        _p("screws_motor", "fastener", group="motor_mount", qty=4, parent="motor", specs={"size": "M3x8"}),
        _p("screw_hatch", "fastener", connects=["fairing"]),
        _p("fairing", "fairing"),
        _p("ballast", "ballast", (0.05, 0, 0), specs={"mass_g": 80}),
    ]
    return Design(name="synth", root=None, parts={p.id: p for p in parts}, mission=Mission(payload_ids=["camera"]))


def _surf(name, span, area, x):
    return SurfaceGeom(name=name, span_m=span, area_m2=area, mac_m=area / span, root_chord_m=area / span,
                       tip_chord_m=area / span, aspect_ratio=span ** 2 / area, taper=1.0, sweep_le_deg=0,
                       dihedral_deg=0, x_le_root_m=x, x_ac_m=x + 0.25 * area / span, z_m=0)


def make_geom() -> AeroGeometry:
    return AeroGeometry(wing=_surf("wing", 1.2, 0.24, 0.25), htail=_surf("htail", 0.4, 0.05, 0.85),
                        vtail=_surf("vtail", 0.15, 0.02, 0.85))


def make_pres(**kw) -> PhysicsResult:
    base = dict(ok=True, mass_kg=1.0, stall_ms=10.0, cruise_ms=15.0, climb_ms=4.0, static_margin=0.1,
                neutral_point_x_m=0.32, tail_vh=0.5, tail_vv=0.04, spar_sf=3.0, motor_current_max_a=20.0,
                motor_current_cruise_a=5.0, battery_current_max_a=20.0, battery_c_max=9.0, endurance_min=30.0,
                range_km=25.0, vmax_ms=30.0, km_per_wh=1.2, payload_fraction=0.1,
                servo_margin={s: 3.0 for s in ("servo_ail_L", "servo_ail_R", "servo_elev", "servo_rud")})
    base.update(kw)
    return PhysicsResult(**base)


MASS = MassProps(total_kg=1.0, cg_m=np.array([0.3, 0, 0]))


def gate_of(design=None, pres=None, geom=None, graph=None):
    return evaluate_gate(design or make_design(), graph, geom or make_geom(), pres or make_pres(), mass=MASS)


def check(g, name):
    return next(c for c in g.checks if c.name == name)


NAMES = ["lifting_surfaces", "static_margin", "stall_margin", "climb", "spar", "tail_volume", "servo_torque",
         "motor_current", "esc_current", "battery_current", "power_chain", "control_chain", "payload", "clearance"]


def test_all_pass():
    g = gate_of()
    assert [c.name for c in g.checks] == NAMES
    assert g.passed, [(c.name, c.detail) for c in g.failures()]


@pytest.mark.parametrize("kw,name,word", [
    (dict(static_margin=0.02), "static_margin", "aft"),
    (dict(static_margin=0.30), "static_margin", "forward"),
    (dict(cruise_ms=12.0), "stall_margin", "stall"),
    (dict(climb_ms=1.0), "climb", "climb"),
    (dict(spar_sf=1.2), "spar", "safety factor"),
    (dict(tail_vh=0.2), "tail_volume", "Vh"),
    (dict(tail_vv=0.01), "tail_volume", "Vv"),
    (dict(motor_current_max_a=35.0), "motor_current", "motor"),
    (dict(battery_c_max=30.0), "battery_current", "25C"),
])
def test_physics_checks_fail(kw, name, word):
    g = gate_of(pres=make_pres(**kw))
    c = check(g, name)
    assert not c.passed and not g.passed
    assert word in c.detail


def test_esc_current_and_servo():
    d = make_design()
    d.parts["esc"].specs["imax_a"] = 15
    assert not check(gate_of(d), "esc_current").passed
    p = make_pres()
    p.servo_margin["servo_elev"] = 1.2
    c = check(gate_of(pres=p), "servo_torque")
    assert not c.passed and "servo_elev" in c.detail and c.value == pytest.approx(1.2)


def test_static_margin_detail_values():
    c = check(gate_of(pres=make_pres(static_margin=0.02)), "static_margin")
    assert "0.02" in c.detail and "x_cg 0.3" in c.detail and c.limit == [0.05, 0.2]


def test_physics_not_ok():
    g = gate_of(pres=PhysicsResult(ok=False, notes=["no wing"]))
    for n in ("static_margin", "climb", "spar"):
        assert not check(g, n).passed and "no wing" in check(g, n).detail


def test_lifting_surfaces():
    geom = make_geom()
    geom.htail = None
    c = check(gate_of(geom=geom), "lifting_surfaces")
    assert not c.passed and "horizontal" in c.detail


def test_power_chain():
    d = make_design()
    d.parts["esc"].connects = []
    c = check(gate_of(d), "power_chain")
    assert not c.passed and "battery" in c.detail
    d = make_design()
    del d.parts["prop"]
    d.parts["motor"].connects = ["esc"]
    assert not check(gate_of(d), "power_chain").passed


def test_control_chain():
    d = make_design()
    del d.parts["servo_elev"]
    c = check(gate_of(d), "control_chain")
    assert not c.passed and "elevator" in c.detail
    d = make_design()
    del d.parts["rx"]
    assert "no receiver" in check(gate_of(d), "control_chain").detail
    d = make_design()
    d.parts["servo_rud"].connects = ["rudder"]  # not wired to the FC
    assert "not linked to the FC" in check(gate_of(d), "control_chain").detail
    d = make_design()
    d.parts["flap_L"] = _p("flap_L", "surface", role="flap", parent="wing_L")
    assert "flap_L" in check(gate_of(d), "control_chain").detail


def test_payload():
    d = make_design()
    del d.parts["camera"]
    c = check(gate_of(d), "payload")
    assert not c.passed and "camera" in c.detail


def test_clearance_graph():
    d = make_design()
    G = nx.Graph()
    G.add_edge("battery", "gps", kind="physical", links=[{"kind": "physical", "gap_mm": -3.0, "joint": "contact"}])
    G.add_edge("main_spar", "wing_L", kind="physical", links=[{"kind": "physical", "gap_mm": -5.0, "joint": "contact"}])
    c = check(gate_of(d, graph=G), "clearance")
    assert not c.passed and "battery/gps" in c.detail and "main_spar" not in c.detail


def test_prop_disc_clearance():
    import trimesh
    d = make_design()
    prop = d.parts["prop"]
    prop.cg_m = np.array([0.0, 0, 0])
    d.parts["motor"].cg_m = np.array([0.03, 0, 0])  # pusher-style axis along -x
    box = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    box.apply_translation([0.0, 0.0, 0.08])  # sits inside the 9" disc (r = 0.114 m)
    f = d.parts["fairing"]
    f.mesh, f.bbox_m = box, np.array(box.bounds)
    c = check(gate_of(d), "clearance")
    assert not c.passed and "prop" in c.detail and "fairing" in c.detail
    box.apply_translation([0.2, 0, 0])
    f.mesh, f.bbox_m = box, np.array(box.bounds)
    assert check(gate_of(d), "clearance").passed


@pytest.mark.parametrize("preset,val,unit", [("endurance", 30.0, "min"), ("range", 25.0, "km"),
                                             ("speed", 108.0, "km/h"), ("payload", 0.1, "fraction"),
                                             ("efficiency", 1.2, "km/Wh")])
def test_score_presets(preset, val, unit):
    g = gate_of()
    v, u = compute_score(preset, make_pres(), g)
    assert v == pytest.approx(val) and u == unit


def test_score_zero_on_fail():
    pres = make_pres(climb_ms=0.5)
    g = gate_of(pres=pres)
    assert compute_score("endurance", pres, g) == (0.0, "min")
    with pytest.raises(ValueError):
        compute_score("nope", pres, g)


def test_assemble_trains():
    from dronebench.assemble import build_physics_input
    d = make_design()
    pin = build_physics_input(d, make_geom(), MASS, None)
    assert len(pin.trains) == 1
    t = pin.trains[0]
    assert (t.prop.diam_in, t.prop.pitch_in, t.esc_imax_a) == (9.0, 6.0, 40.0)
    assert pin.battery.cells_s == 3 and pin.spars[0].outer_d_mm == 10
    assert {s.surface_id for s in pin.servos} == {"aileron_L", "aileron_R", "elevator", "rudder"}
    assert pin.payload_kg == pytest.approx(0.010)
    d.parts["battery2"] = dataclasses.replace(d.parts["battery"], id="battery2", specs=dict(d.parts["battery"].specs))
    pin = build_physics_input(d, make_geom(), MASS, None)
    assert pin.battery.parallel_packs == 2 and pin.battery.capacity_mah == 2200 and pin.battery.mass_g == 360


def test_wing_asymmetry():
    d = make_design()
    for pid, y in (("wing_L", -1), ("wing_R", 1)):
        d.parts[pid].bbox_m = np.array([[0.25, min(0, y * 0.6), 0], [0.45, max(0, y * 0.6), 0.02]])
    assert check(gate_of(d), "lifting_surfaces").passed
    d.parts["wing_L"].bbox_m = np.array([[0.25, -0.3, 0], [0.45, 0.0, 0.02]])
    c = check(gate_of(d), "lifting_surfaces")
    assert not c.passed and "asymmetric" in c.detail
