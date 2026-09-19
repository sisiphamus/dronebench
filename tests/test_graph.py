"""Part graph: proximity contacts, joint inference, electrical/mechanical inference, query helpers."""
from __future__ import annotations

import time

import numpy as np
import pytest
import trimesh

from dronebench.graph import build_graph, electrical_neighbors, interfaces, joint_for, linked, surface_gap_mm
from dronebench.ingest import load_design
from dronebench.geometry import make_tube
from dronebench.model import Design, Material, Part
from fixtures_min import mini_trainer


def _part(pid, type, mesh=None, cg=None, **kw):
    p = Part(id=pid, type=type, **kw)
    if mesh is not None:
        p.mesh, p.bbox_m, p.cg_m = mesh, np.array(mesh.bounds), np.array(mesh.center_mass)
    else:
        p.cg_m = np.array(cg if cg is not None else [0.0, 0.0, 0.0])
    return p


def _design(*parts):
    d = Design(name="t", root=".")
    d.parts = {p.id: p for p in parts}
    return d


def _box(size, center):
    b = trimesh.creation.box(size)
    b.apply_translation(center)
    return b


def test_spar_in_printed_bore():
    sleeve = make_tube(0.03, 0.0103, 0.1, "y", center=(0.3, 0.1, 0))      # printed root rib with a Ø10.3 bore
    spar = make_tube(0.010, 0.008, 1.0, "y", center=(0.3, 0, 0))
    d = _design(_part("rib", "wing", sleeve, role="wing"), _part("spar", "spar", spar, role="main_spar"))
    g = build_graph(d)
    e = g.edges["rib", "spar"]
    assert e["kind"] == "physical" and e["source"] == "inferred"
    assert e["gap_mm"] == pytest.approx(0.15, abs=0.03)
    assert e["joint"] == "tube in printed bore, slip fit" and e["tolerance"] == "+0.1/+0.3 mm"


def test_gaps_contacts_and_penetration():
    a = _part("a", "mount", _box((0.05, 0.05, 0.05), (0, 0, 0)))
    near = _part("near", "mount", _box((0.05, 0.05, 0.05), (0.0505, 0, 0)))      # 0.5 mm gap
    far = _part("far", "mount", _box((0.05, 0.05, 0.05), (0, 0.055, 0)))         # 5 mm gap
    into = _part("into", "mount", _box((0.05, 0.05, 0.05), (0, 0, -0.047)))      # 3 mm overlap
    assert surface_gap_mm(a, near) == pytest.approx(0.5, abs=0.02)
    assert surface_gap_mm(a, far) is None
    assert surface_gap_mm(a, into) == pytest.approx(-3.0, abs=0.1)
    g = build_graph(_design(a, near, far, into))
    assert g.has_edge("a", "near") and g.has_edge("a", "into") and not g.has_edge("a", "far")
    assert g.edges["a", "into"]["gap_mm"] < 0


def test_joint_rules():
    m3 = Part(id="s", type="fastener", specs={"size": "M3x8"})
    printed = Material(name="PLA", density_kg_m3=1240, kind="printed")
    assert joint_for(m3, Part(id="w", type="wing")) == ("M3 clearance hole", "hole Ø3.2-3.4 mm (ISO 273 fine-medium)")
    assert joint_for(Part(id="p", type="prop"), Part(id="m", type="motor"))[1] == "H7/h7 (0 to +0.015 mm)"
    assert joint_for(Part(id="m", type="motor"), Part(id="n", type="other", role="nacelle"))[0] == "bolt pattern"
    assert joint_for(Part(id="a", type="surface", role="aileron"), Part(id="w", type="wing"))[0] == \
        "hinge pin running fit"
    assert joint_for(Part(id="s", type="servo"), Part(id="w", type="wing"))[1] == "+0.2/+0.5 mm"
    assert joint_for(Part(id="b", type="battery"), Part(id="f", type="fuselage"))[0] == "battery bay"
    assert joint_for(Part(id="a", type="other", material=printed), Part(id="b", type="other", material=printed))[0] \
        == "printed interlock/glue joint"
    assert joint_for(Part(id="a", type="gps"), Part(id="b", type="other")) == ("contact", "±0.2 mm")


def _twin():
    """Twin-motor layout: motors/ESCs/props at y = ±0.25, electronics in the fuselage, 3 control surfaces."""
    parts = [_part("battery", "battery", cg=[0.2, 0, 0]), _part("fc", "fc", cg=[0.3, 0, 0]),
             _part("rx", "rx", cg=[0.32, 0, 0]), _part("gps", "gps", cg=[0.25, 0, 0.03])]
    for s, side in ((1, "R"), (-1, "L")):
        parts += [
            _part(f"motor_{side}", "motor", cg=[0.1, s * 0.25, 0]),
            _part(f"esc_{side}", "esc", cg=[0.2, s * 0.22, 0]),
            _part(f"prop_{side}", "prop", cg=[0.08, s * 0.25, 0]),
            _part(f"servo_ail_{side}", "servo", cg=[0.35, s * 0.4, 0]),
            _part(f"ail_{side}", "surface", cg=[0.4, s * 0.45, 0], role="aileron"),
        ]
    parts += [_part("servo_elev", "servo", cg=[0.9, 0, 0]), _part("elev", "surface", cg=[1.0, 0, 0], role="elevator")]
    return _design(*parts)


def test_electrical_and_mechanical_inference():
    d = _twin()
    d.parts["esc_L"].connects = ["motor_L"]
    g = build_graph(d)
    assert electrical_neighbors(g, "motor_R", "esc") == ["esc_R"]
    assert electrical_neighbors(g, "motor_L", "esc") == ["esc_L"]
    link = g.edges["esc_L", "motor_L"]["links"][0]
    assert link["source"] == "bom" and link["from"] == "esc_L" and link["joint"] == "motor phase wires"
    assert g.edges["esc_R", "motor_R"]["source"] == "inferred"
    for esc in ("esc_L", "esc_R"):
        assert electrical_neighbors(g, esc, "battery") == ["battery"]
        assert g.edges["battery", esc]["links"][0]["from"] == "battery"
    assert set(electrical_neighbors(g, "fc")) == {"rx", "gps", "esc_L", "esc_R", "servo_ail_L", "servo_ail_R",
                                                  "servo_elev"}
    assert linked(g, "prop_R", "mechanical") == ["motor_R"] and linked(g, "prop_L", "mechanical") == ["motor_L"]
    assert linked(g, "ail_L", "mechanical", "servo") == ["servo_ail_L"]
    assert linked(g, "elev", "mechanical", "servo") == ["servo_elev"]
    assert g.graph["warnings"] == []


def test_missing_esc_and_pdb():
    d = _twin()
    del d.parts["esc_L"]
    d.parts["pdb"] = _part("pdb", "pdb", cg=[0.2, 0, 0])
    g = build_graph(d)
    assert any("motor_L" in w and "ESC" in w for w in g.graph["warnings"])
    assert electrical_neighbors(g, "esc_R", "pdb") == ["pdb"] and electrical_neighbors(g, "esc_R", "battery") == []
    assert electrical_neighbors(g, "pdb", "battery") == ["battery"]


def test_interfaces_on_ingested_design(tmp_path):
    d = load_design(mini_trainer(tmp_path))
    g = build_graph(d)
    motor = interfaces(g, "motor")
    kinds = {(i["mates"], i["kind"]) for i in motor}
    assert ("prop", "physical") in kinds and ("prop", "mechanical") in kinds and ("esc", "electrical") in kinds
    for i in motor:
        assert set(i) == {"mates", "kind", "joint", "gap_mm", "tolerance", "source", "declared_fit"}
    phys = next(i for i in motor if i["mates"] == "prop" and i["kind"] == "physical")
    assert phys["joint"] == "shaft/prop adapter" and phys["gap_mm"] <= 1.0
    assert g.edges["motor", "prop"]["kind"] == "physical"               # physical link listed first
    spar = {i["mates"]: i for i in interfaces(g, "spar") if i["kind"] == "physical"}
    assert spar["wing"]["gap_mm"] < 0 and spar["wing"]["joint"] == "tube in printed bore, slip fit"
    assert spar["wing"]["declared_fit"]                     # spar lists the wing in specs.fits
    assert not any(i["declared_fit"] for i in motor)


def test_graph_speed_many_parts():
    parts = [_part(f"b{i}", "mount", _box((0.05, 0.05, 0.05), (0.0502 * i, 0, 0))) for i in range(40)]
    parts.append(_part("spar", "spar", make_tube(0.01, 0.008, 2.0, "x", center=(1.0, 0, 0.0299))))
    t = time.perf_counter()
    g = build_graph(_design(*parts))
    assert time.perf_counter() - t < 3.0
    assert all(g.has_edge(f"b{i}", f"b{i + 1}") for i in range(39))
    assert g.degree("spar") == 40
