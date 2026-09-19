"""Planform/airfoil extraction, control surfaces, fuselage, wetted areas, mass properties, cache."""
from __future__ import annotations

import time

import numpy as np
import pytest

from dronebench.geometry import (extract_geometry, make_fuselage, make_naca4_fin, make_naca4_panel,
                                 make_naca4_wing, make_tube, mass_properties, naca_name)
from dronebench.ingest import load_design
from dronebench.model import Design, Part
from fixtures_min import mini_trainer


def _part(pid, mesh, type="wing", role=None, mass_g=100.0):
    p = Part(id=pid, type=type, role=role, mesh=mesh, mass_g=mass_g)
    p.bbox_m, p.cg_m = np.array(mesh.bounds), np.array(mesh.center_mass)
    return p


def _design(*parts):
    d = Design(name="t", root=".")
    d.parts = {p.id: p for p in parts}
    return d


def _mac(cr, ct):
    lam = ct / cr
    return 2 / 3 * cr * (1 + lam + lam ** 2) / (1 + lam)


def test_rectangular_naca2412_wing():
    g = extract_geometry(_design(_part("w", make_naca4_wing(1.0, 0.2, 0.2, "2412", x_le=0.25), role="wing")))
    w = g.wing
    assert w.span_m == pytest.approx(1.0, rel=0.02)
    assert w.area_m2 == pytest.approx(0.2, rel=0.02)
    assert w.mac_m == pytest.approx(0.2, rel=0.02)
    assert w.aspect_ratio == pytest.approx(5.0, rel=0.02)
    assert w.x_ac_m == pytest.approx(0.25 + 0.05, abs=1e-3)
    assert w.thickness_ratio == pytest.approx(0.12, abs=0.0012)
    assert w.camber == pytest.approx(0.02, abs=0.002)
    assert w.airfoil == "NACA2412"
    assert len(w.stations) == 40 and w.part_ids == ["w"]


def test_tapered_swept_dihedral_wing():
    b, cr, ct, sweep, dih = 1.2, 0.25, 0.125, 10.0, 5.0
    g = extract_geometry(_design(_part("w", make_naca4_wing(b, cr, ct, "4412", sweep_deg=sweep,
                                                            dihedral_deg=dih, x_le=0.3), role="wing")))
    w = g.wing
    area = (cr + ct) / 2 * b
    assert w.area_m2 == pytest.approx(area, rel=0.02)
    assert w.mac_m == pytest.approx(_mac(cr, ct), rel=0.02)
    assert w.aspect_ratio == pytest.approx(b ** 2 / area, rel=0.02)
    assert w.root_chord_m == pytest.approx(cr, rel=0.02) and w.tip_chord_m == pytest.approx(ct, rel=0.02)
    assert w.taper == pytest.approx(0.5, rel=0.03)
    assert w.sweep_le_deg == pytest.approx(sweep, abs=1.0)
    assert w.dihedral_deg == pytest.approx(dih, abs=1.0)
    # MAC leading edge sits at the MAC's spanwise station: y_mac = b/6 (1+2 lam)/(1+lam)
    y_mac = b / 6 * (1 + 2 * 0.5) / 1.5
    assert w.x_ac_m == pytest.approx(0.3 + y_mac * np.tan(np.radians(sweep)) + 0.25 * _mac(cr, ct), abs=2e-3)
    assert w.airfoil == "NACA4412" and w.thickness_ratio == pytest.approx(0.12, abs=0.0012)


def test_split_wing_with_gaps_and_ailerons():
    """Falcon-style wing: 2 panels per side, 1 mm gaps, fuselage gap in the middle, separate ailerons."""
    parts = [_part("fus", make_fuselage(1.0, 0.1, x_nose=-0.2), type="fuselage", role="fuselage")]
    for s, side in ((1, "R"), (-1, "L")):
        parts += [
            _part(f"w1{side}", make_naca4_panel(s * 0.05, s * 0.35, 0.22, 0.2, "4412", x_le0=0.2, x_le1=0.2)),
            _part(f"w2{side}", make_naca4_panel(s * 0.351, s * 0.6, 0.2, 0.15, "4412", "3411", 0.2, 0.22)),
        ]
        for p in parts[-2:]:
            p.role = "wing"
    # ailerons as their own bodies, inside the outer panels' planform (TE there is at x >= 0.372)
    ail = [_part(f"ail{side}", make_naca4_panel(s * 0.4, s * 0.58, 0.04, 0.04, "0008", x_le0=0.33, x_le1=0.33),
                 type="surface", role="aileron") for s, side in ((1, "R"), (-1, "L"))]
    g = extract_geometry(_design(*parts, *ail))
    w = g.wing
    # reference area carried through the fuselage: 0.05 m centre gap at root chord 0.22
    expected = 2 * (0.05 * 0.22 + 0.3 * 0.21 + 0.25 * 0.175)
    assert w.span_m == pytest.approx(1.2, rel=0.01)
    assert w.area_m2 == pytest.approx(expected, rel=0.02)
    assert w.airfoil == "NACA4412"
    assert sorted(w.part_ids) == ["ailL", "ailR", "w1L", "w1R", "w2L", "w2R"]
    cs = {c.id: c for c in g.control_surfaces}
    assert set(cs) == {"ailL", "ailR"}
    assert cs["ailR"].area_m2 == pytest.approx(0.18 * 0.04, rel=0.02)
    assert cs["ailR"].span_m == pytest.approx(0.18, rel=0.01) and cs["ailR"].chord_m == pytest.approx(0.04, rel=0.02)
    assert cs["ailR"].parent_surface == "wing"
    assert g.fuselage_length_m == pytest.approx(1.0) and g.fuselage_max_diam_m == pytest.approx(0.1)

    # a missing root panel is a real hole, not a gap to bridge: its area and the centre section go
    # (station edges fall within the 30 mm station spacing, hence 5 %)
    d = _design(*parts, *ail)
    del d.parts["w1L"]
    assert extract_geometry(d).wing.area_m2 == pytest.approx(expected - 0.3 * 0.21 - 0.1 * 0.22, rel=0.05)


def test_conventional_tail_and_fin():
    g = extract_geometry(_design(
        _part("ht", make_naca4_wing(0.4, 0.12, 0.08, "0010", x_le=0.9), type="tail", role="htail"),
        _part("vt", make_naca4_fin(0.15, 0.12, 0.06, "0009", sweep_deg=30, x_le=0.9, z0=0.02), type="tail",
              role="vtail"),
        _part("rud", make_naca4_fin(0.1, 0.03, 0.03, "0006", x_le=1.03, z0=0.03), type="surface", role="rudder"),
    ))
    assert g.wing is None and any("no wing" in w for w in g.warnings)
    v = g.vtail
    assert v.span_m == pytest.approx(0.15, rel=0.01)
    assert v.area_m2 == pytest.approx(0.15 * 0.09 + 0.1 * 0.03, rel=0.03)   # fin + rudder (behind the fin)
    assert v.sweep_le_deg == pytest.approx(30, abs=1.0)
    assert v.airfoil == "NACA0009"
    h = g.htail
    assert h.area_m2 == pytest.approx(0.04, rel=0.02) and h.mac_m == pytest.approx(_mac(0.12, 0.08), rel=0.02)
    rud = g.control_surfaces[0]
    assert rud.parent_surface == "vtail" and rud.span_m == pytest.approx(0.1, rel=0.01)


def test_v_tail_projections():
    dih, c, half = 35.0, 0.1, 0.25
    rise = half * np.tan(np.radians(dih))
    panels = [_part(f"v{side}", make_naca4_panel(0.0, s * half, c, c, "0010", x_le0=0.9, x_le1=0.9, z0=0.0, z1=rise),
                    type="tail", role="vtail_ruddervator") for s, side in ((1, "R"), (-1, "L"))]
    g = extract_geometry(_design(*panels))
    assert g.htail.area_m2 == pytest.approx(2 * half * c, rel=0.02)            # horizontal projection
    assert g.htail.dihedral_deg == pytest.approx(dih, abs=1.0)
    # side projection of both panels; the solid's height includes the airfoil thickness at root and tip
    assert g.vtail.span_m == pytest.approx(rise, abs=0.012)
    assert g.vtail.area_m2 == pytest.approx(2 * c * g.vtail.span_m, rel=0.03)
    assert g.vtail.aspect_ratio == pytest.approx(g.vtail.span_m / c, rel=0.03)   # per panel


def test_naca_name():
    assert naca_name(0.12, 0.04, 0.4) == "NACA4412"
    assert naca_name(0.0904, 0.001, 0.3) == "NACA0009"


def test_mass_properties_and_wetted(tmp_path):
    d = load_design(mini_trainer(tmp_path))
    d.parts["esc"].qty = 2
    mp = mass_properties(d)
    total = sum(p.mass_g * p.qty for p in d.parts.values())
    assert mp.total_kg == pytest.approx(total / 1e3)
    assert mp.by_part_g["esc"] == pytest.approx(2 * d.parts["esc"].mass_g)
    cg = sum(p.mass_g * p.qty * p.cg_m for p in d.parts.values()) / total
    assert np.allclose(mp.cg_m, cg)

    g = extract_geometry(d)
    assert set(g.wetted_area_m2) == {"wing", "htail", "vtail", "fuselage", "nacelles", "other"}
    assert g.wetted_area_m2["wing"] == pytest.approx(d.parts["wing"].area_m2)
    # motor sticks out of the fuselage bbox; the spar is inside the wing bbox; the prop never counts
    assert g.wetted_area_m2["other"] == pytest.approx(d.parts["motor"].area_m2)
    assert g.wing.area_m2 == pytest.approx(0.2, rel=0.02)
    assert g.fuselage_length_m == pytest.approx(0.9)


def test_geometry_cache():
    wing = _part("w", make_naca4_wing(1.0, 0.2, 0.2, "2412"), role="wing")
    ballast = _part("ballast", make_tube(0.02, 0, 0.02, "x", center=(0.05, 0, 0)), type="ballast")
    d = _design(wing, ballast)
    g1 = extract_geometry(d)
    t = time.perf_counter()
    del d.parts["ballast"]
    g2 = extract_geometry(d)
    assert time.perf_counter() - t < 0.01
    assert g2.wing.area_m2 == g1.wing.area_m2
    g2.wing.area_m2 = -1.0                    # callers get copies
    assert extract_geometry(d).wing.area_m2 == g1.wing.area_m2
    d.parts["w"].mesh = make_naca4_wing(1.0, 0.25, 0.25, "2412")
    assert extract_geometry(d).wing.area_m2 == pytest.approx(0.25, rel=0.02)
