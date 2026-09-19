from __future__ import annotations

import pytest

from dronebench.model import Part
from dronebench.plausibility import CLAMPED, check_specs, needs_check


def _part(t, **specs):
    p = Part(id=f"x_{t}", type=t, specs=specs)
    p.sources = {k: "agent_supplied" for k in specs}
    return p


def test_tiny_motor_60a_clamped():
    p = _part("motor", kv=900, imax_a=60, mass_g=5, rm_ohm=0.05)
    ch = check_specs(p)
    fields = [c[0] for c in ch]
    assert "imax_a" in fields
    assert p.specs["imax_a"] == pytest.approx(8 * 5 / 25.2, rel=1e-3)
    assert p.sources["imax_a"] == CLAMPED
    assert "W/g" in next(c[3] for c in ch if c[0] == "imax_a")


def test_2kw_motor_uses_cells_max():
    p = _part("motor", kv=900, imax_a=100, mass_g=5, cells_max=2)
    check_specs(p)
    assert p.specs["imax_a"] == pytest.approx(8 * 5 / 8.4, rel=1e-3)


def test_lipo_400whkg_mass_raised():
    # 6S 10 Ah LiPo = 222 Wh; at 555 g that is 400 Wh/kg
    p = _part("battery", cells_s=6, capacity_mah=10000, mass_g=555, chemistry="lipo", c_rating=300)
    ch = dict((c[0], c) for c in check_specs(p))
    assert p.specs["mass_g"] == pytest.approx(6 * 3.7 * 10 / 200 * 1000, rel=1e-3)
    assert p.specs["c_rating"] == 150
    assert p.sources["mass_g"] == CLAMPED and "Wh/kg" in ch["mass_g"][3]


def test_liion_limit_higher():
    p = _part("battery", cells_s=6, capacity_mah=8400, mass_g=750, chemistry="liion")  # 241 Wh/kg: ok
    assert check_specs(p) == []


def test_kv_rm_servo_prop_esc():
    p = _part("motor", kv=20000, rm_ohm=0.0001, imax_a=10, mass_g=100)
    check_specs(p)
    assert p.specs["kv"] == 5000 and p.specs["rm_ohm"] == 0.005
    s = _part("servo", torque_kgcm=50, mass_g=9)
    check_specs(s)
    assert s.specs["torque_kgcm"] == pytest.approx(10.8)
    pr = _part("prop", diam_in=40, pitch_in=80)
    check_specs(pr)
    assert pr.specs["diam_in"] == 30 and pr.specs["pitch_in"] == 45
    e = _part("esc", imax_a=200, mass_g=10)
    check_specs(e)
    assert e.specs["imax_a"] == 30


def test_heavy_rating_needs_mass():
    p = _part("motor", kv=900, imax_a=30, mass_g=2, cells_max=1)
    check_specs(p)
    # power clamp first: 8 W/g * 2 g / 4.2 V = 3.8 A, so the >= 20 A mass rule no longer applies
    assert p.specs["imax_a"] < 20


def test_plausible_unchanged_and_needs_check():
    p = _part("motor", kv=620, rm_ohm=0.061, i0_a=0.9, imax_a=35, mass_g=128)
    assert check_specs(p) == []
    assert needs_check(p)
    q = Part(id="q", type="motor", specs={"kv": 620})
    q.sources = {"specs": "bom"}
    assert not needs_check(q)
