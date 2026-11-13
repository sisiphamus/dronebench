"""Unit and sanity tests for dronebench.physics."""
import copy
import dataclasses
import math
import time

import numpy as np
import pytest

from dronebench import physics
from dronebench.model import AeroGeometry, MassProps, PhysicsInput
from dronebench.physics import aero, airfoil, propulsion, structure
from dronebench.physics.reference import falcon_v2_input


def run(**kw):
    return physics.run(falcon_v2_input(**kw))


# ---------------------------------------------------------------- airfoil

def test_parse_naca():
    a = airfoil.parse_naca("NACA4411")
    assert (a.camber, a.camber_pos, a.thickness) == (0.04, 0.4, 0.11)
    assert airfoil.parse_naca("naca 0012").thickness == 0.12
    g = airfoil.parse_naca(None, 0.14, 0.03)
    assert g.thickness == 0.14 and g.camber == 0.03


def test_airfoil_values():
    a4411, a0012 = airfoil.parse_naca("NACA4411"), airfoil.parse_naca("NACA0012")
    assert 1.3 <= airfoil.cl_max_2d(a4411, 3e5) <= 1.4
    assert 1.1 <= airfoil.cl_max_2d(a0012, 3e5) <= 1.2
    assert airfoil.cl_max_2d(a4411, 1e6) > airfoil.cl_max_2d(a4411, 1e5)
    assert math.degrees(airfoil.zero_lift_angle(a4411)) == pytest.approx(-4.15, abs=0.3)
    assert airfoil.zero_lift_angle(a0012) == 0.0
    assert 5.8 < airfoil.lift_slope_2d(a4411) < 6.4
    assert 0.006 < airfoil.cd_min(a4411, 3e5) < 0.015
    assert airfoil.cd_min(a4411, 1e5) > airfoil.cd_min(a4411, 1e6)


# ---------------------------------------------------------------- aero

def test_aero_basics():
    w = falcon_v2_input().geom.wing
    assert 4.5 < aero.lift_slope_3d(w) < 5.5
    assert aero.oswald_e(9.71) == pytest.approx(0.764, abs=0.01)
    assert aero.oswald_e(30) == 0.6 and aero.oswald_e(2) == 0.9
    assert aero.skin_friction(1e6) < aero.skin_friction(1e5)


# ---------------------------------------------------------------- propulsion

def test_prop_11x8_static():
    pc = propulsion.prop_coeffs(11, 8)
    kgf = propulsion.prop_thrust(pc, 9000 / 60, 0.0, 1.225) / 9.80665
    assert 1.9 <= kgf <= 2.4


def test_prop_11x8_efficiency():
    pc = propulsion.prop_coeffs(11, 8)
    J = np.linspace(0.05, 0.8, 200)
    ct, cp = propulsion.ct_cp(pc, J)
    eta = J * ct / cp
    assert 0.65 <= eta.max() <= 0.78
    assert 0.4 <= J[eta.argmax()] <= 0.65
    assert eta[np.searchsorted(J, 0.6)] > 0.65


def test_operating_point_and_max_thrust():
    pin = falcon_v2_input()
    op = propulsion.operating_point(pin.trains, pin.battery, 15.0, 2.5, 1.225)
    assert op is not None and 0 < op["throttle"] < 1
    assert op["electrical_power_w"] > 2.5 * 15.0
    assert propulsion.operating_point(pin.trains, pin.battery, 15.0, 500.0, 1.225) is None
    assert propulsion.operating_point([], pin.battery, 15.0, 2.5, 1.225) is None
    mt = propulsion.max_thrust(pin.trains, pin.battery, [0.0, 10.0, 20.0, 30.0])
    assert np.all(np.diff(mt.thrust_n) < 0)
    assert mt.battery_voltage_v[0] < propulsion.battery_voc(pin.battery)


def test_battery():
    b = falcon_v2_input().battery
    assert propulsion.battery_voc(b) == pytest.approx(21.6)
    assert propulsion.battery_resistance(b) == pytest.approx(0.045)
    assert propulsion.battery_energy_wh(b) == pytest.approx(181.44)


# ---------------------------------------------------------------- structure

def test_spar_split_and_none():
    pin = falcon_v2_input()
    both, _ = structure.spar_safety_factor(pin.spars, 34.3, 2.093, 3.5)
    main, _ = structure.spar_safety_factor(pin.spars[:1], 34.3, 2.093, 3.5)
    assert both > main > 1.0
    sf, notes = structure.spar_safety_factor([], 34.3, 2.093, 3.5)
    assert sf == 0.3 and "no spar" in notes[0]


def test_servo_margin():
    pin = falcon_v2_input()
    m, _ = structure.servo_margins(pin.servos, pin.geom.control_surfaces, 500.0)
    m2, _ = structure.servo_margins(pin.servos, pin.geom.control_surfaces, 1000.0)
    assert all(m[k] == pytest.approx(2 * m2[k]) for k in m)


# ---------------------------------------------------------------- run(): monotonic sanity

def test_heavier():
    a, b = run(mass_kg=3.5), run(mass_kg=5.0)
    assert b.stall_ms > a.stall_ms
    assert b.wh_per_km > a.wh_per_km
    assert b.spar_sf < a.spar_sf
    assert b.climb_ms < a.climb_ms


def test_bigger_battery_diminishing():
    def endurance(cap, extra_g):
        pin = falcon_v2_input()
        pin.battery.capacity_mah = cap
        pin.mass.total_kg += extra_g / 1000
        return physics.run(pin).endurance_min
    e1, e2, e3 = endurance(8400, 0), endurance(16800, 840), endurance(25200, 1680)
    assert e3 > e2 > e1
    assert (e3 - e2) < (e2 - e1)


def test_smaller_tail_lower_margin():
    base = run()
    pin = falcon_v2_input()
    h = pin.geom.htail
    pin.geom.htail = dataclasses.replace(h, area_m2=h.area_m2 * 0.5, span_m=h.span_m * 0.5 ** 0.5,
                                         aspect_ratio=h.aspect_ratio)
    small = physics.run(pin)
    assert small.static_margin < base.static_margin
    assert small.tail_vh < base.tail_vh


def test_one_motor_removed():
    base = run()
    pin = falcon_v2_input()
    pin.trains = pin.trains[:1]
    one = physics.run(pin)
    assert one.climb_ms < base.climb_ms
    assert one.vmax_ms < base.vmax_ms


def test_no_wing():
    pin = falcon_v2_input()
    pin.geom.wing = None
    r = physics.run(pin)
    assert not r.ok and any("wing" in n for n in r.notes)
    assert r.mass_kg == 3.5


def test_no_power():
    pin = falcon_v2_input()
    pin.trains, pin.battery = [], None
    r = physics.run(pin)
    assert not r.ok
    assert math.isfinite(r.stall_ms) and math.isfinite(r.static_margin)
    assert math.isnan(r.vmax_ms)


def test_underpowered_no_raise():
    pin = falcon_v2_input(mass_kg=40.0)
    r = physics.run(pin)
    assert isinstance(r.notes, list)


@pytest.mark.parametrize("mutate", [
    lambda p: setattr(p.geom.wing, "area_m2", 0.0),
    lambda p: setattr(p.geom.wing, "aspect_ratio", float("nan")),
    lambda p: setattr(p.geom.wing, "airfoil", "garbage"),
    lambda p: setattr(p.mass, "total_kg", 0.0),
    lambda p: setattr(p.mass, "cg_m", None),
    lambda p: setattr(p.battery, "cells_s", 0),
    lambda p: setattr(p.trains[0].motor, "kv", 0.0),
    lambda p: setattr(p.trains[0].prop, "diam_in", 0.0),
    lambda p: setattr(p, "geom", AeroGeometry()),
    lambda p: setattr(p, "spars", []),
    lambda p: setattr(p.servos[0], "surface_id", "nope"),
    lambda p: setattr(p, "rho", -1.0),
])
def test_robust(mutate):
    pin = falcon_v2_input()
    mutate(pin)
    r = physics.run(pin)          # must not raise
    assert isinstance(r.ok, bool)


def test_speed():
    pin = falcon_v2_input()
    physics.run(pin)
    best = float("inf")
    for _ in range(10):            # best-of-10: robust to other processes loading the machine
        t = time.perf_counter()
        physics.run(pin)
        best = min(best, time.perf_counter() - t)
    assert best < 0.050


def test_servo_design_speed_and_falcon_margin():
    from dronebench.model import ControlSurfaceGeom, ServoSpec
    pin = falcon_v2_input(cruise_kmh=55)
    pin.geom.control_surfaces = [ControlSurfaceGeom("ail", "aileron", 0.02, 0.06, 0.33, "wing"),
                                 ControlSurfaceGeom("elev", "elevator", 0.012, 0.05, 0.24, "htail")]
    pin.servos = [ServoSpec("s_ail", 2.0, "ail"), ServoSpec("s_elev", 2.0, "elev")]
    r = physics.run(pin)
    vd = min(r.vmax_ms, 2 * r.cruise_ms)
    q = 0.5 * pin.rho * vd**2
    assert r.servo_margin["s_ail"] == pytest.approx(2.0 * 0.0981 / (structure.CQ_HINGE * q * 0.02 * 0.06))
    # real Falcon flies on 2.0 kg cm servos: must pass the 1.5 gate
    assert r.servo_margin["s_ail"] >= 1.5 and r.servo_margin["s_elev"] >= 1.5


def test_default_cruise_floor():
    r = physics.run(falcon_v2_input(cruise_kmh=None))
    assert r.cruise_ms == r.best_range_ms >= 1.3 * r.stall_ms * 0.999
    assert r.best_endurance_ms >= 1.2 * r.stall_ms * 0.999


def test_spar_nested_doubler():
    from dronebench.model import SparSpec
    from dronebench.physics.reference import CF_TUBE
    W, b, n = 3.4 * 9.80665, 2.093, 3.5
    main = SparSpec("main", 10, 8, 1.6, CF_TUBE, "main_spar")                     # whole wing
    rear = SparSpec("rear", 8, 6, 1.2, CF_TUBE, "rear_spar")
    doubler = SparSpec("doubler", 8, 6, 0.4, CF_TUBE, "main_spar", y_min_m=-0.2, y_max_m=0.2)
    sf1, _ = structure.spar_safety_factor([main], W, b, n)
    sf2, _ = structure.spar_safety_factor([main, rear], W, b, n)
    sf3, notes = structure.spar_safety_factor([main, rear, doubler], W, b, n)
    M = n * W * b / (3 * math.pi)
    I = lambda D, d: math.pi * (D**4 - d**4) / 64
    I_main, I_8 = I(0.010, 0.008), I(0.008, 0.006)
    assert sf1 == pytest.approx(600e6 / (M * 0.005 / I_main), rel=1e-6)
    assert sf3 == pytest.approx(600e6 / (M * 0.005 / (I_main + 2 * I_8)), rel=1e-6)
    assert sf1 < sf2 < sf3
    assert any("critical at y=0.00" in x for x in notes)
    off_root = dataclasses.replace(doubler, y_min_m=None, y_max_m=None, spans_root=False)
    assert structure.spar_safety_factor([main, rear, off_root], W, b, n)[0] == pytest.approx(sf2)


def test_moment_distribution():
    W, b, n = 34.0, 2.0, 3.5
    assert structure.moment_at(0.0, W, b, n) == pytest.approx(structure.root_moment(W, b, n))
    y = np.linspace(0, 1.0, 20001)
    lp = 2 * n * W / (math.pi * 1.0) * np.sqrt(1 - y**2)
    for y0 in (0.2, 0.5, 0.8):
        m = y >= y0
        assert structure.moment_at(y0, W, b, n) == pytest.approx(np.trapezoid(lp[m] * (y[m] - y0), y[m]), rel=1e-3)
    assert structure.moment_at(1.0, W, b, n) == pytest.approx(0.0, abs=1e-9)


def test_spar_spanwise():
    from dronebench.model import SparSpec
    from dronebench.physics.reference import CF_TUBE
    W, b, n = 3.4 * 9.80665, 2.093, 3.5
    main = SparSpec("main", 10, 8, 1.6, CF_TUBE, y_min_m=-0.8, y_max_m=0.8)
    rear = SparSpec("rear", 8, 6, 1.2, CF_TUBE, "rear_spar", y_min_m=-0.6, y_max_m=0.6)
    dbl = SparSpec("dbl", 8, 6, 0.4, CF_TUBE, y_min_m=-0.2, y_max_m=0.2)
    I_main = math.pi * (0.010**4 - 0.008**4) / 64
    # without the rear spar, main alone carries M(0.2) ~ 0.6 M_root just outboard of the doubler
    m02 = structure.moment_at(0.2001, W, b, n)
    assert 0.55 < m02 / structure.root_moment(W, b, n) < 0.7
    sf_main_02 = 600e6 / (m02 * 0.005 / I_main)
    sf, _ = structure.spar_safety_factor([main, dbl], W, b, n)
    assert sf <= sf_main_02 + 1e-9
    # a short main spar ending mid-semispan: short wing tip is skin-only only if the moment is small
    stub = SparSpec("stub", 10, 8, 0.8, CF_TUBE, y_min_m=-0.4, y_max_m=0.4)
    sf_stub, notes = structure.spar_safety_factor([stub], W, b, n)
    assert sf_stub == 0.3 and any("no spar" in x for x in notes)
    sf_long, _ = structure.spar_safety_factor([main], W, b, n)       # tip beyond 0.8 m: < 5 % M_root
    assert sf_long > 1.0
    # removing the main spar leaves the outer wing (0.6-0.8 m) unsupported
    assert structure.spar_safety_factor([rear, dbl], W, b, n)[0] == 0.3
    # a spar only on the right side still covers the left by symmetry of |y|
    right = dataclasses.replace(main, y_min_m=0.0, y_max_m=0.8)
    assert structure.spar_safety_factor([right], W, b, n)[0] == pytest.approx(sf_long)
