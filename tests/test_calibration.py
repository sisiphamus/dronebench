"""Calibration: Titan Falcon V2 (published ~1.6 Wh/km at 45-65 km/h, 3.5 kg AUW).

The input is hand-built in dronebench.physics.reference (planform, tail, motors, battery, spars
documented there), so this test does not depend on CAD ingest.
"""
import math

import pytest

from dronebench import physics
from dronebench.physics.reference import falcon_v2_input

KMH = 1 / 3.6


@pytest.fixture(scope="module")
def falcon():
    return physics.run(falcon_v2_input(3.5, cg_frac_mac=0.28, cruise_kmh=55))


def test_ok(falcon):
    assert falcon.ok, falcon.notes


def test_wh_per_km_55(falcon):
    assert 1.36 <= falcon.wh_per_km <= 1.84, falcon.wh_per_km


@pytest.mark.parametrize("kmh", [45, 55, 65])
def test_wh_per_km_band(kmh):
    r = physics.run(falcon_v2_input(3.5, cruise_kmh=kmh))
    # published 1.6 across the band; allow +-20 % at the band edges
    assert 1.28 <= r.wh_per_km <= 1.92, (kmh, r.wh_per_km)


def test_stall(falcon):
    assert 30 * KMH <= falcon.stall_ms <= 40 * KMH, falcon.stall_ms / KMH


def test_stall_7kg():
    r = physics.run(falcon_v2_input(7.0))
    assert 45 * KMH <= r.stall_ms <= 55 * KMH, r.stall_ms / KMH


def test_ld(falcon):
    assert 10 <= falcon.ld_max <= 18


@pytest.mark.parametrize("cg", [0.25, 0.30])
def test_static_margin(cg):
    r = physics.run(falcon_v2_input(3.5, cg_frac_mac=cg))
    assert 0.05 <= r.static_margin <= 0.25, r.static_margin


def test_tail_volumes(falcon):
    assert 0.35 <= falcon.tail_vh <= 0.7
    assert 0.02 <= falcon.tail_vv <= 0.06


def test_speed_climb(falcon):
    assert falcon.vmax_ms > 80 * KMH
    assert falcon.climb_ms > 2.0
    assert falcon.stall_ms < falcon.best_endurance_ms <= falcon.best_range_ms < falcon.vmax_ms


def test_spar(falcon):
    assert falcon.spar_sf > 1.5


def test_endurance_range(falcon):
    # 181 Wh pack, 85 % usable: expect roughly 1.5-3.5 h and 80-160 km
    assert 90 <= falcon.endurance_min <= 210
    assert 80 <= falcon.range_km <= 160
    assert math.isclose(falcon.km_per_wh, 1 / falcon.wh_per_km)


def test_currents(falcon):
    assert falcon.motor_current_cruise_a < 10
    assert 20 < falcon.motor_current_max_a < 40
    assert falcon.battery_c_max < 10
