"""Pure physics models. Entry point: run(PhysicsInput) -> PhysicsResult.

Submodules: airfoil (2D sections), aero (3D lift, drag build-up, stability), propulsion (prop,
motor, ESC, battery, operating points), performance (speed sweep), structure (spar, servos).
run() never raises: bad input gives ok=False and/or NaN fields, with the reason in notes.
"""
from __future__ import annotations

import dataclasses
import math
import traceback

from ..model import PhysicsInput, PhysicsResult
from . import aero, airfoil, performance, propulsion, structure  # noqa: F401

__all__ = ["run", "aero", "airfoil", "performance", "propulsion", "structure"]


def _finite(x, default=float("nan")):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _train_problem(t) -> str:
    if t is None or t.motor is None or t.prop is None:
        return "missing motor or prop"
    m, p = t.motor, t.prop
    if not (aero.finite_pos(m.kv) and _finite(m.rm_ohm) >= 0 and _finite(m.i0_a) >= 0):
        return f"{m.id}: kv must be > 0, rm and i0 >= 0"
    if not (aero.finite_pos(p.diam_in) and aero.finite_pos(p.pitch_in)):
        return f"{p.id}: diameter and pitch must be > 0"
    return ""


def _sanitize_surface(s, notes):
    """Fill derived fields (AR, MAC, taper, x_ac) that are missing or non-finite."""
    if s is None or not (aero.finite_pos(s.area_m2) and aero.finite_pos(s.span_m)):
        return s
    ch = {}
    if not aero.finite_pos(s.aspect_ratio):
        ch["aspect_ratio"] = s.span_m**2 / s.area_m2
    if not aero.finite_pos(s.mac_m):
        ch["mac_m"] = s.area_m2 / s.span_m
    if not (math.isfinite(_finite(s.taper)) and s.taper >= 0):
        ch["taper"] = (s.tip_chord_m / s.root_chord_m) if aero.finite_pos(s.root_chord_m) and \
            math.isfinite(_finite(s.tip_chord_m)) else 1.0
    if not math.isfinite(_finite(s.sweep_le_deg)):
        ch["sweep_le_deg"] = 0.0
    if not math.isfinite(_finite(s.x_ac_m)):
        ch["x_ac_m"] = _finite(s.x_le_root_m, 0.0) + 0.25 * ch.get("mac_m", s.mac_m)
    if ch:
        notes.append(f"{s.name}: derived {', '.join(sorted(ch))} from span/area")
        s = dataclasses.replace(s, **ch)
    return s


def run(pin: PhysicsInput) -> PhysicsResult:
    res = PhysicsResult(ok=False)
    try:
        _run(pin, res)
    except Exception as exc:  # never raise to the caller
        res.ok = False
        res.notes.append(f"physics error: {type(exc).__name__}: {exc}")
        res.notes.append(traceback.format_exc(limit=2).strip().splitlines()[-1])
    return res


def _run(pin: PhysicsInput, res: PhysicsResult) -> None:
    notes = res.notes
    mass = _finite(getattr(pin.mass, "total_kg", None), 0.0)
    res.mass_kg = mass
    rho = _finite(pin.rho, 1.225)
    if rho <= 0:
        rho = 1.225
        notes.append("invalid rho, using 1.225")
    res.payload_fraction = pin.payload_kg / mass if mass > 0 else 0.0
    batt = pin.battery
    trains = []
    for t in pin.trains or []:
        bad = _train_problem(t)
        if bad:
            notes.append(f"power train ignored: {bad}")
        else:
            trains.append(t)
    if batt is not None:
        res.battery_energy_wh = propulsion.battery_energy_wh(batt)
        if batt.cells_s <= 0 or batt.capacity_mah <= 0:
            notes.append("battery has no cells or capacity")
            batt = None

    problems = []
    geom = pin.geom
    w = geom.wing if geom is not None else None
    if w is None:
        problems.append("no wing: aerodynamics not computed")
    elif not (aero.finite_pos(w.area_m2) and aero.finite_pos(w.span_m)):
        problems.append("wing geometry invalid (area and span must be > 0)")
        w = None
    else:
        geom = dataclasses.replace(geom, wing=_sanitize_surface(w, notes),
                                   htail=_sanitize_surface(geom.htail, notes), vtail=_sanitize_surface(geom.vtail, notes))
        w = geom.wing
    if mass <= 0:
        problems.append("mass must be > 0")
    if not trains:
        problems.append("no power train (motor + prop)")
    if batt is None:
        problems.append("no battery")

    # structure does not need a valid wing polar, only span
    span = w.span_m if w is not None else 0.0
    if span > 0 and mass > 0:
        res.spar_sf, sn = structure.spar_safety_factor(pin.spars or [], mass * performance.G, span,
                                                       _finite(pin.load_factor, 3.5))
        res.spar_sf = min(res.spar_sf, 99.0)
        notes += sn

    if w is None or mass <= 0:
        notes += problems
        return

    # aerodynamics + stability
    a_w = aero.lift_slope_3d(w)
    res.neutral_point_x_m, nn = aero.neutral_point(geom, a_w)
    notes += nn
    cg_x = _finite(pin.mass.cg_m[0]) if pin.mass.cg_m is not None and len(pin.mass.cg_m) else float("nan")
    res.static_margin = (res.neutral_point_x_m - cg_x) / w.mac_m
    if not math.isfinite(cg_x):
        notes.append("no CG: static margin not computed")
    res.tail_vh, res.tail_vv = aero.tail_volumes(geom)

    perf = performance.evaluate(geom, mass, trains, batt, rho, pin.cruise_ms, _finite(pin.usable_battery_fraction, 0.85))
    notes += perf.notes
    for f in ("stall_ms", "cl_max", "cd0", "oswald_e", "ld_max", "vmax_ms", "climb_ms", "best_endurance_ms",
              "best_range_ms", "cruise_ms", "endurance_min", "range_km", "wh_per_km", "motor_current_cruise_a",
              "motor_current_max_a", "battery_current_max_a"):
        setattr(res, f, getattr(perf, f))
    if math.isfinite(res.wh_per_km) and res.wh_per_km > 0:
        res.km_per_wh = 1.0 / res.wh_per_km
    if batt is not None and math.isfinite(res.battery_current_max_a):
        res.battery_c_max = res.battery_current_max_a / max(propulsion.battery_capacity_ah(batt), 1e-9)

    # servos
    if pin.servos:
        if math.isfinite(res.vmax_ms):
            vq = min(res.vmax_ms, 2.0 * res.cruise_ms) if math.isfinite(res.cruise_ms) else res.vmax_ms
        else:
            vq = 1.3 * (res.cruise_ms if math.isfinite(res.cruise_ms) else 2.0 * res.stall_ms)
            notes.append("servo load: Vmax unknown, using 1.3 x cruise")
        res.servo_margin, sn = structure.servo_margins(pin.servos, geom.control_surfaces or [], 0.5 * rho * vq**2)
        notes += sn

    notes += problems
    res.ok = not problems
