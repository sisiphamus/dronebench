"""Point performance from a vectorised airspeed sweep.

  D(V)  = q S (CD0(V) + CL^2 / (pi e AR)),  CL = W / (q S), CD0 re-evaluated at each V (Reynolds).
  V_s   = sqrt(2 W / (rho S CLmax)), CLmax evaluated at Re(V_s) (two fixed-point passes).
  Sweep V from 1.05 V_s to max(8 V_s, 60 m/s) (<= 120 m/s), 400 points.
  Vmax  = the highest speed of the first contiguous band (from 1.05 V_s) where full-throttle
          thrust >= drag, linearly interpolated at the crossing.
  P_e(V)= power drawn from the cells to hold thrust = drag (propulsion.operating_points).
  Best endurance = argmin P_e over V >= 1.2 V_s;  best range = argmin P_e / V over V >= 1.3 V_s
  (operational margins: nobody loiters at 1.05 V_s, and 1.3 V_s is the gate's cruise margin).
  Cruise = required cruise, else best range (so always >= 1.3 V_s when not set by the mission).
  Endurance = usable energy / P_e(best endurance); range = usable energy / min(P_e / V);
  Wh/km = P_e(cruise) / (3.6 V_cruise). Usable energy = V_nom x Ah x usable fraction.
  Climb = max_V (T_max(V) - D(V)) V / W (small climb angle: level-flight drag).
  L/D max = max over the sweep of W / D(V).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..model import AeroGeometry, BatterySpec, PowerTrain
from . import aero, propulsion as prop

G = 9.80665
ENDURANCE_MIN_VS = 1.2
RANGE_MIN_VS = 1.3


@dataclass
class Performance:
    stall_ms: float = float("nan")
    cl_max: float = float("nan")
    cd0: float = float("nan")          # at cruise
    oswald_e: float = float("nan")
    ld_max: float = float("nan")
    vmax_ms: float = float("nan")
    climb_ms: float = float("nan")
    best_endurance_ms: float = float("nan")
    best_range_ms: float = float("nan")
    cruise_ms: float = float("nan")
    power_cruise_w: float = float("nan")
    endurance_min: float = 0.0
    range_km: float = 0.0
    wh_per_km: float = float("nan")
    motor_current_cruise_a: float = float("nan")
    motor_current_max_a: float = float("nan")
    battery_current_max_a: float = float("nan")
    cd0_breakdown: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def aero_only(geom: AeroGeometry, weight_n: float, rho: float) -> Performance:
    """Stall, CLmax, e, CD0, L/D: everything that does not need propulsion."""
    w = geom.wing
    out = Performance()
    S = w.area_m2
    ar = w.aspect_ratio if aero.finite_pos(w.aspect_ratio) else w.span_m**2 / S
    out.oswald_e = aero.oswald_e(ar)
    cl = 1.2
    vs = math.sqrt(2 * weight_n / (rho * S * cl))
    for _ in range(3):
        cl = aero.wing_cl_max(w, vs, rho)
        vs = math.sqrt(2 * weight_n / (rho * S * cl))
    out.cl_max, out.stall_ms = cl, vs
    return out


def drag(geom: AeroGeometry, weight_n: float, rho: float, V, e: float):
    w = geom.wing
    S = w.area_m2
    ar = w.aspect_ratio if aero.finite_pos(w.aspect_ratio) else w.span_m**2 / S
    V = np.asarray(V, float)
    q = 0.5 * rho * V**2
    cl = weight_n / (q * S)
    cd0, parts, notes = aero.cd0_buildup(geom, S, V, rho)
    return q * S * (cd0 + cl**2 / (math.pi * e * ar)), cd0, parts, notes


def evaluate(geom: AeroGeometry, mass_kg: float, trains: Sequence[PowerTrain], battery: Optional[BatterySpec],
             rho: float = 1.225, cruise_ms: Optional[float] = None, usable_fraction: float = 0.85) -> Performance:
    W = mass_kg * G
    out = aero_only(geom, W, rho)
    vs = out.stall_ms
    v_top = min(max(8 * vs, 60.0), 120.0)
    V = np.linspace(1.05 * vs, v_top, 400)
    D, cd0, parts, notes = drag(geom, W, rho, V, out.oswald_e)
    out.notes += notes
    out.ld_max = float(np.max(W / D))
    out.cd0 = float(cd0[0])
    out.cd0_breakdown = parts
    if not trains or battery is None:
        out.notes.append("no propulsion: speeds, climb, endurance not computed")
        return out

    mt = prop.max_thrust(trains, battery, V, rho)
    out.notes += mt.notes
    excess = mt.thrust_n - D
    if excess[0] < 0:
        out.notes.append("cannot sustain level flight at any speed (full-throttle thrust < drag)")
        out.climb_ms = float(np.max(excess * V / W))
        out.vmax_ms = float("nan")
        _static_currents(out, trains, battery, rho)
        return out
    neg = np.nonzero(excess < 0)[0]
    if neg.size:
        k = neg[0]
        f = excess[k - 1] / (excess[k - 1] - excess[k])
        out.vmax_ms = float(V[k - 1] + f * (V[k] - V[k - 1]))
        band = slice(0, k)
    else:
        out.vmax_ms = float(V[-1])
        out.notes.append(f"Vmax capped at sweep limit {V[-1]:.0f} m/s")
        band = slice(0, V.size)
    out.climb_ms = float(np.max(excess[band] * V[band] / W))

    op = prop.operating_points(trains, battery, V[band], D[band], rho)
    P = np.where(op.feasible, op.electrical_power_w, np.inf)
    if not np.any(np.isfinite(P)):
        out.notes.append("no feasible cruise operating point")
        _static_currents(out, trains, battery, rho)
        return out
    Vb = V[band]
    ie = int(np.argmin(np.where(Vb >= ENDURANCE_MIN_VS * vs, P, np.inf)))
    ir = int(np.argmin(np.where(Vb >= RANGE_MIN_VS * vs, P / Vb, np.inf)))
    if not np.isfinite(P[ie]):
        ie = int(np.argmin(P))
    if not np.isfinite(P[ir]):
        ir = int(np.argmin(P / Vb))
        out.notes.append(f"best-range speed below {RANGE_MIN_VS} x stall (level flight not reachable above it)")
    out.best_endurance_ms = float(V[ie])
    out.best_range_ms = float(V[ir])

    e_usable_wh = prop.battery_energy_wh(battery) * usable_fraction
    out.endurance_min = float(e_usable_wh / P[ie] * 60.0)
    out.range_km = float(e_usable_wh * 3.6 / (P[ir] / V[ir]))

    vc = cruise_ms if (cruise_ms and math.isfinite(cruise_ms) and cruise_ms > 0) else out.best_range_ms
    out.cruise_ms = float(vc)
    if vc < vs:
        out.notes.append(f"cruise {vc:.1f} m/s is below stall {vs:.1f} m/s")
    Dc, cd0c, _, _ = drag(geom, W, rho, [vc], out.oswald_e)
    out.cd0 = float(cd0c[0])
    opc = prop.operating_points(trains, battery, [vc], Dc, rho)
    if opc.feasible[0] and vc >= vs:
        out.power_cruise_w = float(opc.electrical_power_w[0])
        out.wh_per_km = out.power_cruise_w / (3.6 * vc)
        out.motor_current_cruise_a = float(np.max(opc.motor_current_a[:, 0]))
    else:
        out.notes.append(f"cruise {vc:.1f} m/s not reachable in level flight")
    _static_currents(out, trains, battery, rho)
    return out


def _static_currents(out: Performance, trains, battery, rho):
    st = prop.max_thrust(trains, battery, [0.0], rho)
    out.motor_current_max_a = float(np.max(st.motor_current_a[:, 0]))
    out.battery_current_max_a = float(st.battery_current_a[0])
