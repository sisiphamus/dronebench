"""Propeller, motor, ESC and battery models, plus operating-point solvers.

Everything is vectorised over airspeed (numpy arrays) so the performance sweep costs a few
milliseconds. Units: SI, rpm where a name says so, n = revolutions per second.

Generic propeller model (thin-electric APC-style props), r = pitch / diameter, x = J / J0:
  J0   = 1.10 r + 0.05                   advance ratio of zero thrust (effective pitch > geometric
                                         because of blade camber; UIUC 11x8E: CT -> 0 near J 0.85)
  CT   = CT0 (1 - x^1.3)                 CT0 = 0.065 + 0.073 r
  CP   = CP0 (1 - 0.6 x^2)               CP0 = CT0 / k,  k = eta_pk / (0.3735 J0),
                                         eta_pk = 0.60 + 0.17 r (clamped 0.55 - 0.80)
  0.3735 is max_x x (1 - x^1.3) / (1 - 0.6 x^2), so the peak efficiency J CT / CP equals eta_pk
  (at x ~ 0.64). Shape exponents fitted to UIUC (Brandt & Selig) APC 11x8E curves: CT ~0.04,
  CP ~0.032, eta ~0.74 at J = 0.6. Checks for an APC 11x8E (r = 0.727): CT0 = 0.118,
  CP0 = 0.052, static thrust 2.0 kgf at 9000 rpm (APC data ~2.0-2.2), peak efficiency 0.72 at
  J ~ 0.54. Thrust T = CT rho n^2 D^4, power P = CP rho n^3 D^5, torque Q = P / (2 pi n).
  Extra blades: CT0 and CP0 scale by (blades / 2)^0.8 (empirical; efficiency unchanged).

Motor (first-order DC model): Kt = 30 / (pi KV) [N m / A], I = Q / Kt + I0,
  V_motor = I Rm + rpm / KV, electrical input P = V_motor I. I0 is taken as constant.

ESC: power efficiency 0.95; at full throttle it delivers 0.98 of battery voltage (FET drop).

Battery: open-circuit voltage = cells x nominal (LiPo 3.7 V, Li-ion 3.6 V: the discharge-average,
  not the full-charge 4.2 V), pack resistance R = cells_s x ir / (cells_p x parallel_packs) where
  ir_mohm_per_cell is per individual cell. Terminal voltage V = V_oc - I R. Energy drawn from
  the cells is V_oc I (so internal-resistance heat is charged against the energy budget).
  Thrust is shared equally between trains (identical-train assumption for mixed trains).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..model import BatterySpec, PowerTrain

INCH = 0.0254
ESC_EFF = 0.95
ESC_VFRAC = 0.98
CELL_V_NOM = {"lipo": 3.7, "liion": 3.6, "li-ion": 3.6, "lihv": 3.8, "life": 3.3, "nimh": 1.2}
_ETA_SHAPE = 0.3735


# ---------------------------------------------------------------- propeller

@dataclass
class PropCoeffs:
    diam_m: float
    r: float
    ct0: float
    cp0: float
    j0: float


def prop_coeffs(diam_in: float, pitch_in: float, blades: int = 2) -> PropCoeffs:
    d = max(float(diam_in), 1.0) * INCH
    r = min(max(float(pitch_in) / max(float(diam_in), 1e-6), 0.2), 1.6)
    j0 = 1.10 * r + 0.05
    ct0 = 0.065 + 0.073 * r
    eta_pk = min(max(0.60 + 0.17 * r, 0.55), 0.80)
    cp0 = ct0 / (eta_pk / (_ETA_SHAPE * j0))
    bf = (max(int(blades or 2), 2) / 2.0) ** 0.8
    return PropCoeffs(d, r, ct0 * bf, cp0 * bf, j0)


def ct_cp(pc: PropCoeffs, J):
    """CT, CP at advance ratio J (array ok). CP is floored at 0 (no windmilling power)."""
    x = np.clip(np.asarray(J, dtype=float) / pc.j0, 0.0, None)
    ct = pc.ct0 * (1.0 - x**1.3)
    cp = np.maximum(pc.cp0 * (1.0 - 0.6 * x**2), 0.0)
    return ct, cp


def prop_thrust(pc: PropCoeffs, n, V, rho):
    n = np.maximum(np.asarray(n, float), 1e-9)
    ct, _ = ct_cp(pc, np.asarray(V, float) / (n * pc.diam_m))
    return ct * rho * n**2 * pc.diam_m**4


def prop_torque(pc: PropCoeffs, n, V, rho):
    n = np.maximum(np.asarray(n, float), 1e-9)
    _, cp = ct_cp(pc, np.asarray(V, float) / (n * pc.diam_m))
    return cp * rho * n**2 * pc.diam_m**5 / (2 * math.pi)


# ---------------------------------------------------------------- motor

def motor_state(train: PowerTrain, pc: PropCoeffs, n, V, rho):
    """Current and voltage the motor needs to spin the prop at n rev/s at airspeed V."""
    m = train.motor
    kv = max(float(m.kv), 1.0)
    kt = 30.0 / (math.pi * kv)
    q = prop_torque(pc, n, V, rho)
    i = q / kt + max(float(m.i0_a), 0.0)
    vm = i * max(float(m.rm_ohm), 0.0) + 60.0 * np.asarray(n, float) / kv
    return i, vm


# ---------------------------------------------------------------- battery

def cell_v_nom(b: BatterySpec) -> float:
    return CELL_V_NOM.get((b.chemistry or "lipo").lower(), 3.7)


def pack_parallel(b: BatterySpec) -> int:
    return max(int(getattr(b, "cells_p", 1) or 1), 1) * max(int(b.parallel_packs or 1), 1)


def battery_voc(b: BatterySpec) -> float:
    return max(int(b.cells_s), 1) * cell_v_nom(b)


def battery_resistance(b: BatterySpec) -> float:
    return max(int(b.cells_s), 1) * max(float(b.ir_mohm_per_cell), 0.0) * 1e-3 / pack_parallel(b)


def battery_capacity_ah(b: BatterySpec) -> float:
    """Total capacity (capacity_mah is per pack; packs in parallel add)."""
    return max(float(b.capacity_mah), 0.0) / 1000.0 * max(int(b.parallel_packs or 1), 1)


def battery_energy_wh(b: BatterySpec) -> float:
    return battery_voc(b) * battery_capacity_ah(b)


def _battery_current(b: BatterySpec, p_term):
    """Battery current delivering terminal power p_term (array). NaN if beyond max power."""
    voc, r = battery_voc(b), battery_resistance(b)
    p = np.asarray(p_term, float)
    if r <= 0:
        return p / voc
    disc = voc**2 - 4 * r * p
    with np.errstate(invalid="ignore"):
        return np.where(disc >= 0, (voc - np.sqrt(np.maximum(disc, 0))) / (2 * r), np.nan)


# ---------------------------------------------------------------- solvers

def _bisect(f, lo, hi, iters=48):
    """Vectorised bisection for f increasing in x; returns x with f(x) ~ 0 (clipped to [lo, hi])."""
    lo = np.array(lo, float)
    hi = np.array(hi, float)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return 0.5 * (lo + hi)


def _unique_trains(trains: Sequence[PowerTrain]):
    """Group identical trains so each is solved once: [(train, PropCoeffs, [indices])]."""
    groups: dict[tuple, list] = {}
    for k, t in enumerate(trains):
        m, p = t.motor, t.prop
        key = (m.kv, m.rm_ohm, m.i0_a, p.diam_in, p.pitch_in, p.blades)
        if key not in groups:
            groups[key] = [t, prop_coeffs(p.diam_in, p.pitch_in, p.blades), []]
        groups[key][2].append(k)
    return [tuple(g) for g in groups.values()]


def _n_bounds(train: PowerTrain, pc: PropCoeffs, V, vmax_motor):
    n_lo = np.asarray(V, float) / (pc.diam_m * pc.j0) + 1e-6       # zero-thrust rpm
    n_hi = np.maximum(max(float(train.motor.kv), 1.0) * vmax_motor * 1.5 / 60.0, n_lo * 4 + 10)
    return n_lo, n_hi


@dataclass
class OperatingPoint:
    throttle: np.ndarray              # max over trains, 0-1 (>1 = infeasible)
    motor_current_a: np.ndarray       # (n_trains, nV)
    motor_rpm: np.ndarray             # (n_trains, nV)
    battery_current_a: np.ndarray
    battery_voltage_v: np.ndarray
    electrical_power_w: np.ndarray    # drawn from the cells, V_oc * I_batt
    feasible: np.ndarray


def operating_points(trains: Sequence[PowerTrain], battery: BatterySpec, V, thrust_total, rho) -> OperatingPoint:
    """Vectorised steady operating point: total thrust thrust_total at airspeed V (arrays)."""
    V = np.atleast_1d(np.asarray(V, float))
    T = np.broadcast_to(np.atleast_1d(np.asarray(thrust_total, float)), V.shape)
    nt = len(trains)
    voc = battery_voc(battery)
    t_each = np.maximum(T, 0.0) / nt
    I = np.zeros((nt, V.size))
    VM = np.zeros((nt, V.size))
    RPM = np.zeros((nt, V.size))
    for tr, pc, idx in _unique_trains(trains):
        lo, hi = _n_bounds(tr, pc, V, voc)
        n = _bisect(lambda x: prop_thrust(pc, x, V, rho) - t_each, lo, hi, iters=34)
        i, vm = motor_state(tr, pc, n, V, rho)
        I[idx], VM[idx], RPM[idx] = i, vm, n * 60
    p_term = (VM * I).sum(axis=0) / ESC_EFF
    ib = _battery_current(battery, p_term)
    vb = voc - battery_resistance(battery) * ib
    with np.errstate(invalid="ignore", divide="ignore"):
        thr = (VM / (ESC_VFRAC * vb)).max(axis=0)
    feas = np.isfinite(ib) & (thr <= 1.0)
    return OperatingPoint(thr, I, RPM, ib, vb, voc * ib, feas)


def operating_point(trains: Sequence[PowerTrain], battery: Optional[BatterySpec], V_airspeed: float,
                    thrust_required: float, rho: float = 1.225) -> Optional[dict]:
    """Scalar convenience wrapper. Returns None if the thrust cannot be reached (or no power)."""
    if not trains or battery is None:
        return None
    op = operating_points(trains, battery, [V_airspeed], [thrust_required], rho)
    if not op.feasible[0]:
        return None
    return {"throttle": float(op.throttle[0]), "motor_current_a": [float(x) for x in op.motor_current_a[:, 0]],
            "motor_rpm": [float(x) for x in op.motor_rpm[:, 0]],
            "battery_current_a": float(op.battery_current_a[0]), "battery_voltage_v": float(op.battery_voltage_v[0]),
            "electrical_power_w": float(op.electrical_power_w[0])}


@dataclass
class MaxThrust:
    thrust_n: np.ndarray              # total, per airspeed
    motor_current_a: np.ndarray       # (n_trains, nV)
    battery_current_a: np.ndarray
    battery_voltage_v: np.ndarray
    notes: list[str] = field(default_factory=list)


def max_thrust(trains: Sequence[PowerTrain], battery: Optional[BatterySpec], V, rho: float = 1.225) -> MaxThrust:
    """Full-throttle thrust vs airspeed. Battery sag is solved by fixed-point iteration."""
    V = np.atleast_1d(np.asarray(V, float))
    nt = len(trains)
    if nt == 0 or battery is None:
        z = np.zeros(V.size)
        return MaxThrust(z, np.zeros((nt, V.size)), z, z, ["no power train or battery"])
    voc, rb = battery_voc(battery), battery_resistance(battery)
    groups = _unique_trains(trains)
    vb = np.full(V.size, voc)
    n_prev: list = [None] * len(groups)
    Iu = [np.zeros(V.size)] * len(groups)
    thrust = np.zeros(V.size)
    for it in range(12):
        vavail = ESC_VFRAC * vb
        thrust = np.zeros(V.size)
        pm = np.zeros(V.size)
        for g, (tr, pc, idx) in enumerate(groups):
            lo, hi = _n_bounds(tr, pc, V, voc)
            if n_prev[g] is not None:     # warm start: battery voltage moves little between passes
                lo, hi = np.maximum(lo, 0.8 * n_prev[g]), np.minimum(hi, 1.2 * n_prev[g] + 1.0)
            n = _bisect(lambda x: motor_state(tr, pc, x, V, rho)[1] - vavail, lo, hi, iters=30 if it == 0 else 16)
            n_prev[g] = n
            i, vm = motor_state(tr, pc, n, V, rho)
            Iu[g] = i
            pm += len(idx) * vm * i
            thrust += len(idx) * prop_thrust(pc, n, V, rho)
        ib = pm / (ESC_EFF * vb)
        vb_new = np.maximum(voc - rb * ib, 0.3 * voc)
        done = np.max(np.abs(vb_new - vb)) < 5e-3
        vb = 0.4 * vb + 0.6 * vb_new
        if done:
            break
    I = np.zeros((nt, V.size))
    for g, (_, _, idx) in enumerate(groups):
        I[idx] = Iu[g]
    ib = (ESC_VFRAC * vb * I).sum(axis=0) / (ESC_EFF * vb)
    notes = []
    if np.any((vb < 3.0 * max(int(battery.cells_s), 1)) & (thrust > 0.05 * max(float(thrust.max()), 1e-9))):
        notes.append("battery sags below 3.0 V/cell at full throttle")
    return MaxThrust(np.maximum(thrust, 0.0), I, ib, vb, notes)
