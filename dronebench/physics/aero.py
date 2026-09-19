"""3D aerodynamics: lift slope, CLmax, Oswald e, CD0 build-up, neutral point, tail volumes.

Equations (low-speed, incompressible, beta = 1):
  * Lift slope (Helmbold / DATCOM):
      a = 2 pi AR / (2 + sqrt(AR^2 / kappa^2 * (1 + tan^2 L_c/2) + 4)),  kappa = a0 / (2 pi)
    with a0 the section slope at the surface's root airfoil and L_c/2 the half-chord sweep.
  * Wing CLmax = 0.9 CLmax_2D(Re at MAC) (clean wing, flaps ignored).
  * Oswald e (Raymer, straight wing): 1.78 (1 - 0.045 AR^0.68) - 0.64, clamped 0.6 - 0.9.
  * CD0 = 1.15 * sum_i Cf_i FF_i Swet_i / S_ref.
      Cf: laminar run to Re_x = 5e4 then turbulent (early transition: printed-layer lines, seams,
          tape trip the boundary layer near the leading edge); Cf = f Cf_lam + (1 - f) Cf_turb,
          f = min(1, 5e4 / Re_L),
          Cf_lam = 1.328 / sqrt(Re), Cf_turb = 0.455 / (log10 Re)^2.58 (Schlichting).
      FF surfaces = 1 + 0.6 / 0.3 (t/c) + 100 (t/c)^4;  FF fuselage = 1 + 60 / f^3 + f / 400.
      Swet: geom.wetted_area_m2 when given, else 2.04 x planform (surfaces, fuselage overlap
          ignored: conservative) and pi d L 0.8 (fuselage). Other wetted entries (nacelles,
          pods, gear) use FF 1.25 and Re on sqrt(Swet).
      +15 %: interference, gaps, control-surface hinges, printed-surface roughness, cooling and
          antennas (Raymer suggests 5-10 % for clean airframes; printed hobby airframes are
          rougher, so 15 %).
  * Neutral point: x_np = [x_ac_w a_w + eta a_t St/S (1 - de/da) x_ac_t] / [a_w + eta a_t St/S (1 - de/da)],
      eta = 0.9, de/da = 2 a_w / (pi AR) (per rad), then shifted forward by the fuselage term
      dx = K_f 57.3 W_f^2 L_f / (S a_w) with K_f = 0.012 per deg (Raymer Fig. 16.14, wing near
      30-40 % of fuselage length).
  * Tail volumes: Vh = S_h (x_ac_h - x_ac_w) / (S MAC);  Vv = S_v (x_ac_v - x_ac_w) / (S b).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..model import AeroGeometry, SurfaceGeom
from . import airfoil as af

MU_AIR = 1.789e-5          # Pa s, sea-level standard
MISC_FACTOR = 1.15
TAIL_EFF = 0.9
K_FUS_PER_DEG = 0.012
RE_TRANSITION = 5e4


def finite_pos(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x)) and float(x) > 0
    except (TypeError, ValueError):
        return False


def surface_airfoil(s: SurfaceGeom) -> af.Airfoil:
    return af.parse_naca(s.airfoil, s.thickness_ratio, s.camber)


def reynolds(V, length, rho=1.225):
    return rho * np.asarray(V, float) * length / MU_AIR


def sweep_at(s: SurfaceGeom, frac: float) -> float:
    """Sweep (rad) of the frac-chord line from LE sweep, taper and AR (straight-tapered approx)."""
    ar = s.aspect_ratio if finite_pos(s.aspect_ratio) else 6.0
    lam = s.taper if (s.taper is not None and math.isfinite(s.taper) and s.taper >= 0) else 1.0
    le = math.radians(s.sweep_le_deg or 0.0) if math.isfinite(s.sweep_le_deg or 0.0) else 0.0
    return math.atan(math.tan(le) - 4 * frac * (1 - lam) / (ar * (1 + lam)))


def lift_slope_3d(s: SurfaceGeom, ar: Optional[float] = None) -> float:
    """Helmbold / DATCOM lift-curve slope per rad."""
    ar = ar if ar is not None else s.aspect_ratio
    if not finite_pos(ar):
        ar = 6.0
    a0 = af.lift_slope_2d(surface_airfoil(s))
    kappa = a0 / (2 * math.pi)
    tl = math.tan(sweep_at(s, 0.5))
    return 2 * math.pi * ar / (2 + math.sqrt(ar**2 / kappa**2 * (1 + tl**2) + 4))


def oswald_e(ar: float) -> float:
    if not finite_pos(ar):
        return 0.75
    e = 1.78 * (1 - 0.045 * ar**0.68) - 0.64
    return float(min(max(e, 0.6), 0.9))


def wing_cl_max(s: SurfaceGeom, V: float, rho: float = 1.225) -> float:
    mac = s.mac_m if finite_pos(s.mac_m) else 0.2
    re_ = float(reynolds(V, mac, rho))
    return 0.9 * af.cl_max_2d(surface_airfoil(s), re_)


def skin_friction(re_):
    re_ = np.maximum(np.asarray(re_, float), 1e3)
    f = np.minimum(1.0, RE_TRANSITION / re_)
    return f * 1.328 / np.sqrt(re_) + (1 - f) * 0.455 / np.log10(re_) ** 2.58


def cd0_buildup(geom: AeroGeometry, s_ref: float, V, rho: float = 1.225):
    """CD0 (array over V) and a breakdown dict {component: CD0 at V[0]}, plus notes."""
    V = np.atleast_1d(np.asarray(V, float))
    wet = dict(geom.wetted_area_m2 or {})
    total = np.zeros(V.size)
    parts: dict[str, float] = {}
    notes: list[str] = []
    for name in ("wing", "htail", "vtail"):
        s: Optional[SurfaceGeom] = getattr(geom, name)
        if s is None or not finite_pos(s.area_m2):
            wet.pop(name, None)
            continue
        swet = wet.pop(name, None)
        if not finite_pos(swet):
            swet = 2.04 * s.area_m2
            notes.append(f"{name} wetted area estimated as 2.04 x planform")
        tc = surface_airfoil(s).thickness
        ff = 1 + 0.6 / 0.3 * tc + 100 * tc**4
        mac = s.mac_m if finite_pos(s.mac_m) else s.area_m2 / max(s.span_m or 1.0, 1e-3)
        cd = skin_friction(reynolds(V, mac, rho)) * ff * swet / s_ref
        parts[name] = float(cd[0])
        total += cd
    L, d = geom.fuselage_length_m, geom.fuselage_max_diam_m
    swet = wet.pop("fuselage", None)
    if finite_pos(L) and finite_pos(d):
        if not finite_pos(swet):
            swet = math.pi * d * L * 0.8
            notes.append("fuselage wetted area estimated as 0.8 pi d L")
        fr = L / d
        ff = 1 + 60 / fr**3 + fr / 400
        cd = skin_friction(reynolds(V, L, rho)) * ff * swet / s_ref
        parts["fuselage"] = float(cd[0])
        total += cd
    elif finite_pos(swet):
        cd = skin_friction(reynolds(V, math.sqrt(swet), rho)) * 1.25 * swet / s_ref
        parts["fuselage"] = float(cd[0])
        total += cd
    else:
        notes.append("no fuselage geometry: fuselage drag omitted")
    for name, swet in wet.items():
        if not finite_pos(swet) or name in ("wing", "htail", "vtail", "total"):
            continue
        cd = skin_friction(reynolds(V, math.sqrt(swet), rho)) * 1.25 * swet / s_ref
        parts[name] = float(cd[0])
        total += cd
    total *= MISC_FACTOR
    parts["misc_15pct"] = float(total[0] * (1 - 1 / MISC_FACTOR))
    return total, parts, notes


def neutral_point(geom: AeroGeometry, a_w: float) -> tuple[float, list[str]]:
    """Stick-fixed neutral point x (m). Requires geom.wing."""
    w = geom.wing
    notes: list[str] = []
    num, den = w.x_ac_m * a_w, a_w
    h = geom.htail
    if h is not None and finite_pos(h.area_m2):
        a_t = lift_slope_3d(h)
        deda = min(2 * a_w / (math.pi * max(w.aspect_ratio, 1.0)), 0.9)
        k = TAIL_EFF * a_t * h.area_m2 / w.area_m2 * (1 - deda)
        num += k * h.x_ac_m
        den += k
    else:
        notes.append("no horizontal tail: neutral point = wing aerodynamic centre")
    x_np = num / den
    L, d = geom.fuselage_length_m, geom.fuselage_max_diam_m
    if finite_pos(L) and finite_pos(d):
        x_np -= K_FUS_PER_DEG * 57.3 * d**2 * L / (w.area_m2 * a_w)
    return float(x_np), notes


def tail_volumes(geom: AeroGeometry) -> tuple[float, float]:
    w = geom.wing
    vh = vv = float("nan")
    if geom.htail is not None and finite_pos(geom.htail.area_m2):
        vh = geom.htail.area_m2 * (geom.htail.x_ac_m - w.x_ac_m) / (w.area_m2 * w.mac_m)
    elif geom.htail is None:
        vh = 0.0
    if geom.vtail is not None and finite_pos(geom.vtail.area_m2):
        vv = geom.vtail.area_m2 * (geom.vtail.x_ac_m - w.x_ac_m) / (w.area_m2 * w.span_m)
    elif geom.vtail is None:
        vv = 0.0
    return float(vh), float(vv)
