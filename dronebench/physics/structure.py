"""Wing spar bending and control-surface servo torque.

Spar: each half-wing is a cantilever from the root. Lift at load factor n (all-up weight; the
  relief from wing self-weight is ignored, conservative) is elliptic along each semispan s:
  L'(eta) = L0 sqrt(1 - (eta/s)^2), L0 = 2 n W / (pi s). Bending moment outboard of station y,
  with u = y / s (closed form of M(y) = int_y^s L'(eta)(eta - y) d eta):
      M(y) = L0 s^2 [ (1 - u^2)^1.5 / 3 - u (pi/4 - (u sqrt(1 - u^2) + asin u) / 2) ]
  At the root M = n W b / (3 pi). Checked at 21 stations from root to tip plus just outboard of
  every spar end. At each station the spars covering it (SparSpec.y_min_m / y_max_m contain |y|
  or -|y|; None = whole wing; spans_root False with no range = [0.03 m, tip]) share M by EI.
  Tubes: I = pi (D^4 - d^4) / 64; stress in tube i: sigma_i = M_i (D_i / 2) / I_i with
  M_i = M E_i I_i / sum(E I). Nested tubes (a doubler sleeved in the main spar) simply add EI.
  SF = min over stations and tubes; the critical station goes into the notes.
  A station with no spar: SF 0.3 (printed skin alone is not rated), except where M(y) < 5 % of the
  root moment (outer ~25 % of the semispan): the printed skin and ribs carry that tip moment.
  Tail booms (role tail_boom) are skipped; solid rods (inner 0) are fine.
  Default strengths when Material.tensile_mpa is missing: CF tube 600 MPa (pultruded/rolled
  bending strength is 800-1000, 600 is conservative), aluminium 240 (6061-T6 yield), steel 350,
  wood 60, fibreglass 300, printed/polymer 40. Default moduli (GPa): CF 100, Al 69, steel 200,
  wood 10, GF 25, polymer 3.
  No spar: SF 0.3 (printed skin alone is not rated).

Servo: hinge moment H = Cq q_design S_cs c_cs, Cq = 0.12 (= Ch_delta ~0.35 /rad x ~20 deg
  (0.35 rad) deflection, plain unbalanced surface; the deflection actually used at high speed),
  q_design = 0.5 rho V_design^2 with V_design = min(Vmax, 2 x cruise) (fallback 1.3 x cruise when
  Vmax is unknown). Required torque in kg cm = H / 0.0981. Linkage mechanical advantage assumed
  1:1. Margin = rated / required. Sanity check: the Falcon V2 flies on 2.0 kg cm ES08MA II servos.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from ..model import ControlSurfaceGeom, Material, ServoSpec, SparSpec

CQ_HINGE = 0.12
NO_SPAR_SF = 0.3


def _kind(m: Material) -> str:
    s = f"{m.name} {m.kind}".lower()
    if "carbon" in s or "cf" in s.split() or "cf " in s or s.startswith("cf") or "composite" in s:
        return "cf"
    if "glass" in s or "gf" in s.split() or "g10" in s or "fr4" in s:
        return "gf"
    if "alu" in s or "6061" in s or "7075" in s or "al " in s or s.startswith("al"):
        return "al"
    if "steel" in s or "metal" in s:
        return "steel" if "steel" in s else "al"
    if "wood" in s or "balsa" in s or "spruce" in s or "bamboo" in s or "ply" in s:
        return "wood"
    return "polymer"


_STRENGTH = {"cf": 600.0, "gf": 300.0, "al": 240.0, "steel": 350.0, "wood": 60.0, "polymer": 40.0}
_MODULUS = {"cf": 100.0, "gf": 25.0, "al": 69.0, "steel": 200.0, "wood": 10.0, "polymer": 3.0}


def spar_strength_mpa(m: Material) -> tuple[float, bool]:
    if m.tensile_mpa and math.isfinite(m.tensile_mpa) and m.tensile_mpa > 0:
        return float(m.tensile_mpa), False
    return _STRENGTH[_kind(m)], True


def spar_modulus_gpa(m: Material) -> float:
    if m.modulus_gpa and math.isfinite(m.modulus_gpa) and m.modulus_gpa > 0:
        return float(m.modulus_gpa)
    return _MODULUS[_kind(m)]


def root_moment(weight_n: float, span_m: float, load_factor: float) -> float:
    return load_factor * weight_n * span_m / (3 * math.pi)


def moment_at(y: float, weight_n: float, span_m: float, load_factor: float) -> float:
    """Elliptic-lift bending moment at |y| (closed form, see module docstring)."""
    s = span_m / 2
    u = min(abs(y) / s, 1.0)
    l0 = 2 * load_factor * weight_n / (math.pi * s)
    r = math.sqrt(max(1 - u * u, 0.0))
    return l0 * s * s * (r**3 / 3 - u * (math.pi / 4 - (u * r + math.asin(u)) / 2))


SKIN_ONLY_FRACTION = 0.05
N_STATIONS = 21


def _extent(sp: SparSpec, semispan: float) -> tuple[float, float]:
    lo = sp.y_min_m if sp.y_min_m is not None and math.isfinite(sp.y_min_m) else None
    hi = sp.y_max_m if sp.y_max_m is not None and math.isfinite(sp.y_max_m) else None
    if lo is None:
        lo = 0.03 if sp.spans_root is False else -semispan
    if hi is None:
        hi = semispan
    return min(lo, hi), max(lo, hi)


def _covers(ext: tuple[float, float], y: float) -> bool:
    lo, hi = ext
    return lo - 1e-9 <= y <= hi + 1e-9 or lo - 1e-9 <= -y <= hi + 1e-9


def spar_safety_factor(spars: Sequence[SparSpec], weight_n: float, span_m: float,
                       load_factor: float) -> tuple[float, list[str]]:
    notes: list[str] = []
    semispan = span_m / 2
    wing_spars = [s for s in spars if (s.role or "main_spar") != "tail_boom" and s.outer_d_mm and s.outer_d_mm > 0]
    if not wing_spars:
        return NO_SPAR_SF, ["no spar: printed skin alone not rated (spar_sf = 0.3)"]
    m_root = root_moment(weight_n, span_m, load_factor)
    props = []
    for s in wing_spars:
        D = s.outer_d_mm / 1000
        d = min(max(s.inner_d_mm or 0.0, 0.0), s.outer_d_mm * 0.999) / 1000
        E = spar_modulus_gpa(s.material) * 1e9
        strength, assumed = spar_strength_mpa(s.material)
        if assumed:
            notes.append(f"{s.id}: strength assumed {strength:.0f} MPa ({_kind(s.material)})")
        props.append((s, _extent(s, semispan), E, E * math.pi * (D**4 - d**4) / 64, D / 2, strength * 1e6))
    stations = {round(y, 6) for y in (semispan * i / (N_STATIONS - 1) for i in range(N_STATIONS - 1))}
    for _, (lo, hi), *_ in props:
        for e in (abs(lo), abs(hi)):
            if e + 1e-4 < semispan:
                stations.add(round(e + 1e-4, 6))
    best = (float("inf"), 0.0, "")
    for y in sorted(stations):
        m = moment_at(y, weight_n, span_m, load_factor)
        here = [p for p in props if _covers(p[1], y)]
        if not here:
            if m >= SKIN_ONLY_FRACTION * m_root and NO_SPAR_SF < best[0]:
                best = (NO_SPAR_SF, y, "no spar (printed skin alone not rated)")
            continue
        sum_ei = sum(p[3] for p in here)
        for s, _, E, _, c, strength in here:
            sigma = m * E * c / sum_ei
            sf = strength / sigma if sigma > 0 else float("inf")
            if sf < best[0]:
                best = (sf, y, f"{s.id} ({'+'.join(p[0].id for p in here)} share M = {m:.1f} N m)")
    notes.append(f"root bending moment {m_root:.1f} N m at n = {load_factor}")
    if math.isfinite(best[0]):
        notes.append(f"spar critical at y={best[1]:.2f} m: {best[2]}")
    return float(best[0]), notes


def servo_margins(servos: Sequence[ServoSpec], surfaces: Sequence[ControlSurfaceGeom],
                  q_max: float) -> tuple[dict[str, float], list[str]]:
    by_id = {c.id: c for c in surfaces}
    out: dict[str, float] = {}
    notes: list[str] = []
    for sv in servos:
        cs: Optional[ControlSurfaceGeom] = by_id.get(sv.surface_id) if sv.surface_id else None
        if cs is None:
            out[sv.id] = float("nan")
            notes.append(f"servo {sv.id}: surface '{sv.surface_id}' not found, margin not computed")
            continue
        h = CQ_HINGE * q_max * max(cs.area_m2, 0.0) * max(cs.chord_m, 0.0)
        req = h / 0.0981
        out[sv.id] = float(sv.torque_kgcm / req) if req > 0 else float("inf")
    return out, notes
