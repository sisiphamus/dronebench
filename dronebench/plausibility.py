"""Plausibility clamps for agent-supplied specs.

`check_specs(part)` clamps implausible values IN PLACE, tags each changed field
`part.sources[field] = "agent_supplied_clamped"` and returns [(field, old, new, reason)].
The pipeline runs it on every part that has an agent_supplied source tag (edits.py writes those).

Limits (all deliberately generous; they catch fantasy numbers, not optimistic ones):
  battery   pack energy = S x 3.7 V (LiPo) or 3.6 V (Li-ion) x Ah. Specific energy <= 200 Wh/kg (LiPo)
            or 270 Wh/kg (Li-ion cells); above that mass_g is raised to energy / limit. C rating <= 150.
  motor     kv 100-5000; rm_ohm >= 0.005; power density <= 8 W/g with P = imax_a x V_max, where V_max =
            4.2 V x specs.cells_max if given, else 25.2 V (6S full charge, the largest pack this benchmark
            flies) -> imax_a clamped to 8 x mass_g / V_max. A motor rated >= 20 A must weigh > 5 g.
  servo     torque_kgcm <= 1.2 x mass_g (the best hobby servos reach ~0.5-0.8 kg·cm per gram).
  prop      diam 3-30 in; pitch 0.3-1.5 x diam.
  esc       imax_a <= 3 A per gram.
  any       (checked first) if the part has a mesh and specs.mass_g differs > 3x from volume x material
            density, the mesh estimate wins; the rules above then use that mass.
"""
from __future__ import annotations

from typing import Any

from .model import Part

CLAMPED = "agent_supplied_clamped"
WH_PER_KG = {"lipo": 200.0, "liion": 270.0}
CELL_V = {"lipo": 3.7, "liion": 3.6}
MAX_C = 150.0
MOTOR_W_PER_G = 8.0
DEFAULT_VMAX = 25.2
SERVO_KGCM_PER_G = 1.2
ESC_A_PER_G = 3.0
MESH_RATIO = 3.0


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _chem(part: Part) -> str:
    c = str(part.specs.get("chemistry", "lipo")).lower().replace("-", "").replace(" ", "")
    return "liion" if c in ("liion", "li", "lion", "18650", "21700") else "lipo"


def _set(part: Part, changes: list, field: str, new: Any, reason: str) -> None:
    old = part.specs.get(field)
    if isinstance(new, float):
        new = round(new, 4)
    part.specs[field] = new
    part.sources[field] = CLAMPED
    changes.append((field, old, new, reason))


def _clamp_range(part, changes, field, lo, hi, what):
    v = _num(part.specs.get(field))
    if v is None:
        return
    if lo is not None and v < lo:
        _set(part, changes, field, float(lo), f"{what} {v:g} below plausible minimum {lo:g}")
    elif hi is not None and v > hi:
        _set(part, changes, field, float(hi), f"{what} {v:g} above plausible maximum {hi:g}")


def mesh_mass_g(part: Part) -> float | None:
    """Mass from mesh volume x material density (printed parts via materials.printed_mass_kg)."""
    if part.mesh is None or not part.volume_m3:
        return None
    try:
        from . import materials
        mat = part.material or materials.resolve_material(part)
        if mat.kind == "printed" and part.area_m2:
            return materials.printed_mass_kg(part.volume_m3, part.area_m2, mat) * 1000
        if mat.density_kg_m3 > 0:
            return part.volume_m3 * mat.density_kg_m3 * 1000
    except Exception:
        pass
    return None


def check_specs(part: Part) -> list[tuple[str, Any, Any, str]]:
    changes: list = []
    s = part.specs
    t = part.type
    # mesh first, so the per-gram rules below use the corrected mass
    mass = _num(s.get("mass_g"))
    est = mesh_mass_g(part) if mass else None
    if est and (mass > MESH_RATIO * est or mass < est / MESH_RATIO):
        _set(part, changes, "mass_g", est,
             f"stated {mass:g} g is > {MESH_RATIO:g}x off the mesh estimate {est:.0f} g; mesh used")
    if t == "battery":
        chem = _chem(part)
        cells, cap, mass = _num(s.get("cells_s")), _num(s.get("capacity_mah")), _num(s.get("mass_g"))
        if cells and cap and mass:
            wh = cells * CELL_V[chem] * cap / 1000
            if wh / (mass / 1000) > WH_PER_KG[chem]:
                need = wh / WH_PER_KG[chem] * 1000
                _set(part, changes, "mass_g", need,
                     f"{wh:.0f} Wh in {mass:g} g is {wh / mass * 1000:.0f} Wh/kg > {WH_PER_KG[chem]:.0f} Wh/kg "
                     f"({chem}); mass raised")
        _clamp_range(part, changes, "c_rating", None, MAX_C, "C rating")
    elif t == "motor":
        _clamp_range(part, changes, "kv", 100, 5000, "kv")
        _clamp_range(part, changes, "rm_ohm", 0.005, None, "winding resistance (ohm)")
        imax, mass = _num(s.get("imax_a")), _num(s.get("mass_g"))
        vmax = 4.2 * _num(s.get("cells_max")) if _num(s.get("cells_max")) else DEFAULT_VMAX
        if imax and mass:
            limit = MOTOR_W_PER_G * mass / vmax
            if imax > limit:
                _set(part, changes, "imax_a", limit,
                     f"{imax:g} A x {vmax:g} V = {imax * vmax:.0f} W from {mass:g} g is "
                     f"{imax * vmax / mass:.0f} W/g > {MOTOR_W_PER_G:g} W/g")
                imax = limit
        if imax and imax >= 20 and mass is not None and mass <= 5:
            _set(part, changes, "mass_g", 5.1, f"a motor rated {imax:g} A cannot weigh {mass:g} g")
    elif t == "servo":
        tq, mass = _num(s.get("torque_kgcm")), _num(s.get("mass_g"))
        if tq and mass and tq > SERVO_KGCM_PER_G * mass:
            _set(part, changes, "torque_kgcm", SERVO_KGCM_PER_G * mass,
                 f"{tq:g} kg·cm from a {mass:g} g servo > {SERVO_KGCM_PER_G} kg·cm/g")
    elif t == "prop":
        _clamp_range(part, changes, "diam_in", 3, 30, "prop diameter (in)")
        d, p = _num(s.get("diam_in")), _num(s.get("pitch_in"))
        if d and p:
            _clamp_range(part, changes, "pitch_in", 0.3 * d, 1.5 * d, "prop pitch (in)")
    elif t == "esc":
        imax, mass = _num(s.get("imax_a")), _num(s.get("mass_g"))
        if imax and mass and imax > ESC_A_PER_G * mass:
            _set(part, changes, "imax_a", ESC_A_PER_G * mass,
                 f"{imax:g} A from a {mass:g} g ESC > {ESC_A_PER_G:g} A/g")

    return changes


def needs_check(part: Part) -> bool:
    return any(str(v).startswith("agent_supplied") for v in part.sources.values())
