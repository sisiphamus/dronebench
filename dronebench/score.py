"""Preset scores (PLAN §1.7). 0 when the gate fails."""
from __future__ import annotations

import math

from .model import GateResult, PhysicsResult

PRESETS = {
    "endurance": ("min", lambda r: r.endurance_min),
    "range": ("km", lambda r: r.range_km),
    "speed": ("km/h", lambda r: r.vmax_ms * 3.6),
    "payload": ("fraction", lambda r: r.payload_fraction),
    "efficiency": ("km/Wh", lambda r: r.km_per_wh),
}


def compute_score(preset: str, pres: PhysicsResult, gate: GateResult) -> tuple[float, str]:
    """(value, unit) for a preset. Unknown preset raises ValueError."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; one of {', '.join(PRESETS)}")
    unit, fn = PRESETS[preset]
    if gate is not None and not gate.passed:
        return 0.0, unit
    if pres is None or not pres.ok:
        return 0.0, unit
    v = fn(pres)
    if v is None or not math.isfinite(v) or v < 0:
        return 0.0, unit
    return float(v), unit
