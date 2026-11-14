"""Merge the flown 6-DOF evaluation into a report produced by the analytic pipeline.

The analytic report (fast, about 2 s) keeps everything about parts, wiring, clearance and structure. The flown
evaluation (about 10 s) replaces what only a flight can tell: stability, stall, climb, and the performance score.
If the flight evaluation itself cannot run (no tail, no wing), the analytic numbers stay and the report says so.
"""
from __future__ import annotations

from typing import Optional

from . import flight as Fl
from . import mission as M

# analytic checks that the flown evaluation replaces
REPLACED = ("static_margin", "stall_margin", "climb")


def merge_flown(report: dict, design, bits: dict, preset: Optional[str] = None, cd0_factor: float = 1.0) -> dict:
    preset = preset or report.get("preset") or "endurance"
    try:
        veh = Fl.Vehicle.from_bits(bits, design, cd0_factor=cd0_factor)
        fl = M.evaluate_flight(veh, preset, payload_fraction=float(bits["pres"].payload_fraction or 0.0))
    except Exception as e:
        return _fail(report, f"{type(e).__name__}: {e}")
    report["flight"] = fl
    if not fl.get("ok") and not [c for c in fl.get("checks", []) if c["name"] != "trim"]:
        return _fail(report, str(fl.get("reason") or "the flown evaluation produced nothing"))
    gate = report["gate"]
    kept = [c for c in gate["checks"] if c["name"] not in REPLACED]
    flown_checks = [dict(c) for c in fl["checks"]]
    gate["checks"] = kept + flown_checks
    gate["pass"] = all(c["pass"] for c in gate["checks"])
    sc = report["score"]
    sc["analytic_value"] = sc["value"]
    sc["value"] = round(float(fl.get("score", 0.0)), 4) if gate["pass"] else 0.0
    sc["fidelity"] = "flown"
    sc["error_band"] = None
    m = fl.get("metrics", {})
    ac = report["aircraft"]
    for k_src, k_dst in (("endurance_min", "endurance_min"), ("range_km", "range_km"), ("wh_per_km", "wh_per_km"),
                         ("vmax_kmh", "vmax_kmh"), ("climb_ms", "climb_ms")):
        if k_src in m and m[k_src] == m[k_src]:
            ac["analytic_" + k_dst] = ac.get(k_dst)
            ac[k_dst] = round(float(m[k_src]), 3)
    ac["static_margin"] = round(float(fl["static_margin"]), 4)
    ac["stall_kmh"] = round(fl["sweep"]["v_stall"] * 3.6, 2)
    return report


def _fail(report: dict, reason: str) -> dict:
    """A design the flight model cannot fly or represent (no trim, no tail, an unsupported layout) is not silently scored
    by the closed-form numbers: the report keeps everything analytic, fails the gate with the reason, and scores 0."""
    report.setdefault("flight", {"ok": False, "reason": reason, "fidelity": "not run"})
    report["gate"]["checks"].append({"name": "flight_model", "value": "not flown", "limit": "flyable", "pass": False,
                                     "detail": reason})
    report["gate"]["pass"] = False
    report["score"]["analytic_value"] = report["score"]["value"]
    report["score"]["value"] = 0.0
    report["score"]["fidelity"] = "flown"
    report["warnings"].append(f"flown evaluation failed: {reason}")
    return report
