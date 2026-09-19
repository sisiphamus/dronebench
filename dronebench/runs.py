"""Run log: one JSON record per scored step, appended to a JSONL file.

A record is small on purpose (the full report lives in report.json). The viewer
reads these to draw the score-vs-step curve, one line per `agent` label.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

KEY_METRICS = ("stall_kmh", "cruise_kmh", "vmax_kmh", "climb_ms", "endurance_min", "range_km",
               "wh_per_km", "static_margin", "spar_sf")


def record_from_report(report: dict, step: Optional[int] = None, ops: Optional[list] = None,
                       agent: str = "agent", design: Optional[str] = None) -> dict:
    """Build a run record from a report dict (schema PLAN §4)."""
    gate = report.get("gate") or {}
    score = report.get("score") or {}
    aircraft = report.get("aircraft") or {}
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "step": step,
        "design": design or report.get("design"),
        "preset": report.get("preset") or score.get("preset"),
        "score": score.get("value"),
        "unit": score.get("unit"),
        "gate_pass": bool(gate.get("pass")),
        "failures": [c.get("name") for c in gate.get("checks", []) if not c.get("pass")],
        "mass_g": aircraft.get("mass_g"),
        "metrics": {k: aircraft.get(k) for k in KEY_METRICS if k in aircraft},
        "ops": ops or [],
        "agent": agent,
    }


def load_runs(path) -> list[dict]:
    """Read a runs JSONL file. Missing file -> []. Bad lines are skipped."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def next_step(path, agent: Optional[str] = None) -> int:
    """Next step number for `agent` (all agents if None): max existing step + 1, or 0."""
    steps = [r.get("step") for r in load_runs(path)
             if isinstance(r.get("step"), int) and (agent is None or r.get("agent") == agent)]
    return max(steps) + 1 if steps else 0


def append_run(runs_path, record: dict[str, Any]) -> dict:
    """Append one record. Fills timestamp and step (auto-increment per agent) if missing."""
    p = Path(runs_path)
    rec = dict(record)
    rec.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    rec.setdefault("agent", "agent")
    if rec.get("step") is None:
        rec["step"] = next_step(p, rec["agent"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec, default=_json_default) + "\n")
    return rec


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)
