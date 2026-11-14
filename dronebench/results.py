"""Result files: a finished run in one JSON, and a replay that checks it.

    dronebench result WS --out result.json     export the run (agent, scores, every accepted and rejected step)
    dronebench verify result.json              re-fly it from the starting design and compare every score

A score in a result file is never trusted. `verify` rebuilds the starting design, replays each accepted step's
edits through the same gate and simulator, and compares the scores it gets with the ones claimed. Rejected steps
change nothing, so they are not replayed. The simulator is deterministic; a small tolerance covers differences
in floating point between machines.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from . import __version__
from . import workspace as W

FORMAT = "dronebench-result/1"
TOLERANCE = 0.005   # relative, between a claimed score and the replayed one


class ResultError(Exception):
    pass


def export_result(ws) -> dict:
    ws = Path(ws).resolve()
    m = W.meta(ws)
    recs = W._runs(ws)
    base = next((r for r in recs if r.get("step") == 0), None)
    steps = [r for r in recs if isinstance(r.get("step"), int) and r["step"] > 0]
    out_steps = [{"step": r["step"], "kind": r.get("kind"), "accepted": bool(r.get("accepted")),
                  "score": round(float(r.get("score") or 0.0), 4), "note": r.get("note") or "",
                  "reason": r.get("reason"), "ops": r.get("ops") or []} for r in steps]
    agent = next((r.get("agent") for r in steps if r.get("agent")), "agent")
    baseline = float(base["score"]) if base else float(m["baseline"])
    best = max([baseline] + [s["score"] for s in out_steps if s["accepted"]])
    return {"format": FORMAT, "dronebench": __version__, "fixture": m["fixture"], "builder": m.get("builder"),
            "preset": m["preset"], "fidelity": m.get("fidelity", "flown"), "start_params": m.get("start_params") or {},
            "agent": agent, "baseline": round(baseline, 4), "best": round(best, 4), "unit": m.get("unit"),
            "budget": m["max_steps"], "attempts": len(out_steps), "accepted": sum(1 for s in out_steps if s["accepted"]),
            "steps": out_steps}


def _close(claimed: float, got: float) -> bool:
    return abs(claimed - got) <= TOLERANCE * max(abs(got), 1e-9) + 1e-6


def verify_result(result: dict, design: Optional[str] = None) -> dict:
    """Replay a result file. `design` is a design folder to start from, for results made on your own design."""
    if not isinstance(result, dict) or result.get("format") != FORMAT:
        raise ResultError(f"not a DroneBench result file (format should be '{FORMAT}')")
    mism: list[dict] = []
    tmp = Path(tempfile.mkdtemp(prefix="dronebench-verify-"))
    try:
        ws = tmp / "w"
        try:
            m = W.init(ws, design or result["fixture"], result.get("preset"), max(int(result.get("budget") or 0), len(result["steps"]), 1),
                       False, False, result.get("fidelity", "flown"), result.get("start_params") or None)
        except W.WorkspaceError as e:
            raise ResultError(str(e))
        if not _close(result["baseline"], m["baseline"]):
            mism.append({"what": "baseline", "claimed": result["baseline"], "replayed": m["baseline"]})
        replayed, best = 0, m["baseline"]
        for s in result["steps"]:
            if not s["accepted"]:
                continue
            if s.get("kind") == "build":
                sets = ((s.get("ops") or [{}])[0]).get("set") or {}
                o = W.build_step(ws, sets, s.get("note", ""), "replay", True, False)
            else:
                o = W.try_step(ws, s.get("ops") or [], s.get("note", ""), "replay", True, False)
            replayed += 1
            got = o.get("score") or 0.0
            if not o["accepted"]:
                mism.append({"what": f"step {s['step']} accepted", "claimed": True, "replayed": False,
                             "why": "; ".join(filter(None, [o.get("reason")] + (o.get("failures") or []) + (o.get("errors") or [])))})
                continue
            if not _close(s["score"], got):
                mism.append({"what": f"step {s['step']} score", "claimed": s["score"], "replayed": round(got, 4)})
            best = max(best, got)
        if not _close(result["best"], best):
            mism.append({"what": "best", "claimed": result["best"], "replayed": round(best, 4)})
        return {"ok": not mism, "replayed": replayed, "mismatches": mism, "best": round(best, 4), "baseline": round(m["baseline"], 4)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
