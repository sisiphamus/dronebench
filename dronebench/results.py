"""Result files: a finished run in one JSON, and a replay that checks it.

    dronebench result WS --out result.json     export the run (agent, scores, every accepted and rejected step)
    dronebench verify result.json              re-fly it from the starting design and compare every score
    dronebench verify result.json --standard   ... and check it is the standard task (what a leaderboard submission must be)

A score in a result file is never trusted. `verify` rebuilds the starting design, replays each accepted step's
edits through the same gate and simulator, and compares the scores it gets with the ones claimed. Rejected steps
change nothing, so they are not replayed. The simulator is deterministic; a tolerance of 0.1% covers differences
in floating point between machines.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from . import __version__
from . import workspace as W

FORMAT = "dronebench-result/1"
TOLERANCE = 1e-3   # relative, between a claimed score and the replayed one
# What a leaderboard submission must be: the plain Falcon, its default mission, flown, within 30 steps, no files from outside.
STANDARD = {"fixture": "falcon_v2", "preset": "endurance", "fidelity": "flown", "max_budget": 30}


class ResultError(Exception):
    pass


def design_digest(design_dir) -> str:
    """Fingerprint of a design's parts list and mission (what defines the starting aircraft). Formatting and the preset,
    which a run may choose, do not count."""
    docs = []
    for name in ("bom.yaml", "mission.yaml"):
        d = yaml.safe_load((Path(design_dir) / name).read_text()) or {}
        if name == "mission.yaml":
            d.pop("preset", None)
        docs.append(d)
    return hashlib.sha256(json.dumps(docs, sort_keys=True, default=str).encode()).hexdigest()


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
    start = ws / "steps" / "0" / "design"
    return {"format": FORMAT, "dronebench": __version__, "fixture": m["fixture"], "builder": m.get("builder"),
            "preset": m["preset"], "fidelity": m.get("fidelity", "flown"), "start_params": m.get("start_params") or {},
            "start_design_sha256": design_digest(start) if (start / "bom.yaml").exists() else None,
            "agent": agent, "baseline": round(baseline, 4), "best": round(best, 4), "unit": m.get("unit"),
            "budget": m["max_steps"], "attempts": len(out_steps), "accepted": sum(1 for s in out_steps if s["accepted"]),
            "steps": out_steps}


def uses_local_files(result: dict) -> list[int]:
    """Accepted steps whose edits read a mesh from the author's disk (`add` with `cad_file`). They cannot be replayed elsewhere."""
    return [s["step"] for s in result.get("steps", []) if s.get("accepted")
            and any(isinstance(o, dict) and o.get("cad_file") for o in s.get("ops") or [])]


def _schema(result) -> None:
    if not isinstance(result, dict) or result.get("format") != FORMAT:
        raise ResultError(f"not a DroneBench result file (format should be '{FORMAT}')")
    try:
        for k in ("fixture", "preset", "fidelity"):
            if not isinstance(result[k], str):
                raise TypeError(k)
        for k in ("baseline", "best", "budget"):
            float(result[k])
        if not isinstance(result["steps"], list):
            raise TypeError("steps")
        for s in result["steps"]:
            int(s["step"])
            bool(s["accepted"])
            float(s["score"])
            if not isinstance(s.get("ops") or [], list):
                raise TypeError("ops")
    except (KeyError, TypeError, ValueError) as e:
        raise ResultError(f"malformed result file (problem with {e!s})")
    if int(result["budget"]) < len(result["steps"]):
        raise ResultError(f"budget {int(result['budget'])} is smaller than the {len(result['steps'])} attempts in the file")


def standard_problems(result: dict) -> list[str]:
    """Why this result is not the standard task (empty list: it is)."""
    probs = []
    if result.get("fixture") != STANDARD["fixture"]:
        probs.append(f"fixture is '{result.get('fixture')}', the standard task is '{STANDARD['fixture']}'")
    if result.get("start_params"):
        probs.append("it starts from your own numbers (start_params); the standard task starts from the fixture as shipped")
    if result.get("preset") != STANDARD["preset"]:
        probs.append(f"preset is '{result.get('preset')}', the standard task is '{STANDARD['preset']}'")
    if result.get("fidelity") != STANDARD["fidelity"]:
        probs.append(f"fidelity is '{result.get('fidelity')}', the standard task is '{STANDARD['fidelity']}'")
    if int(result.get("budget") or 0) > STANDARD["max_budget"]:
        probs.append(f"budget {result.get('budget')} is more than the standard {STANDARD['max_budget']} steps")
    if uses_local_files(result):
        probs.append("an accepted step reads a mesh file from disk (cad_file), which cannot be replayed")
    return probs


def _close(claimed: float, got: float) -> bool:
    return abs(claimed - got) <= TOLERANCE * max(abs(got), 1e-9) + 1e-6


def verify_result(result: dict, design: Optional[str] = None, standard: bool = False) -> dict:
    """Replay a result file. `design` is a design folder to start from, for results made on your own design.

    Without `design`, the starting design must be a built-in fixture (a result file cannot point at a folder of its own).
    """
    _schema(result)
    mism: list[dict] = []
    if standard:
        mism += [{"what": "standard task", "claimed": "-", "replayed": "-", "why": p} for p in standard_problems(result)]
        if mism:
            return {"ok": False, "replayed": 0, "mismatches": mism, "best": None, "baseline": None}
    name = str(result["fixture"])
    if design is None and (Path(name).name != name or not (W.FIXTURES / name / "bom.yaml").exists()):
        raise ResultError(f"'{name}' is not a built-in fixture; pass --design FOLDER to replay a result made on your own design")
    tmp = Path(tempfile.mkdtemp(prefix="dronebench-verify-"))
    try:
        ws = tmp / "w"
        try:
            m = W.init(ws, design or name, result.get("preset"), max(int(result["budget"]), 1), False, False,
                       result.get("fidelity", "flown"), result.get("start_params") or None)
        except W.WorkspaceError as e:
            raise ResultError(str(e))
        want = result.get("start_design_sha256")
        if design is not None and want and design_digest(ws / "steps" / "0" / "design") != want:
            raise ResultError("the design folder is not the one this result started from (parts list or mission differ)")
        if not _close(float(result["baseline"]), m["baseline"]):
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
            if not _close(float(s["score"]), got):
                mism.append({"what": f"step {s['step']} score", "claimed": s["score"], "replayed": round(got, 4)})
            best = max(best, got)
        if not _close(float(result["best"]), best):
            mism.append({"what": "best", "claimed": result["best"], "replayed": round(best, 4)})
        return {"ok": not mism, "replayed": replayed, "mismatches": mism, "best": round(best, 4), "baseline": round(m["baseline"], 4)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
