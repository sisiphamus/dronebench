"""Random-search baseline: a greedy hill climb over random edit ops.

Each step proposes one random op (remove a non-locked part, swap a motor/prop/battery/servo for a
catalog part of the same type, nudge a printed part's infill, scale the wing span ±5 %, move the
battery ±10 mm). The candidate is kept only if it passes the gate AND raises the score; every step is
logged to runs.jsonl (accepted or not).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

SWAP_TYPES = ("motor", "prop", "battery", "servo")


def _catalog(ptype: str) -> list[dict]:
    try:
        from .components import CATALOG
    except Exception:
        return []
    return [e for e in CATALOG if e.get("type") == ptype]


def propose(report: dict, rng: random.Random) -> Optional[list[dict]]:
    parts = report.get("parts", [])
    kinds = ["remove", "swap", "infill", "span", "battery"]
    kind = rng.choice(kinds)
    if kind == "remove":
        cands = [p for p in parts if not p.get("locked")]
        if not cands:
            return None
        return [{"op": "remove", "id": rng.choice(cands)["id"]}]
    if kind == "swap":
        cands = [p for p in parts if p["type"] in SWAP_TYPES]
        if not cands:
            return None
        p = rng.choice(cands)
        options = [e for e in _catalog(p["type"]) if e["name"] != p.get("part_number")]
        if not options:
            return None
        e = rng.choice(options)
        ids = [q["id"] for q in parts if q["type"] == p["type"] and q.get("part_number") == p.get("part_number")]
        return [{"op": "swap", "id": i, "entry": {"part_number": e["name"], "specs": dict(e.get("specs") or {})}}
                for i in ids]  # swap matched pairs (both motors) together
    if kind == "infill":
        cands = [p for p in parts if (p.get("material") or {}).get("kind") == "printed"
                 and (p.get("material") or {}).get("infill") is not None]
        if not cands:
            return None
        p = rng.choice(cands)
        v = min(1.0, max(0.0, round(p["material"]["infill"] + rng.choice((-0.05, 0.05)), 3)))
        return [{"op": "set", "id": p["id"], "path": "material.infill", "value": v}]
    if kind == "span":
        ids = [p["id"] for p in parts if p.get("role") in ("wing", "aileron", "flap")
               or (p["type"] == "spar" and p.get("role") in ("main_spar", "rear_spar"))]
        ids = [i for i in ids if next(q for q in parts if q["id"] == i).get("cad")]
        if not ids:
            return None
        f = rng.choice((0.95, 1.05))
        return [{"op": "scale", "ids": ids, "factors": [1.0, f, 1.0], "about": "origin"}]
    bats = [p for p in parts if p["type"] == "battery"]
    if not bats:
        return None
    return [{"op": "move", "ids": [b["id"] for b in bats], "offset_m": [rng.choice((-0.01, 0.01)), 0.0, 0.0]}]


def random_search(design_dir, steps: int = 30, preset: Optional[str] = None, seed: int = 0, out_dir=None,
                  runs_path=None, agent: str = "random") -> dict:
    from .env import DroneBench
    from .runs import append_run, record_from_report
    rng = random.Random(seed)
    design_dir = Path(design_dir)
    out_dir = Path(out_dir) if out_dir else design_dir.parent / f"{design_dir.name}_baseline"
    env = DroneBench(design_dir, preset=preset, workdir=out_dir, necessity=False, max_steps=steps + 1)
    report = env.reset()
    start = env.score_of(report)
    if runs_path:
        append_run(runs_path, {**record_from_report(report, step=0, agent=agent, design=str(design_dir)),
                               "accepted": True})
    history = []
    for i in range(1, steps + 1):
        ops = None
        for _ in range(10):
            ops = propose(env.report, rng)
            if ops:
                break
        if not ops:
            break
        prev = (env.current, env.report)
        obs, reward, _, info = env.step(ops)
        accepted = not info["rejected"] and reward > 0
        if not info["rejected"] and not accepted:     # passed the gate but did not improve: revert
            env.current, env.report = prev
        history.append({"step": i, "ops": ops, "score": info["score"], "accepted": accepted,
                        "reason": info.get("reason") or ("no improvement" if not accepted else "improved")})
        if runs_path:
            rep = info.get("candidate_report") or (obs if not info["rejected"] else None)
            if rep is not None:
                rec = record_from_report(rep, step=i, ops=ops, agent=agent, design=info.get("candidate_path"))
            else:
                rec = {"step": i, "score": info["score"], "gate_pass": not info["rejected"], "ops": ops,
                       "agent": agent, "failures": info.get("failures") or ["invalid_ops"],
                       "preset": env.preset, "design": info.get("candidate_path") or info.get("path")}
            rec.update(accepted=accepted, best_score=env.score_of(env.report))
            append_run(runs_path, rec)
    return {"start_score": start, "best_score": env.score_of(env.report), "best_design": str(env.current),
            "steps": len(history), "accepted": sum(h["accepted"] for h in history), "history": history}
