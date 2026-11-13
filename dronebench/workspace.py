"""Workspaces: the agent-facing loop. One folder per attempt.

    ws/
      workspace.json     fixture, preset, created, best step, current step
      TASK.md            the brief an agent reads
      design/            the CURRENT accepted design (never edit by hand; use `try` / `build`)
      params.yaml        only for parametric designs (falcon_v2): knobs for `build`
      runs.jsonl         one record per attempted step, accepted or not
      steps/N/           every attempt: design/, report.json, ops.json (or params.json)

An attempt is judged like this (one rule for every agent):

  * invalid ops / failed build         -> rejected, nothing changes
  * gate fails                         -> rejected, the check name and number come back
  * gate passes, score >= best score   -> accepted, `design/` becomes this attempt
  * gate passes, score < best score    -> rejected as a regression (unless keep=True)

`try_step` and `build_step` return a plain dict (the outcome), the same one the CLI prints and the
dashboard shows. All state lives on disk, so a second process (the dashboard) can read it live.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures"

EPS = 1e-9


class WorkspaceError(Exception):
    pass


# ---------------------------------------------------------------- small helpers
def _read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)  # atomic: the dashboard never reads a half-written file


def _link_tree(src: Path, dst: Path) -> None:
    def link_or_copy(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    shutil.copytree(src, dst, copy_function=link_or_copy)


def score_of(report: Optional[dict]) -> float:
    v = ((report or {}).get("score") or {}).get("value")
    return float(v) if isinstance(v, (int, float)) else 0.0


def meta(ws) -> dict:
    m = _read_json(Path(ws) / "workspace.json")
    if m is None:
        raise WorkspaceError(f"not a DroneBench workspace: {ws} (run `dronebench init {ws}` first)")
    return m


def _evaluate(design: Path, preset: Optional[str], necessity: bool) -> dict:
    from .pipeline import evaluate
    return evaluate(design, preset=preset, necessity=necessity)


def fixture_path(name: str) -> Path:
    p = Path(name)
    if p.is_dir() and (p / "bom.yaml").exists():
        return p
    q = FIXTURES / name
    if q.is_dir() and (q / "bom.yaml").exists():
        return q
    names = sorted(d.name for d in FIXTURES.iterdir() if (d / "bom.yaml").exists()) if FIXTURES.is_dir() else []
    raise WorkspaceError(f"unknown fixture '{name}'. Built-in: {', '.join(names) or 'none found'}, or a design folder path")


# ---------------------------------------------------------------- the brief
TASK_TEMPLATE = """# DroneBench task: {fixture}

You are an engineer improving a small electric fixed-wing UAV, working inside DroneBench.
The aircraft is `design/`. DroneBench analyses it (aero, propulsion, structure, stability, wiring),
checks that it flies, and gives it a score. You never edit files by hand: every change goes through
a command, and DroneBench accepts or rejects it with reasons.

## Goal

Raise the **{preset}** score ({unit_hint}) as far as you can, **without failing the gate**. A design
that fails any gate check is rejected. Every part must be justified: parts DroneBench marks
`unnecessary` cost mass for nothing. Baseline score: **{baseline}**.

The mission payload ({payload}) is locked and must stay. Do not touch `mission.yaml`, `locked`
flags or `sources` tags. Specs you type for new or swapped parts are tagged `agent_supplied` and
checked for plausibility (energy per kg, power per gram, ...); implausible values are clamped.

## Commands (run from the workspace folder: `{ws}`)

```bash
dronebench status .                    # score, gate margins, what limits you, recent steps
dronebench parts design --table        # every part: function, necessity, margin
dronebench try . --ops ops.json --note "why"      # edit -> score -> accept or reject
dronebench build . --set wing.span_mm=2200 --note "why"   # change a design parameter, regenerate the aircraft{build_note}
dronebench serve .                     # live dashboard (3D model, score curve, step log)
```

`try` applies a JSON list of ops to the current design, scores the result, and keeps it only if the
gate passes and the score is at least as good as your best. A rejected try tells you which gate check
failed and by how much. Add `--keep` to accept a gate-passing design even if the score dipped (use it
when a step is a setup for the next one).

Ops (`ops.json` is a JSON list; frame: x aft, y right, z up, metres):

```json
[{{"op": "remove", "id": "ballast_nose"}},
 {{"op": "swap", "id": "prop_L", "entry": {{"part_number": "APC 12x6E"}}}},
 {{"op": "set", "id": "wing_L1", "path": "material.infill", "value": 0.0}},
 {{"op": "scale", "ids": ["wing_L1", "wing_R1"], "factors": [1.0, 1.05, 1.0], "about": "origin"}},
 {{"op": "move", "ids": ["battery"], "offset_m": [-0.01, 0, 0]}},
 {{"op": "add", "entry": {{"id": "gps2", "type": "gps", "specs": {{"mass_g": 20}}}}}}]
```

## Loop

1. `dronebench status .` and `dronebench parts design --table`. Parts marked `unnecessary` are removal
   candidates; the smallest gate margins say what stops you from going further.
2. Choose ONE change. Say in one line what you expect to move and which margin could suffer.
3. `try` or `build` it. Read the outcome. If it was rejected, learn from the reason; do not repeat it.
4. Repeat. Stop after **{steps} steps**, or earlier if five attempts in a row do not improve the score.
5. Finish with: starting score, final score, the kept changes and why each helped, and the final gate
   margins (`dronebench status .`).

Be honest about the model: results are analytic estimates (about +-15%), good for ranking designs,
not a certificate of airworthiness.
"""

UNIT_HINT = {"endurance": "minutes of flight", "range": "km", "speed": "km/h top speed",
             "payload": "payload mass fraction", "efficiency": "km per Wh"}


def _build_note(has_params: bool) -> str:
    if not has_params:
        return " (this design has no params.yaml; use `try`)"
    return ("; `params.yaml` lists the knobs. Note `build` regenerates all CAD and the BOM from "
            "`params.yaml`, so it replaces earlier `try` edits to the parts list")


# ---------------------------------------------------------------- init / state
def init(ws, fixture: str = "falcon_v2", preset: Optional[str] = None, steps: int = 12, force: bool = False,
         necessity: bool = True) -> dict:
    ws = Path(ws).resolve()
    src = fixture_path(fixture)
    if ws.exists() and any(ws.iterdir()):
        if not force:
            raise WorkspaceError(f"{ws} is not empty (use --force to replace it)")
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, ws / "design")
    has_params = (ws / "design" / "params.yaml").exists()
    if has_params:
        shutil.copy2(ws / "design" / "params.yaml", ws / "params.yaml")
    if preset:  # write the preset into the workspace copy of the mission
        mp = ws / "design" / "mission.yaml"
        m = yaml.safe_load(mp.read_text()) or {}
        m["preset"] = preset
        mp.write_text(yaml.safe_dump(m, sort_keys=False))
    t = time.time()
    s0 = ws / "steps" / "0"
    s0.mkdir(parents=True)
    _link_tree(ws / "design", s0 / "design")
    report = _evaluate(ws / "design", None, necessity)
    _write_json(s0 / "report.json", report)
    preset = report.get("preset") or preset
    payload = ", ".join((yaml.safe_load((ws / "design" / "mission.yaml").read_text()) or {}).get("payload_ids") or []) or "none"
    (ws / "TASK.md").write_text(TASK_TEMPLATE.format(
        fixture=src.name, preset=preset, unit_hint=UNIT_HINT.get(preset, ""), baseline=_fmt_score(report),
        payload=payload, ws=ws, steps=steps, build_note=_build_note(has_params)))
    m = {"fixture": src.name, "preset": preset, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "max_steps": steps, "has_params": has_params, "steps": 0, "best_step": 0, "current_step": 0,
         "baseline": score_of(report), "best": score_of(report), "unit": (report.get("score") or {}).get("unit")}
    _write_json(ws / "workspace.json", m)
    from .runs import append_run, record_from_report
    rec = record_from_report(report, step=0, ops=[], agent="agent", design=str(s0 / "design"))
    rec.update(accepted=True, note="baseline", seconds=round(time.time() - t, 1), best_score=score_of(report))
    append_run(ws / "runs.jsonl", rec)
    return m


def _fmt_score(report: dict) -> str:
    s = report.get("score") or {}
    v = s.get("value")
    return f"{v:.1f} {s.get('unit') or ''}".strip() if isinstance(v, (int, float)) else "n/a"


def _report(ws: Path, n: int) -> Optional[dict]:
    return _read_json(ws / "steps" / str(n) / "report.json")


def current_report(ws) -> dict:
    ws = Path(ws)
    m = meta(ws)
    r = _report(ws, m["current_step"])
    if r is None:
        raise WorkspaceError("workspace is missing its current report")
    return r


def limiting(report: dict, n: int = 4) -> list[dict]:
    """The gate checks with the least headroom (numeric two-sided or one-sided limits)."""
    out = []
    for c in (report.get("gate") or {}).get("checks", []):
        v, lim = c.get("value"), c.get("limit")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        head = None
        if isinstance(lim, (list, tuple)) and len(lim) == 2:
            lo, hi = lim
            parts = []
            if lo is not None and lo != 0:
                parts.append((v - lo) / abs(lo))
            if hi is not None and hi != 0:
                parts.append((hi - v) / abs(hi))
            head = min(parts) if parts else None
        elif isinstance(lim, (int, float)) and lim:
            head = (v - lim) / abs(lim)
        if head is not None:
            out.append({"name": c["name"], "value": v, "limit": lim, "headroom": round(head, 3), "pass": c.get("pass")})
    out.sort(key=lambda x: x["headroom"])
    return out[:n]


def _unnecessary(report: dict) -> list[dict]:
    rows = [p for p in report.get("parts", []) if p.get("necessity") == "unnecessary" and not p.get("locked")]
    rows.sort(key=lambda p: -(p.get("score_delta_if_removed") or 0))
    return [{"id": p["id"], "gain": p.get("score_delta_if_removed"), "mass_g": p.get("mass_g")} for p in rows]


def status(ws) -> dict:
    ws = Path(ws)
    m = meta(ws)
    rep = current_report(ws)
    runs = [r for r in _runs(ws)]
    return {"workspace": str(ws), "meta": m, "score": rep.get("score"), "gate": (rep.get("gate") or {}).get("pass"),
            "aircraft": rep.get("aircraft"), "limiting": limiting(rep), "unnecessary": _unnecessary(rep),
            "steps_used": m["steps"], "steps_left": max(0, m["max_steps"] - m["steps"]),
            "recent": [{k: r.get(k) for k in ("step", "accepted", "score", "note", "reason", "failures")}
                       for r in runs[-5:]]}


def status_text(ws) -> str:
    s = status(ws)
    m = s["meta"]
    sc = s["score"] or {}
    lines = [f"{m['fixture']}  preset={m['preset']}  step {m['steps']}/{m['max_steps']}",
             f"score {(sc.get('value') or 0):.1f} {sc.get('unit') or ''}   (baseline {m['baseline']:.1f}, "
             f"{(sc.get('value') or 0) - m['baseline']:+.1f})   gate {'PASS' if s['gate'] else 'FAIL'}"]
    a = s.get("aircraft") or {}
    r = lambda k, n=1: "-" if a.get(k) is None else f"{a[k]:.{n}f}"
    lines.append(f"mass {r('mass_g', 0)} g  stall {r('stall_kmh')} km/h  cruise {r('cruise_kmh', 0)} km/h  "
                 f"SM {r('static_margin', 3)}  spar SF {r('spar_sf', 2)}  Wh/km {r('wh_per_km', 2)}")
    lines.append("tightest checks: " + ", ".join(f"{c['name']} {c['headroom'] * 100:+.0f}%" for c in s["limiting"]))
    un = s["unnecessary"]
    if un:
        lines.append("unnecessary parts: " + ", ".join(f"{u['id']} (+{u['gain']:.1f})" if u.get("gain") else u["id"] for u in un[:8]))
    return "\n".join(lines)


def _runs(ws) -> list[dict]:
    from .runs import load_runs
    return load_runs(Path(ws) / "runs.jsonl")


# ---------------------------------------------------------------- attempts
def _outcome_base(step: int, kind: str, note: str, agent: str) -> dict:
    return {"step": step, "kind": kind, "note": note, "agent": agent, "accepted": False, "status": "rejected",
            "reason": None, "errors": [], "failures": [], "score": None, "previous_best": None, "delta": 0.0}


def _log(ws: Path, step: int, out: dict, report: Optional[dict], ops, agent: str, design: Optional[Path], seconds: float):
    from .runs import append_run, record_from_report
    if report is not None:
        rec = record_from_report(report, step=step, ops=ops, agent=agent, design=str(design))
    else:
        rec = {"step": step, "preset": meta(ws)["preset"], "score": 0.0, "gate_pass": False, "ops": ops,
               "agent": agent, "design": str(design) if design else None, "failures": ["invalid"]}
    rec.update(accepted=out["accepted"], note=out["note"], reason=out["reason"], errors=out["errors"],
               fail_detail=out["failures"], kind=out["kind"], delta=out["delta"], seconds=round(seconds, 2),
               best_score=max(out["score"] or 0.0, out["previous_best"] or 0.0) if out["accepted"] else out["previous_best"])
    append_run(ws / "runs.jsonl", rec)


def _judge(ws: Path, m: dict, step: int, cand: Path, ops, kind: str, note: str, agent: str, keep: bool,
           errors: list[str], t0: float, necessity: bool) -> dict:
    out = _outcome_base(step, kind, note, agent)
    out["previous_best"] = m["best"]
    report = None
    if errors:
        out.update(reason="invalid " + ("ops" if kind == "try" else "build"), errors=errors)
        _log(ws, step, out, None, ops, agent, cand, time.time() - t0)
        return out
    try:
        report = _evaluate(cand, m["preset"], False)   # fast pass: gate + score (about 2 s)
    except Exception as e:  # the design could not even be analysed
        out.update(reason="evaluation failed", errors=[f"{type(e).__name__}: {e}"])
        _log(ws, step, out, None, ops, agent, cand, time.time() - t0)
        return out
    new = score_of(report)
    out["score"] = new
    fails = [c for c in report["gate"]["checks"] if not c.get("pass")]
    if fails:
        out.update(reason="gate failed", failures=[
            f"{c['name']}: value {c.get('value')} vs limit {c.get('limit')}" + (f" ({c['detail']})" if c.get("detail") else "")
            for c in fails])
        out["score"] = 0.0
    else:
        out["delta"] = round(new - m["best"], 4)
        if new + EPS >= m["best"] or keep:
            out.update(accepted=True, status="improved" if new > m["best"] + EPS else "neutral" if abs(new - m["best"]) <= EPS else "kept")
        else:
            out.update(reason=f"regression: {new:.2f} < best {m['best']:.2f} (use --keep to accept anyway)")
    if out["accepted"] and necessity:   # only designs we keep get the full removal test (per-part necessity)
        report = _evaluate(cand, m["preset"], True)
    _write_json(ws / "steps" / str(step) / "report.json", report)
    _log(ws, step, out, report, ops, agent, cand, time.time() - t0)
    return out


def _begin(ws: Path):
    m = meta(ws)
    if m["steps"] >= m["max_steps"] + 0:
        raise WorkspaceError(f"step budget used up ({m['steps']}/{m['max_steps']}). Raise it in workspace.json to continue")
    return m


def _accept(ws: Path, m: dict, step: int, out: dict) -> None:
    cur = ws / "design"
    shutil.rmtree(cur)
    _link_tree(ws / "steps" / str(step) / "design", cur)
    m["current_step"] = step
    new = out["score"]
    if new > m["best"]:
        m["best"], m["best_step"] = new, step
    out["best"] = m["best"]


def try_step(ws, ops: list, note: str = "", agent: str = "agent", keep: bool = False, necessity: bool = True) -> dict:
    """Apply ops to the current design, score the result, accept or reject it. See module docstring."""
    from .edits import apply_edits
    ws = Path(ws).resolve()
    m = _begin(ws)
    t0 = time.time()
    step = m["steps"] + 1
    sdir = ws / "steps" / str(step)
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    _write_json(sdir / "ops.json", ops)
    new_dir, errors = apply_edits(ws / "design", ops, sdir / "design")
    out = _judge(ws, m, step, new_dir, ops, "try", note, agent, keep, errors, t0, necessity)
    m["steps"] = step
    if out["accepted"]:
        _accept(ws, m, step, out)
    _write_json(ws / "workspace.json", m)
    out["best"] = m["best"]
    return out


# ---- build (parametric designs)
def _load_builder(fixture: str):
    path = FIXTURES / f"build_{fixture}.py"
    if not path.exists():
        raise WorkspaceError(f"no parametric builder for '{fixture}' (expected {path.name}); use `try` instead")
    spec = importlib.util.spec_from_file_location(f"_db_build_{fixture}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def parse_sets(items: list[str]) -> dict:
    """['wing.span_mm=2200', 'propulsion.prop=APC 12x6E'] -> {'wing.span_mm': 2200, ...} (YAML scalars)."""
    out = {}
    for it in items:
        if "=" not in it:
            raise WorkspaceError(f"bad --set '{it}' (use path=value, e.g. wing.span_mm=2200)")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def _set_path(d: dict, path: str, value) -> None:
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            raise WorkspaceError(f"unknown parameter '{path}' (no '{k}' here). Keys: {', '.join(map(str, cur)) if isinstance(cur, dict) else '-'}")
        cur = cur[k]
    if not isinstance(cur, dict) or keys[-1] not in cur:
        raise WorkspaceError(f"unknown parameter '{path}'. Keys here: {', '.join(map(str, cur)) if isinstance(cur, dict) else '-'}")
    cur[keys[-1]] = value


def build_step(ws, sets: dict, note: str = "", agent: str = "agent", keep: bool = False, necessity: bool = True) -> dict:
    """Change design parameters, regenerate the whole aircraft from params.yaml, score it."""
    ws = Path(ws).resolve()
    m = _begin(ws)
    if not m.get("has_params"):
        raise WorkspaceError("this design is not parametric (no params.yaml). Use `try` with ops")
    t0 = time.time()
    step = m["steps"] + 1
    sdir = ws / "steps" / str(step)
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    params = yaml.safe_load((ws / "params.yaml").read_text())
    errors: list[str] = []
    try:
        for k, v in sets.items():
            _set_path(params, k, v)
        mod = _load_builder(m["fixture"])
        mod.build(params, sdir / "design", verbose=False)
        mission = ws / "design" / "mission.yaml"       # keep the locked mission (preset, payload, cruise)
        shutil.copy2(mission, sdir / "design" / "mission.yaml")
        (sdir / "design" / "params.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    except WorkspaceError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"build failed: {type(e).__name__}: {e}")
    ops = [{"op": "params", "set": sets}]
    _write_json(sdir / "ops.json", ops)
    out = _judge(ws, m, step, sdir / "design", ops, "build", note, agent, keep, errors, t0, necessity)
    m["steps"] = step
    if out["accepted"]:
        _accept(ws, m, step, out)
        (ws / "params.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    _write_json(ws / "workspace.json", m)
    out["best"] = m["best"]
    return out


def outcome_text(o: dict) -> str:
    head = f"step {o['step']}: {o['status'].upper()}"
    if o["accepted"]:
        head += f"  score {o['score']:.2f} ({o['delta']:+.2f} vs previous best {o['previous_best']:.2f})"
    lines = [head]
    if o.get("reason"):
        lines.append("reason: " + o["reason"])
    for e in o.get("errors", []):
        lines.append("  error: " + e)
    for f in o.get("failures", []):
        lines.append("  gate: " + f)
    return "\n".join(lines)
