"""`dronebench` command line: score | parts | view | edit | baseline | init | try | build | status | serve | run.

Exit codes: 0 ok, 2 usage error, 1 internal error (traceback only with --debug).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from pathlib import Path

PRESETS = ("endurance", "range", "speed", "payload", "efficiency")


class UsageError(Exception):
    """Bad input from the user (exit code 2)."""


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def _dumps(obj, pretty=False) -> str:
    return json.dumps(obj, indent=2 if pretty else None, default=_json_default,
                      separators=None if pretty else (",", ":"))


def _design_dir(path: str) -> Path:
    p = Path(path)
    if not p.is_dir():
        raise UsageError(f"design folder not found: {path}")
    return p


def _evaluate(design_dir: Path, preset=None, necessity=True) -> dict:
    from dronebench.pipeline import evaluate
    return evaluate(design_dir, preset=preset, necessity=necessity)


# ---------- formatting ----------

def _fmt(v, nd=2) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x, nd) for x in v) + "]"
    return str(v)


def _limit_str(limit) -> str:
    if isinstance(limit, (list, tuple)) and len(limit) == 2:
        lo, hi = limit
        if lo is None:
            return f"<= {_fmt(hi)}"
        if hi is None:
            return f">= {_fmt(lo)}"
        return f"{_fmt(lo)}..{_fmt(hi)}"
    return _fmt(limit)


def summary_text(report: dict) -> str:
    score = report.get("score") or {}
    gate = report.get("gate") or {}
    ac = report.get("aircraft") or {}
    lines = [f"{report.get('design', '?')}  preset={report.get('preset') or score.get('preset')}",
             f"  score      {_fmt(score.get('value'))} {score.get('unit') or ''}".rstrip()]
    fails = [c for c in gate.get("checks", []) if not c.get("pass")]
    lines.append(f"  gate       {'PASS' if gate.get('pass') else 'FAIL'}")
    for c in fails:
        detail = f"  ({c['detail']})" if c.get("detail") else ""
        lines.append(f"    x {c.get('name')}: {_fmt(c.get('value'))} vs {_limit_str(c.get('limit'))}{detail}")
    rows = [("mass", ac.get("mass_g"), "g"), ("stall", ac.get("stall_kmh"), "km/h"),
            ("cruise", ac.get("cruise_kmh"), "km/h"), ("Wh/km", ac.get("wh_per_km"), ""),
            ("endurance", ac.get("endurance_min"), "min"), ("SM", ac.get("static_margin"), ""),
            ("spar SF", ac.get("spar_sf"), "")]
    for name, v, unit in rows:
        lines.append(f"  {name:<10} {_fmt(v)} {unit}".rstrip())
    for w in report.get("warnings", [])[:5]:
        lines.append(f"  ! {w}")
    return "\n".join(lines)


def _margin_of(part: dict):
    op = part.get("operating_point") or {}
    m = op.get("margin")
    return m if isinstance(m, (int, float)) else None


def parts_table(parts: list[dict]) -> str:
    cols = ("id", "type", "function", "necessity", "mass_g", "material", "margin")
    rows = []
    for p in parts:
        mat = p.get("material") or {}
        rows.append((str(p.get("id", "")), str(p.get("type", "")), str(p.get("function", "")),
                     str(p.get("necessity", "")), _fmt(p.get("mass_g"), 1),
                     str(mat.get("name", "") if isinstance(mat, dict) else mat), _fmt(_margin_of(p))))
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    out = [fmt.format(*cols), fmt.format(*("-" * w for w in widths))]
    out += [fmt.format(*r) for r in rows]
    return "\n".join(out)


# ---------- commands ----------

def cmd_score(args) -> int:
    d = _design_dir(args.design)
    report = _evaluate(d, preset=args.preset, necessity=not args.no_necessity)
    if args.out:
        Path(args.out).write_text(_dumps(report, pretty=True))
    if args.runs:
        from dronebench.runs import append_run, record_from_report
        rec = record_from_report(report, step=args.step, agent=args.label, design=str(d))
        append_run(args.runs, rec)
    print(summary_text(report) if args.summary else _dumps(report, pretty=args.pretty))
    return 0


def cmd_parts(args) -> int:
    d = _design_dir(args.design)
    report = _evaluate(d, preset=args.preset, necessity=not args.no_necessity)
    parts = report.get("parts", [])
    print(parts_table(parts) if args.table else _dumps(parts, pretty=args.pretty))
    return 0


def cmd_view(args) -> int:
    dirs = [_design_dir(p) for p in args.designs]
    if len(dirs) > 2:
        raise UsageError("view takes one or two designs")
    if args.runs and not Path(args.runs).exists():
        raise UsageError(f"runs file not found: {args.runs}")
    reports = [_evaluate(d, preset=args.preset, necessity=True) for d in dirs]
    from dronebench.viewer import render_viewer
    out = render_viewer(dirs, reports, args.out, runs_jsonl=args.runs)
    print(str(out))
    if args.open:
        webbrowser.open(Path(out).resolve().as_uri())
    return 0


def cmd_edit(args) -> int:
    d = _design_dir(args.design)
    ops_path = Path(args.ops)
    if not ops_path.exists():
        raise UsageError(f"ops file not found: {args.ops}")
    try:
        ops = json.loads(ops_path.read_text())
    except json.JSONDecodeError as e:
        raise UsageError(f"ops file is not valid JSON: {e}")
    if isinstance(ops, dict):
        ops = ops.get("ops", [ops])
    if not isinstance(ops, list):
        raise UsageError("ops must be a JSON list of edit ops")
    from dronebench.edits import apply_edits
    new_dir, errors = apply_edits(d, ops, Path(args.out))
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 1
    print(str(new_dir))
    return 0


def cmd_baseline(args) -> int:
    d = _design_dir(args.design)
    if args.steps < 1:
        raise UsageError("--steps must be >= 1")
    from dronebench.baseline import random_search
    out_dir = Path(args.out) if args.out else d.parent / f"{d.name}_baseline"
    result = random_search(d, args.steps, args.preset, args.seed, out_dir, args.runs)
    if result is not None:
        print(_dumps(result, pretty=True) if not isinstance(result, str) else result)
    return 0


# ---------- workspace commands (the agent loop) ----------

def _agent_label(args) -> str:
    import os
    return getattr(args, "agent", None) or os.environ.get("DRONEBENCH_AGENT") or "agent"


def _ws_call(fn, *a, **kw):
    from dronebench.workspace import WorkspaceError
    try:
        return fn(*a, **kw)
    except WorkspaceError as e:
        raise UsageError(str(e))


def cmd_init(args) -> int:
    from dronebench import workspace as W
    m = _ws_call(W.init, args.workspace, args.fixture, args.preset, args.steps, args.force)
    print(f"workspace ready: {Path(args.workspace).resolve()}")
    print(f"  fixture {m['fixture']}  preset {m['preset']}  baseline {m['baseline']:.2f} {m.get('unit') or ''}  budget {m['max_steps']} steps")
    print("  next: cd into it, read TASK.md, then `dronebench serve .` in another terminal")
    return 0


def cmd_try(args) -> int:
    from dronebench import workspace as W
    src = sys.stdin.read() if args.ops == "-" else (Path(args.ops).read_text() if Path(args.ops).exists() else None)
    if src is None:
        raise UsageError(f"ops file not found: {args.ops}")
    try:
        ops = json.loads(src)
    except json.JSONDecodeError as e:
        raise UsageError(f"ops are not valid JSON: {e}")
    if isinstance(ops, dict):
        ops = ops.get("ops", [ops])
    if not isinstance(ops, list):
        raise UsageError("ops must be a JSON list")
    o = _ws_call(W.try_step, args.workspace, ops, args.note or "", _agent_label(args), args.keep)
    print(_dumps(o, pretty=True) if args.json else W.outcome_text(o))
    return 0 if o["accepted"] else 3


def cmd_build(args) -> int:
    from dronebench import workspace as W
    sets = _ws_call(W.parse_sets, args.set)
    if not sets:
        raise UsageError("nothing to build: pass --set path=value (see params.yaml)")
    o = _ws_call(W.build_step, args.workspace, sets, args.note or "", _agent_label(args), args.keep)
    print(_dumps(o, pretty=True) if args.json else W.outcome_text(o))
    return 0 if o["accepted"] else 3


def cmd_status(args) -> int:
    from dronebench import workspace as W
    if args.json:
        print(_dumps(_ws_call(W.status, args.workspace), pretty=True))
    else:
        print(_ws_call(W.status_text, args.workspace))
    return 0


def cmd_serve(args) -> int:
    from dronebench.server import serve
    _ws_call(serve, args.workspace, args.port, args.open)
    return 0


def cmd_export(args) -> int:
    from dronebench.server import export_site
    out = _ws_call(export_site, args.workspace, args.out)
    print(f"static dashboard written to {out} (serve it from any web host)")
    return 0


def cmd_run(args) -> int:
    from dronebench import agents
    if args.agent == "random":
        r = _ws_call(agents.run_random, args.workspace, args.steps, args.seed)
        print(f"random agent: {r['start']:.2f} -> {r['best']:.2f}")
        return 0
    return _ws_call(agents.run_claude, args.workspace, args.steps, args.model)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dronebench", description="Benchmark small electric fixed-wing UAV designs.")
    ap.add_argument("--debug", action="store_true", help="show tracebacks on errors")
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.required = True

    def common(p, necessity=True):
        p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--preset", choices=PRESETS, default=None, help="scoring preset (default: mission.yaml)")
        if necessity:
            p.add_argument("--no-necessity", action="store_true", help="skip the per-part removal test (fast)")

    p = sub.add_parser("score", help="score a design and print the report")
    p.add_argument("design")
    common(p)
    p.add_argument("--out", help="also write the full report JSON here")
    p.add_argument("--runs", help="append a record to this runs.jsonl")
    p.add_argument("--label", default="agent", help="agent label for the runs record (default: agent)")
    p.add_argument("--step", type=int, default=None, help="step number (default: auto-increment)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pretty", action="store_true", help="indent the JSON")
    g.add_argument("--summary", action="store_true", help="short human-readable summary instead of JSON")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("parts", help="print the per-part report")
    p.add_argument("design")
    common(p)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pretty", action="store_true")
    g.add_argument("--table", action="store_true", help="readable table")
    p.set_defaults(func=cmd_parts)

    p = sub.add_parser("view", help="write a self-contained 3D viewer HTML")
    p.add_argument("designs", nargs="+", metavar="DESIGN", help="one design, or two for before/after")
    common(p, necessity=False)
    p.add_argument("--runs", help="runs.jsonl for the score curve")
    p.add_argument("--out", default="viewer.html")
    p.add_argument("--open", action="store_true", help="open in the default browser")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("edit", help="apply JSON edit ops to a copy of a design")
    p.add_argument("design")
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--ops", required=True, help="JSON file with a list of ops")
    p.add_argument("--out", required=True, help="new design folder")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("baseline", help="random-search baseline")
    p.add_argument("design")
    common(p, necessity=False)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", default=None, help="runs.jsonl to append to")
    p.add_argument("--out", default=None, help="working folder (default: <design>_baseline)")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("init", help="create a workspace (design + brief + step log) for an agent to iterate in")
    p.add_argument("workspace")
    p.add_argument("--fixture", default="falcon_v2", help="falcon_v2 (default), trainer, bad_tail, step_mix, or a design folder")
    p.add_argument("--preset", choices=PRESETS, default=None)
    p.add_argument("--steps", type=int, default=12, help="step budget (default 12)")
    p.add_argument("--force", action="store_true", help="replace a non-empty folder")
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("try", help="apply JSON edit ops to a workspace, score, accept or reject")
    p.add_argument("workspace", nargs="?", default=".")
    p.add_argument("--ops", required=True, help="JSON file with a list of ops ('-' = stdin)")
    p.add_argument("--note", help="one line: what you expect")
    p.add_argument("--keep", action="store_true", help="accept a gate-passing design even if the score dipped")
    p.add_argument("--agent", help="label for the run log (default: $DRONEBENCH_AGENT or 'agent')")
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_try)

    p = sub.add_parser("build", help="change design parameters (params.yaml) and regenerate the aircraft")
    p.add_argument("workspace", nargs="?", default=".")
    p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE", help="e.g. wing.span_mm=2200 (repeatable)")
    p.add_argument("--note")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--agent")
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("status", help="workspace summary: score, tightest checks, removable parts")
    p.add_argument("workspace", nargs="?", default=".")
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("serve", help="live dashboard for a workspace (localhost)")
    p.add_argument("workspace", nargs="?", default=".")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="open in the default browser")
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("export", help="static copy of the dashboard for a finished run (for hosting)")
    p.add_argument("workspace", nargs="?", default=".")
    p.add_argument("--out", required=True)
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("run", help="run a built-in agent in a workspace (random baseline or Claude Code)")
    p.add_argument("workspace", nargs="?", default=".")
    p.add_argument("--agent", choices=("random", "claude"), default="random")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None)
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_run)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:  # argparse exits 2 on usage errors, 0 on --help
        return int(e.code or 0)
    debug = getattr(args, "debug", False)
    try:
        return args.func(args)
    except UsageError as e:
        print(f"dronebench: error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        if debug:
            raise
        print(f"dronebench: internal error: {type(e).__name__}: {e} (rerun with --debug for a traceback)",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
