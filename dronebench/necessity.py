"""Removal test: is each part required, helpful or unnecessary?

Candidates: every non-fastener part on its own, every fastener `group` as one unit, and each
ungrouped fastener on its own. For each candidate we evaluate a copy of the design without it:

  * Design copy: shallow (the parts dict is new, Part objects and meshes are shared, nothing is reloaded).
  * Graph: the base graph minus the removed node(s) (a subgraph view). Edges between the remaining
    parts are unchanged. Electrical links are NOT re-inferred: re-inference would re-wire orphans to
    surviving parts (motor_L to esc_R after esc_L is removed) and hide a real loss. Links to a removed
    part simply disappear, so e.g. removing esc_L leaves motor_L with no ESC and power_chain fails.
  * Geometry: extract_geometry runs for every variant, but geometry.py caches the planform
    measurement by aero-part set, so meshes are only re-sliced when an aero part was removed. The
    non-aero wetted area is recomputed, so removing an external fairing lowers drag.
  * Mass, physics, gate and score always rerun.

Verdicts (if the base design already fails the gate, only checks that newly fail count):
  required     the gate fails (reason: failing check names + the first failure's detail), a locked
               payload part (payload check), or a fastener unit whose removal leaves a structural /
               propulsion part with no fastener at all (rule "joint_fastened"), or a part whose removal
               cuts a critical part (wing, tail, motor, battery, ...) off the airframe's load path: the
               physical/mechanical graph (fasteners bridge what they hold, wires carry no load) from the
               heaviest fuselage part (rule "load_path"). Both rules live here, not in the gate.
  helpful      the gate passes and either the score drops, or a margin moves INTO its caution band
               (CAUTION below: static margin < 0.08 or > 0.18, cruise/stall < 1.4, climb < 3 m/s, spar SF < 2,
               servo margin < 2, motor/ESC/battery current margin < 1.15, Vh < 0.40, Vv < 0.025), or one
               already inside its band gets > 5 % worse.
  unnecessary  the gate passes, no check moves near its limit and the score does not drop.
"""
from __future__ import annotations

import dataclasses
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .gate import SM_LIMITS
from .model import Design, GateResult, NecessityResult, PhysicsResult

MARGIN_DROP = 0.05
# parts that must stay attached: losing every fastener on one of these is a failure
CRITICAL_TYPES = {"motor", "prop", "mount", "wing", "tail", "spar", "fuselage", "surface", "hinge", "servo",
                  "battery", "horn", "linkage", "esc", "payload", "camera"}
AERO_TYPES = {"wing", "tail", "surface", "fuselage"}
AERO_ROLES = {"wing", "htail", "vtail", "vtail_ruddervator", "aileron", "flap", "elevator", "rudder",
              "ruddervator", "elevon", "fuselage", "nacelle"}
# Caution bands (value inside the band = "near its limit"): (low, high); None = open.
CAUTION = {
    "static_margin": (0.08, 0.18),   # gate 0.05-0.20
    "stall_margin": (1.4, None),     # cruise/stall; gate 1.3
    "climb": (3.0, None),            # m/s; gate 2
    "spar_sf": (2.0, None),          # gate 1.5
    "servo_margin": (2.0, None),     # gate 1.5
    "motor_current": (1.15, None),   # limit / full-throttle current; gate 1.0
    "esc_current": (1.15, None),
    "battery_current": (1.15, None), # C rating / C needed; gate 1.0
    "tail_vh": (0.40, None),         # gate 0.35-0.7
    "tail_vv": (0.025, None),        # gate 0.02-0.06
}


def _ok(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def margins(pres: PhysicsResult, pin=None) -> dict[str, float]:
    """Raw values of the quantities the helpful test watches (see CAUTION)."""
    m = {}
    if _ok(pres.static_margin):
        m["static_margin"] = pres.static_margin
    if _ok(pres.cruise_ms) and _ok(pres.stall_ms) and pres.stall_ms > 0:
        m["stall_margin"] = pres.cruise_ms / pres.stall_ms
    if _ok(pres.climb_ms):
        m["climb"] = pres.climb_ms
    if _ok(pres.spar_sf):
        m["spar_sf"] = pres.spar_sf
    sv = [v for v in pres.servo_margin.values() if _ok(v)]
    if sv:
        m["servo_margin"] = min(sv)
    if pin is not None and _ok(pres.motor_current_max_a) and pres.motor_current_max_a > 0 and pin.trains:
        m["motor_current"] = min(t.motor.imax_a for t in pin.trains) / pres.motor_current_max_a
        m["esc_current"] = min(t.esc_imax_a for t in pin.trains) / pres.motor_current_max_a
    if pin is not None and pin.battery is not None and _ok(pres.battery_c_max) and pres.battery_c_max > 0:
        m["battery_current"] = pin.battery.c_rating / pres.battery_c_max
    if _ok(pres.tail_vh):
        m["tail_vh"] = pres.tail_vh
    if _ok(pres.tail_vv):
        m["tail_vv"] = pres.tail_vv
    return {k: float(v) for k, v in m.items()}


def in_caution(name: str, value: float) -> bool:
    lo, hi = CAUTION.get(name, (None, None))
    return (lo is not None and value < lo) or (hi is not None and value > hi)


def _band(name: str) -> str:
    lo, hi = CAUTION[name]
    return " or ".join(x for x in ((f"< {lo:g}" if lo is not None else ""), (f"> {hi:g}" if hi is not None else "")) if x)


def caution_moves(before: dict, after: dict) -> dict[str, str]:
    """name -> description, for margins that move INTO their caution band (or, already inside it,
    get worse by > MARGIN_DROP relative)."""
    out = {}
    for k, a in after.items():
        b = before.get(k)
        if b is None or k not in CAUTION or not in_caution(k, a):
            continue
        if not in_caution(k, b):
            out[k] = f"{k} {b:.3g}→{a:.3g} (caution {_band(k)})"
        else:
            lo, hi = CAUTION[k]
            worse = (lo is not None and a < b and (b - a) > MARGIN_DROP * abs(b)) or \
                    (hi is not None and a > b and (a - b) > MARGIN_DROP * abs(b))
            if worse:
                out[k] = f"{k} {b:.3g}→{a:.3g} (already in caution {_band(k)}, worse)"
    return out


# ---------------------------------------------------------------- candidates

def candidates(design: Design) -> list[tuple[str, list[str]]]:
    """[(key, [part ids])]. Fastener groups are keyed 'group:<name>'."""
    out, groups = [], {}
    for p in design.parts.values():
        if p.type == "fastener" and p.group:
            groups.setdefault(p.group, []).append(p.id)
        else:
            out.append((p.id, [p.id]))
    out += [(f"group:{g}", ids) for g, ids in groups.items()]
    return out


def key_for(part) -> str:
    return f"group:{part.group}" if part.type == "fastener" and part.group else part.id


def fastened_parts(design: Design, fid: str, links) -> set[str]:
    """Non-fastener parts a fastener holds: graph/connects/parent links, part ids named in its group or
    notes, and (if none of those) the parts whose bounding box (+5 mm) contains the fastener's CG."""
    import re
    import numpy as np
    p = design.parts[fid]
    held = {n for n in links.nbrs(fid) if design.parts[n].type != "fastener"}
    if p.parent in design.parts:
        held.add(p.parent)
    others = [q for q in design.parts.values() if q.type != "fastener" and len(q.id) > 2]
    text = f"{p.group or ''} {p.notes or ''}"
    held |= {q.id for q in others if re.search(rf"(?<![A-Za-z0-9]){re.escape(q.id)}(?![A-Za-z0-9])", text)}
    if not held and p.cg_m is not None:
        c = np.asarray(p.cg_m, float)
        held |= {q.id for q in others if q.bbox_m is not None
                 and np.all(c >= np.asarray(q.bbox_m)[0] - 0.005) and np.all(c <= np.asarray(q.bbox_m)[1] + 0.005)}
    return held


def joint_rule(design: Design, removed: list[str], links) -> Optional[str]:
    """Reason string if removing these fasteners leaves a critical part with no fastener."""
    removed_set = set(removed)
    held_by_removed: dict[str, set] = {}
    for fid in removed:
        for h in fastened_parts(design, fid, links):
            held_by_removed.setdefault(h, set()).add(fid)
    if not held_by_removed:
        return None
    still = set()
    for f in design.of_type("fastener"):
        if f.id not in removed_set:
            still |= fastened_parts(design, f.id, links)
    for h in sorted(held_by_removed):
        part = design.parts[h]
        if h not in still and (part.type in CRITICAL_TYPES or part.locked):
            grp = design.parts[removed[0]].group or removed[0]
            return f"joint {grp} unfastened: {h} would detach"
    return None


NON_STRUCTURAL = {"wire", "connector"}


def structure_adjacency(design: Design, links) -> dict[str, set[str]]:
    """Load-carrying adjacency: physical/mechanical links (no wires), fasteners bridge what they hold."""
    adj: dict[str, set[str]] = {pid: set() for pid in design.parts}
    for pid, p in design.parts.items():
        if p.type in NON_STRUCTURAL:
            continue
        for n in links.nbrs(pid, ("physical", "mechanical")):
            if design.parts[n].type not in NON_STRUCTURAL:
                adj[pid].add(n)
                adj[n].add(pid)
        if p.type == "fastener":
            for h in fastened_parts(design, pid, links):
                adj[pid].add(h)
                adj[h].add(pid)
    return adj


def airframe_root(design: Design, mass=None) -> Optional[str]:
    from .assemble import part_mass_g
    fus = [p for p in design.parts.values() if p.type == "fuselage" or p.role == "fuselage"]
    pool = fus or list(design.parts.values())
    return max(pool, key=lambda p: part_mass_g(p, mass)).id if pool else None


def _reach(adj: dict, root: str, removed: set) -> set:
    seen, stack = {root}, [root]
    while stack:
        for n in adj.get(stack.pop(), ()):
            if n not in seen and n not in removed:
                seen.add(n)
                stack.append(n)
    return seen


def load_path_rule(design: Design, removed: list[str], adj: dict, root: Optional[str],
                   attached: set) -> Optional[str]:
    """Reason string if removing these parts cuts a critical part off the airframe's load path."""
    rm = set(removed)
    if root is None or root in rm:
        return None
    rank = {t: k for k, t in enumerate(("wing", "tail", "motor", "battery", "payload", "camera", "spar", "fuselage"))}
    lost = sorted((i for i in (attached - rm - _reach(adj, root, rm))
                   if design.parts[i].type in CRITICAL_TYPES or design.parts[i].locked),
                  key=lambda i: (not design.parts[i].locked, rank.get(design.parts[i].type, 99), i))
    if not lost:
        return None
    more = f" +{len(lost) - 4} more" if len(lost) > 4 else ""
    return f"load path: without it {', '.join(lost[:4])}{more} no longer attach to {root}"


# ---------------------------------------------------------------- evaluation

def variant(design: Design, removed: list[str]) -> Design:
    rm = set(removed)
    return dataclasses.replace(design, parts={k: v for k, v in design.parts.items() if k not in rm},
                               warnings=[])


def needs_geometry(design: Design, removed: list[str]) -> bool:
    return any(design.parts[i].type in AERO_TYPES or design.parts[i].role in AERO_ROLES for i in removed)


def default_evaluator(var: Design, removed: list[str], base: dict, preset: str) -> dict:
    from .pipeline import core
    G = base.get("graph")
    if G is not None:
        G = G.subgraph([n for n in G.nodes if n in var.parts])
    return core(var, preset, graph=G)


def judge(key: str, removed: list[str], base: dict, res: dict) -> NecessityResult:
    gate: GateResult = res["gate"]
    rid = key[6:] if key.startswith("group:") else key
    before = base.get("margins") or margins(base["pres"], base.get("pin"))
    base_score = base["score"][0] if isinstance(base["score"], tuple) else base["score"]
    new_score = res["score"][0] if isinstance(res["score"], tuple) else res["score"]
    delta = float(new_score - base_score)
    base_fails = {c.name for c in base["gate"].failures()} if base.get("gate") is not None else set()
    fails = [c for c in gate.failures() if c.name not in base_fails]
    if fails:
        pre = "gate fail: " if not base_fails else "new gate failures (base already fails " \
            f"{', '.join(sorted(base_fails))}): "
        return NecessityResult(rid, "required", pre + ", ".join(c.name for c in fails)
                               + (f" ({fails[0].detail})" if fails[0].detail else ""), delta, before, {})
    after = margins(res["pres"], res.get("pin"))
    moves = caution_moves(before, after)
    if moves:
        return NecessityResult(rid, "helpful", "margin into caution band: " + "; ".join(moves.values()),
                               delta, before, after)
    if delta < -1e-9:
        return NecessityResult(rid, "helpful", f"score −{-delta:.3g}", delta, before, after)
    pre = "" if gate.passed else "no new gate failures; "
    return NecessityResult(rid, "unnecessary",
                           f"{pre}no check near its limit; score {'+' if delta >= 0 else ''}{delta:.3g}",
                           delta, before, after)


def removal_test(design: Design, preset: str, base: dict, evaluator: Optional[Callable] = None,
                 workers: int = 1, timing: Optional[dict] = None) -> dict[str, NecessityResult]:
    """base: {"design", "graph", "geom", "pres", "pin", "gate", "score", "links"} from pipeline.core.
    Returns {key: NecessityResult}; fastener groups are keyed 'group:<name>' (see key_for)."""
    from .assemble import LinkIndex
    evaluator = evaluator or default_evaluator
    base = dict(base)
    base.setdefault("design", design)
    base["margins"] = margins(base["pres"], base.get("pin"))
    links = base.get("links") or LinkIndex(design, base.get("graph"))
    t0 = time.perf_counter()
    out: dict[str, NecessityResult] = {}
    todo = []
    for key, ids in candidates(design):
        rid = key[6:] if key.startswith("group:") else key
        locked = [i for i in ids if design.parts[i].locked or i in design.mission.payload_ids]
        if locked:
            out[key] = NecessityResult(rid, "required", f"payload check: {', '.join(locked)} is locked mission payload",
                                       0.0, base["margins"], {})
            continue
        if all(design.parts[i].type == "fastener" for i in ids):
            why = joint_rule(design, ids, links)
            if why:
                out[key] = NecessityResult(rid, "required", why, 0.0, base["margins"], {})
                continue
        todo.append((key, ids))

    adj = structure_adjacency(design, links)
    root = airframe_root(design, base.get("mass"))
    attached = _reach(adj, root, set()) if root else set()

    def run(item):
        key, ids = item
        try:
            res = evaluator(variant(design, ids), ids, base, preset)
            r = judge(key, ids, base, res)
            if r.verdict != "required":
                why = load_path_rule(design, ids, adj, root, attached)
                if why:
                    r = NecessityResult(r.id, "required", why, r.score_delta, r.margins_before, r.margins_after)
            return key, r
        except Exception as e:  # a crash while evaluating the variant counts as required, with the reason
            rid = key[6:] if key.startswith("group:") else key
            return key, NecessityResult(rid, "required", f"evaluation failed without it: {type(e).__name__}: {e}")

    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run, todo))
    else:
        results = [run(i) for i in todo]
    out.update(dict(results))
    if timing is not None:
        timing["necessity_ms"] = (time.perf_counter() - t0) * 1000
        timing["necessity_candidates"] = len(out)
    return out
