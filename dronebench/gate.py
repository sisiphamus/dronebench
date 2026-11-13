"""The flight gate: pass/fail checks every design must clear (PLAN §1.6).

Each Check has a value, a limit ([lo, hi], None = open), passed, and a one-sentence detail.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .assemble import CONTROL_ROLES, LinkIndex, pair_motors, part_cg, prop_dims, servo_surface, spec
from .model import AeroGeometry, Check, Design, GateResult, PhysicsResult

SM_LIMITS = (0.05, 0.20)
STALL_RATIO = 1.3
CLIMB_MS = 2.0
SPAR_SF = 1.5
VH_LIMITS = (0.35, 0.7)
VV_LIMITS = (0.02, 0.06)
SERVO_MARGIN = 1.5
CLEARANCE_MM = 1.0
PROP_CLEARANCE_MM = 5.0
# pairs that may overlap on purpose (spar in wing, hinge in surface, screws, inserts)
FIT_TYPES = {"fastener", "spar", "hinge", "linkage", "horn"}
FIT_JOINT_WORDS = ("fit", "insert", "slot", "socket", "embedded", "glued")
PHYSICS_CHECKS = ("static_margin", "stall_margin", "climb", "spar", "tail_volume", "servo_torque",
                  "motor_current", "esc_current", "battery_current")


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _f(x, nd=3) -> str:
    return "nan" if _bad(x) else f"{x:.{nd}g}" if abs(x) < 1e4 else f"{x:.0f}"


def _range_check(name, value, lo, hi, what) -> Check:
    if _bad(value):
        return Check(name, None, [lo, hi], False, f"{what} could not be computed")
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    return Check(name, float(value), [lo, hi], ok, "")


def _physics_note(pres: PhysicsResult) -> str:
    return "; ".join(pres.notes) if pres.notes else "no reason given"


# ---------------------------------------------------------------- individual checks

def check_lifting_surfaces(design: Design, geom: AeroGeometry) -> Check:
    vtail_kind = bool(design.of_role("vtail_ruddervator", "ruddervator"))
    if geom is not None and geom.vtail is not None and 20 <= abs(geom.vtail.dihedral_deg) <= 70:
        vtail_kind = True
    has = {"wing": geom is not None and geom.wing is not None,
           "htail": geom is not None and (geom.htail is not None or (vtail_kind and geom.vtail is not None)),
           "vtail": geom is not None and geom.vtail is not None}
    missing = [k for k, v in has.items() if not v]
    detail = "" if not missing else ("missing: " + ", ".join(
        {"wing": "wing", "htail": "horizontal stabiliser (tail or V-tail)", "vtail": "vertical stabiliser"}[m]
        for m in missing))
    asym = planform_asymmetry(design)
    if asym:
        detail = "; ".join(x for x in (detail, asym) if x)
    return Check("lifting_surfaces", sorted(k for k, v in has.items() if v), ["wing", "htail", "vtail"],
                 not missing and not asym, detail)


ASYMMETRY = 0.10
_SURFACE_GROUPS = {"wing": ("wing", "aileron", "flap", "elevon"), "htail": ("htail", "elevator")}


def planform_asymmetry(design: Design) -> str:
    """Left vs right planform (bbox x-y area of meshed parts) per surface; > 10 % difference = can't trim in roll."""
    probs = []
    for name, roles in _SURFACE_GROUPS.items():
        side = {"left": 0.0, "right": 0.0}
        for p in design.parts.values():
            if p.role not in roles or p.bbox_m is None:
                continue
            bb = np.asarray(p.bbox_m, float)
            cy = bb[:, 1].mean()
            if abs(cy) < 0.01:
                continue
            side["left" if cy < 0 else "right"] += float((bb[1, 0] - bb[0, 0]) * (bb[1, 1] - bb[0, 1]))
        big = max(side.values())
        if big > 0 and abs(side["left"] - side["right"]) / big > ASYMMETRY:
            probs.append(f"{name} asymmetric: left {side['left']:.3f} m² vs right {side['right']:.3f} m² "
                         "(cannot trim in roll)")
    return "; ".join(probs)


def check_static_margin(pres, mass_cg_x=None) -> Check:
    c = _range_check("static_margin", pres.static_margin, *SM_LIMITS, "static margin")
    if c.value is not None and not c.passed:
        lo, hi = SM_LIMITS
        where = f"(x_cg {_f(mass_cg_x)} m, neutral point {_f(pres.neutral_point_x_m)} m)"
        if c.value < lo:
            c.detail = f"static margin {c.value:.3f} < {lo}: CG too far aft {where}"
        else:
            c.detail = f"static margin {c.value:.3f} > {hi}: CG too far forward, tail-heavy trim drag {where}"
    return c


def check_stall_margin(pres) -> Check:
    ratio = pres.cruise_ms / pres.stall_ms if not (_bad(pres.cruise_ms) or _bad(pres.stall_ms)) and pres.stall_ms > 0 else float("nan")
    c = _range_check("stall_margin", ratio, STALL_RATIO, None, "cruise/stall ratio")
    if c.value is not None and not c.passed:
        c.detail = (f"cruise {pres.cruise_ms * 3.6:.1f} km/h is only {c.value:.2f}× stall "
                    f"{pres.stall_ms * 3.6:.1f} km/h (need ≥ {STALL_RATIO}×)")
    return c


def check_climb(pres) -> Check:
    c = _range_check("climb", pres.climb_ms, CLIMB_MS, None, "climb rate")
    if c.value is not None and not c.passed:
        c.detail = f"max climb {c.value:.2f} m/s < {CLIMB_MS} m/s: not enough thrust for {pres.mass_kg:.2f} kg"
    return c


def check_spar(design, pres) -> Check:
    if not design.of_type("spar"):
        return Check("spar", None, [SPAR_SF, None], False, "no spar: wing has no bending structure")
    c = _range_check("spar", pres.spar_sf, SPAR_SF, None, "spar safety factor")
    if c.value is not None and not c.passed:
        c.detail = f"spar safety factor {c.value:.2f} < {SPAR_SF} at {3.5} g: spar would break in a hard pull-up"
    return c


def check_tail_volume(pres) -> Check:
    vh, vv = pres.tail_vh, pres.tail_vv
    limit = {"vh": list(VH_LIMITS), "vv": list(VV_LIMITS)}
    if _bad(vh) or _bad(vv):
        return Check("tail_volume", {"vh": None if _bad(vh) else float(vh), "vv": None if _bad(vv) else float(vv)},
                     limit, False, "tail volume could not be computed (tail missing?)")
    probs = []
    if not VH_LIMITS[0] <= vh <= VH_LIMITS[1]:
        probs.append(f"Vh {vh:.3f} outside {VH_LIMITS[0]}–{VH_LIMITS[1]}"
                     + (" (horizontal tail too small or too close)" if vh < VH_LIMITS[0] else " (oversized)"))
    if not VV_LIMITS[0] <= vv <= VV_LIMITS[1]:
        probs.append(f"Vv {vv:.3f} outside {VV_LIMITS[0]}–{VV_LIMITS[1]}"
                     + (" (fin too small)" if vv < VV_LIMITS[0] else " (oversized)"))
    return Check("tail_volume", {"vh": float(vh), "vv": float(vv)}, limit, not probs, "; ".join(probs))


def check_servo_torque(design, pres, links) -> Check:
    margins, missing = {}, []
    for s in design.of_type("servo"):
        if servo_surface(s, links) is None:
            continue  # not driving a surface (e.g. release servo); control_chain handles surfaces
        m = pres.servo_margin.get(s.id)
        if _bad(m):
            missing.append(s.id)
        else:
            margins[s.id] = float(m)
    if not margins and not missing:
        return Check("servo_torque", None, [SERVO_MARGIN, None], True, "no surface servos")
    weak = {k: v for k, v in margins.items() if v < SERVO_MARGIN}
    worst = min(margins.values()) if margins else None
    parts = []
    if weak:
        parts.append("; ".join(f"{k} margin {v:.2f} < {SERVO_MARGIN}" for k, v in sorted(weak.items(), key=lambda kv: kv[1])))
    if missing:
        parts.append("no hinge-moment margin for " + ", ".join(missing))
    return Check("servo_torque", worst, [SERVO_MARGIN, None], not weak and not missing, "; ".join(parts))


def check_currents(design, pres, pin) -> list[Check]:
    out = []
    trains = pin.trains if pin is not None else []
    imax = pres.motor_current_max_a
    if not trains:
        for n in ("motor_current", "esc_current"):
            out.append(Check(n, None, [None, None], False, "no complete power train (see power_chain)"))
    else:
        mlim = min(t.motor.imax_a for t in trains)
        elim = min(t.esc_imax_a for t in trains)
        mid = min(trains, key=lambda t: t.motor.imax_a).motor.id
        c = _range_check("motor_current", imax, None, mlim, "full-throttle motor current")
        if c.value is not None and not c.passed:
            c.detail = f"full-throttle current {c.value:.1f} A > {mid} rating {mlim:.0f} A: motor would overheat"
        out.append(c)
        c = _range_check("esc_current", imax, None, elim, "full-throttle ESC current")
        if c.value is not None and not c.passed:
            c.detail = f"full-throttle current {c.value:.1f} A > ESC rating {elim:.0f} A"
        out.append(c)
    b = pin.battery if pin is not None else None
    if b is None:
        out.append(Check("battery_current", None, [None, None], False, "no usable battery"))
    else:
        c = _range_check("battery_current", pres.battery_c_max, None, b.c_rating, "battery C needed")
        if c.value is not None and not c.passed:
            c.detail = (f"full throttle needs {c.value:.1f}C ({pres.battery_current_max_a:.1f} A) "
                        f"> battery rating {b.c_rating:.0f}C")
        out.append(c)
    return out


def check_power_chain(design, links) -> Check:
    motors = design.of_type("motor")
    if not motors:
        return Check("power_chain", 0, [1, None], False, "no motor")
    problems, complete = [], 0
    pairs = pair_motors(design, links)
    for m in motors:
        pr = pairs[m.id]
        gaps = []
        if pr["prop"] is None:
            gaps.append("no prop")
        if pr["esc"] is None:
            gaps.append("no ESC linked")
        elif links.path_to(pr["esc"], ("battery",), kinds=("electrical",)) is None:
            gaps.append(f"{pr['esc']} not linked to a battery")
        if gaps:
            problems.append(f"{m.id}: " + ", ".join(gaps))
        else:
            complete += 1
    if not design.of_type("battery"):
        problems.insert(0, "no battery")
    ok = complete == len(motors) and bool(design.of_type("battery"))
    return Check("power_chain", complete, [len(motors), None], ok,
                 "" if ok else "battery→ESC→motor→prop incomplete: " + "; ".join(problems))


def check_control_chain(design, links) -> Check:
    probs = []
    fcs = design.of_type("fc")
    if not fcs:
        probs.append("no flight controller")
    if not design.of_type("rx"):
        probs.append("no receiver")
    surfaces = [p for p in design.parts.values() if p.role in CONTROL_ROLES and p.type != "servo"]
    driven = {}
    for s in design.of_type("servo"):
        surf = servo_surface(s, links)
        if surf:
            driven.setdefault(surf, []).append(s.id)
    roles = {s.role for s in surfaces}
    for axis, need in (("roll", ("aileron", "elevon")), ("pitch", ("elevator", "elevon", "ruddervator")),
                       ("yaw", ("rudder", "ruddervator"))):
        if not roles & set(need):
            probs.append(f"no {axis} control surface ({' / '.join(need)})")
    ok_count = 0
    for surf in surfaces:
        servos = driven.get(surf.id, [])
        if not servos:
            probs.append(f"{surf.id} ({surf.role}) has no servo")
            continue
        wired = [s for s in servos if links.path_to(s, ("fc",), kinds=("electrical",),
                                                        through=("wire", "connector", "pdb", "rx")) is not None]
        if fcs and not wired:
            probs.append(f"{', '.join(servos)} (moves {surf.id}) not linked to the FC")
            continue
        ok_count += 1
    return Check("control_chain", ok_count, [len(surfaces), None], not probs, "; ".join(probs))


def check_payload(design) -> Check:
    want = list(dict.fromkeys(list(design.mission.payload_ids) + [p.id for p in design.parts.values() if p.locked]))
    missing = [i for i in want if i not in design.parts]
    return Check("payload", len(want) - len(missing), [len(want), None], not missing,
                 "" if not missing else "locked payload missing: " + ", ".join(missing))


def _declared_fit(a, b, links_ab) -> bool:
    if any(l.get("declared_fit") for l in links_ab):
        return True  # graph.py marks specs.fits pairs
    if a.type in FIT_TYPES or b.type in FIT_TYPES:
        return True
    if a.parent == b.id or b.parent == a.id or b.id in a.connects or a.id in b.connects:
        return True
    if b.id in (a.specs.get("fits") or []) or a.id in (b.specs.get("fits") or []):
        return True  # BOM-declared fit: specs.fits lists parts this part may sit inside
    return any(any(w in str(l.get("joint") or "").lower() for w in FIT_JOINT_WORDS) for l in links_ab)


def interpenetrations(design, G) -> list[tuple[str, str, float]]:
    from .assemble import edge_links
    out = []
    if G is None:
        return out
    for u, v in G.edges():
        if u not in design.parts or v not in design.parts:
            continue
        ls = [l for l in edge_links(G, u, v) if (l.get("kind") or "physical") == "physical"]
        gaps = [l.get("gap_mm") for l in ls if isinstance(l.get("gap_mm"), (int, float))]
        if not gaps:
            continue
        g = min(gaps)
        if g < -CLEARANCE_MM and not _declared_fit(design.parts[u], design.parts[v], ls):
            out.append((u, v, float(g)))
    return sorted(out)


_PROP_CACHE: dict = {}


def _disc_points(center, axis, r) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    pts = [center + rr * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w)) for rr in (0.45 * r, 0.7 * r, r)]
    return np.vstack(pts)


def prop_strikes(design, links) -> list[tuple[str, str, float]]:
    """(prop, part, distance_mm) where an airframe mesh is within 5 mm of the prop disc.
    Results are memoised per (prop mesh, other mesh) pair, so removal-test reruns cost nothing."""
    out = []
    pairs = {v["prop"]: k for k, v in pair_motors(design, links).items() if v["prop"]}
    for prop in design.of_type("prop"):
        d, _ = prop_dims(prop)
        c = part_cg(prop)
        if d is None or c is None:
            continue
        r = d * 0.0254 / 2
        motor = design.parts.get(pairs.get(prop.id, ""), None)
        axis = np.array([1.0, 0.0, 0.0])
        if motor is not None and part_cg(motor) is not None:
            a = c - part_cg(motor)
            if np.linalg.norm(a) > 1e-3:
                axis = a / np.linalg.norm(a)
        pts = None
        near = set(links.nbrs(prop.id)) | ({motor.id} if motor is not None else set())
        for o in design.parts.values():
            if o.id == prop.id or o.id in near or o.mesh is None or o.bbox_m is None or o.type in (
                    "prop", "motor", "fastener", "wire", "connector"):
                continue
            bb = np.asarray(o.bbox_m, float)
            if np.any(c + r + 0.01 < bb[0]) or np.any(c - r - 0.01 > bb[1]):
                continue
            key = (prop.id, id(prop.mesh), tuple(np.round(c, 5)), o.id, id(o.mesh), round(r, 5))
            dist = _PROP_CACHE.get(key)
            if dist is None:
                if pts is None:
                    pts = _disc_points(c, axis, r)
                try:
                    _, dd, _ = o.mesh.nearest.on_surface(pts)
                    dist = float(np.min(dd)) * 1000
                    if o.mesh.is_watertight and np.any(o.mesh.contains(pts)):
                        dist = 0.0
                except Exception:
                    dist = float("inf")
                _PROP_CACHE[key] = dist
            if dist < PROP_CLEARANCE_MM:
                out.append((prop.id, o.id, dist))
    return out


def check_clearance(design, G, links) -> Check:
    pen = interpenetrations(design, G)
    strikes = prop_strikes(design, links)
    probs = []
    if pen:
        probs.append("interpenetrating: " + ", ".join(f"{u}/{v} {g:.1f} mm" for u, v, g in pen[:8]))
    if strikes:
        probs.append("prop disc hits: " + ", ".join(f"{p}→{o} ({d:.0f} mm)" for p, o, d in strikes[:8]))
    return Check("clearance", len(pen) + len(strikes), [None, 0], not probs, "; ".join(probs))


def check_harness(design: Design, G) -> Check:
    """Every ESC needs a power lead from the battery, and the wiring must weigh at least a minimum for the
    ESCs, servos and FC it serves (10 g/ESC, 4 g/servo, 5 g FC). Deleting the wire parts to save their mass is
    not a design improvement."""
    if G is None:
        return Check("harness", 0, [None, 0], True, "no part graph given; not checked")

    def wires_of(pid):
        if pid not in G:
            return set()
        return {n for n in G.neighbors(pid) if G.nodes[n].get("type") == "wire"
                and any(l["kind"] == "electrical" for l in G.edges[pid, n]["links"])}

    probs = []
    batts = [b.id for b in design.of_type("battery")]
    fcs = [f.id for f in design.of_type("fc")]
    for e in design.of_type("esc"):
        if batts and not any(wires_of(e.id) & wires_of(b) for b in batts):
            probs.append(f"{e.id}: no power wiring to the battery")
    wires = design.of_type("wire")
    have = sum(float(w.mass_g or 0) for w in wires)
    need = 10.0 * len(design.of_type("esc")) + 4.0 * len(design.of_type("servo")) + (5.0 if fcs else 0.0)
    if have + 1e-9 < need:
        probs.append(f"harness mass {have:.0f} g < {need:.0f} g minimum for {len(design.of_type('esc'))} ESC(s), "
                     f"{len(design.of_type('servo'))} servo(s)")
    return Check("harness", len(probs), [None, 0], not probs, "; ".join(probs[:6]))


CONNECT_MM = 25.0   # parts closer than this are treated as held together (standoffs, brackets, mounting pods)


def _surface_points(part, n=300) -> np.ndarray:
    import trimesh
    pts, _ = trimesh.sample.sample_surface(part.mesh, n, seed=0)
    return np.vstack([np.asarray(part.mesh.vertices), pts])


def airframe_clusters(design: Design, G) -> list[list[str]]:
    """Groups of meshed parts that hang together: physical contacts, then anything within CONNECT_MM."""
    import networkx as nx
    from scipy.spatial import cKDTree
    meshed = [p for p in design.parts.values() if getattr(p, "mesh", None) is not None]
    h = nx.Graph()
    h.add_nodes_from(p.id for p in meshed)
    for u, v, a in G.edges(data=True):
        if u in h and v in h and any(l["kind"] == "physical" for l in a["links"]):
            h.add_edge(u, v)
    comps = list(nx.connected_components(h))
    if len(comps) > 1:
        trees, boxes = {}, {}
        by_id = {p.id: p for p in meshed}
        for i, c in enumerate(comps):
            pts = np.vstack([_surface_points(by_id[x]) for x in c])
            trees[i] = cKDTree(pts)
            boxes[i] = (pts.min(0), pts.max(0))
        cg = nx.Graph()
        cg.add_nodes_from(trees)
        lim = CONNECT_MM * 1e-3
        for i in trees:
            for j in trees:
                if i < j:
                    lo_i, hi_i = boxes[i]; lo_j, hi_j = boxes[j]
                    if (np.maximum(lo_i, lo_j) - np.minimum(hi_i, hi_j) > lim).any():
                        continue
                    d, _ = trees[j].query(trees[i].data, distance_upper_bound=lim)
                    if np.isfinite(d).any():
                        cg.add_edge(i, j)
        merged = [sorted(x for k in grp for x in comps[k]) for grp in nx.connected_components(cg)]
    else:
        merged = [sorted(c) for c in comps]
    return sorted(merged, key=len, reverse=True)


def check_airframe_connected(design: Design, G) -> Check:
    if G is None:
        return Check("airframe_connected", 1, [None, 1], True, "no part graph given; not checked")
    groups = airframe_clusters(design, G)
    if len(groups) <= 1:
        return Check("airframe_connected", 1, [None, 1], True, "every part is attached to the airframe")
    stray = [x for g in groups[1:] for x in g]
    return Check("airframe_connected", len(groups), [None, 1], False,
                 f"{len(groups)} separate pieces; not attached to the main airframe: " + ", ".join(stray[:10]))


# ---------------------------------------------------------------- gate

def evaluate_gate(design: Design, graph, geom: AeroGeometry, pres: PhysicsResult, pin=None,
                  mass=None, links: Optional[LinkIndex] = None) -> GateResult:
    """Run every check. `pin` (PhysicsInput) gives the ratings; rebuilt if not given."""
    links = links or LinkIndex(design, graph)
    if pin is None:
        from .assemble import build_physics_input
        from .model import MassProps
        pin = build_physics_input(design, geom, mass or MassProps(pres.mass_kg, np.zeros(3)), graph, links=links)
    cg_x = float(mass.cg_m[0]) if mass is not None else float(pin.mass.cg_m[0]) if pin.mass is not None else None
    phys = [check_static_margin(pres, cg_x), check_stall_margin(pres), check_climb(pres), check_spar(design, pres),
            check_tail_volume(pres), check_servo_torque(design, pres, links), *check_currents(design, pres, pin)]
    if not pres.ok:
        note = _physics_note(pres)
        for c in phys:
            if c.name == "servo_torque" and c.passed and c.value is None:
                continue
            c.passed = False
            c.detail = f"physics could not run: {note}" + (f" ({c.detail})" if c.detail else "")
    checks = [check_lifting_surfaces(design, geom), *phys, check_power_chain(design, links), check_control_chain(design, links), check_payload(design),
              check_clearance(design, graph, links), check_airframe_connected(design, graph), check_harness(design, graph)]
    order = ["lifting_surfaces", "static_margin", "stall_margin", "climb", "spar", "tail_volume", "servo_torque",
             "motor_current", "esc_current", "battery_current", "power_chain", "control_chain", "payload", "clearance", "airframe_connected", "harness"]
    checks.sort(key=lambda c: order.index(c.name))
    return GateResult(passed=all(c.passed for c in checks), checks=checks)
