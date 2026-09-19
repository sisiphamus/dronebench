"""Turn a Design (+ graph, geometry, mass) into the physics input.

Also holds `LinkIndex`, the small link lookup the gate, functions and necessity
modules share. It merges three link sources: graph edges (builder A's format:
edge attrs `kind` and `links: [{kind, joint, gap_mm, tolerance, source}]`),
the BOM `connects` lists (both directions, any kind) and `parent` (mechanical).
It works with graph=None, so tests and callers never depend on graph.py.
"""
from __future__ import annotations

import re
from collections import deque
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np

from .model import (AeroGeometry, BatterySpec, Design, MassProps, Material, MotorSpec, Part, PhysicsInput,
                    PowerTrain, PropSpec, ServoSpec, SparSpec)

CONTROL_ROLES = ("aileron", "flap", "elevator", "rudder", "ruddervator", "elevon")
# parts a power / signal path may pass through between two real components
PASS_THROUGH = ("pdb", "wire", "connector")
ELECTRICAL_TYPES = {"motor", "esc", "battery", "servo", "fc", "rx", "gps", "telemetry", "vtx", "camera",
                    "remote_id", "pdb", "wire", "connector", "sensor"}
LOAD_FACTOR = 3.5


# ---------------------------------------------------------------- links

def edge_links(G, u, v) -> list[dict]:
    """All links on edge u-v as dicts (handles both the `links` list and a bare edge)."""
    if G is None or not G.has_edge(u, v):
        return []
    d = G.get_edge_data(u, v) or {}
    links = d.get("links")
    if links:
        return [dict(l) for l in links]
    return [dict(d)] if d else [{}]


class LinkIndex:
    """Adjacency by link kind over one design. kinds: physical | electrical | mechanical | any."""

    def __init__(self, design: Design, G=None):
        self.design = design
        self.G = G
        self.adj: dict[str, dict[str, set[str]]] = {}   # id -> neighbour -> kinds
        ids = set(design.parts)

        def add(a, b, kind):
            if a == b or a not in ids or b not in ids:
                return
            self.adj.setdefault(a, {}).setdefault(b, set()).add(kind)
            self.adj.setdefault(b, {}).setdefault(a, set()).add(kind)

        if G is not None:
            for u, v in G.edges():
                for l in edge_links(G, u, v):
                    add(u, v, l.get("kind") or "physical")
        for p in design.parts.values():
            for c in p.connects:
                # like graph.py: a connect between two electrical parts is a wire, otherwise any kind
                both = c in ids and p.type in ELECTRICAL_TYPES and design.parts[c].type in ELECTRICAL_TYPES
                add(p.id, c, "electrical" if both else "any")
            if p.parent:
                add(p.id, p.parent, "mechanical")

    def nbrs(self, pid: str, kinds: Optional[Iterable[str]] = None, types: Optional[Iterable[str]] = None) -> list[str]:
        kinds = set(kinds) if kinds else None
        types = set(types) if types else None
        out = []
        for n, ks in sorted(self.adj.get(pid, {}).items()):
            if kinds is not None and not ("any" in ks or ks & kinds):
                continue
            if types is not None and self.design.parts[n].type not in types:
                continue
            out.append(n)
        return out

    def path_to(self, start: str, target_types: Iterable[str], kinds=("electrical",),
                through: Iterable[str] = PASS_THROUGH, max_depth: int = 4) -> Optional[str]:
        """First part of target_types reachable from start, passing only through `through` types."""
        target_types, through = set(target_types), set(through)
        seen, q = {start}, deque([(start, 0)])
        while q:
            cur, d = q.popleft()
            if d >= max_depth:
                continue
            for n in self.nbrs(cur, kinds):
                if n in seen:
                    continue
                seen.add(n)
                t = self.design.parts[n].type
                if t in target_types:
                    return n
                if t in through:
                    q.append((n, d + 1))
        return None


def part_mass_g(part: Part, mass: Optional[MassProps] = None) -> float:
    if mass is not None and part.id in mass.by_part_g:
        return float(mass.by_part_g[part.id])
    if part.mass_g is not None:  # ingest stores one unit's mass; qty multiplies
        return float(part.mass_g) * max(part.qty, 1)
    return float(part.specs.get("mass_g") or 0.0) * max(part.qty, 1)


def part_cg(part: Part) -> Optional[np.ndarray]:
    if part.cg_m is not None:
        return np.asarray(part.cg_m, float)
    if part.bbox_m is not None:
        return np.asarray(part.bbox_m, float).mean(axis=0)
    return None


# ---------------------------------------------------------------- specs

@lru_cache(maxsize=512)
def _catalog_pn(part_number: str) -> dict:
    try:
        from . import components
        hit = components.lookup(part_number)
    except Exception:
        return {}
    if not isinstance(hit, dict):
        return {}
    specs = hit.get("specs") if isinstance(hit.get("specs"), dict) else hit
    return dict(specs)


def _catalog(part: Part) -> dict:
    """Catalog specs for a part number (components.lookup, memoised), flattened to a spec dict."""
    return _catalog_pn(part.part_number) if part.part_number else {}


def _num(v):
    try:
        return float(v) if isinstance(v, str) and re.fullmatch(r"[-+0-9.eE]+", v) else v
    except ValueError:
        return v


def spec(part: Part, key: str, *alts: str, default=None):
    """Spec value from the BOM, then the catalog. Numeric strings are converted."""
    keys = (key, *alts)
    for k in keys:
        if part.specs.get(k) is not None:
            return _num(part.specs[k])
    cat = _catalog(part)
    for k in keys:
        if cat.get(k) is not None:
            return _num(cat[k])
    return default


_PROP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)")
_S_RE = re.compile(r"(\d+)\s*[sS](?![a-zA-Z])")
_MAH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mAh", re.I)


def prop_dims(part: Part) -> tuple[Optional[float], Optional[float]]:
    d, p = spec(part, "diam_in", "diameter_in"), spec(part, "pitch_in")
    if (d is None or p is None) and part.part_number:
        m = _PROP_RE.search(part.part_number)
        if m:
            d = d if d is not None else float(m.group(1))
            p = p if p is not None else float(m.group(2))
    if d is None and part.bbox_m is not None:
        ext = np.ptp(np.asarray(part.bbox_m, float), axis=0)
        d = float(ext.max()) / 0.0254
    if p is None and d is not None:
        p = 0.7 * d
    return (float(d) if d is not None else None, float(p) if p is not None else None)


def motor_spec(part: Part) -> Optional[MotorSpec]:
    kv = spec(part, "kv")
    if kv is None and part.part_number:
        m = re.search(r"(\d{2,5})\s*kv", part.part_number, re.I)
        kv = float(m.group(1)) if m else None
    if kv is None:
        return None
    return MotorSpec(id=part.id, kv=float(kv), rm_ohm=float(spec(part, "rm_ohm", default=0.05)),
                     i0_a=float(spec(part, "i0_a", default=1.0)), imax_a=float(spec(part, "imax_a", default=30.0)),
                     mass_g=float(spec(part, "mass_g", default=part.mass_g or 0.0)))


def battery_spec(part: Part) -> Optional[BatterySpec]:
    s = spec(part, "cells_s", "s", "cells")
    cap = spec(part, "capacity_mah", "mah")
    pn = part.part_number or ""
    if s is None and _S_RE.search(pn):
        s = float(_S_RE.search(pn).group(1))
    if cap is None and _MAH_RE.search(pn):
        cap = float(_MAH_RE.search(pn).group(1))
    if s is None or cap is None:
        return None
    cells_p = spec(part, "cells_p")
    if cells_p is None:
        m = re.search(r"\d\s*[sS]\s*(\d+)\s*[pP]", pn)
        cells_p = float(m.group(1)) if m else 1
    chem = str(spec(part, "chemistry", default="lipo")).lower().replace("-", "").replace(" ", "")
    chem = "liion" if chem in ("liion", "li", "18650", "21700", "lion") else "lipo"
    return BatterySpec(id=part.id, cells_s=int(s), capacity_mah=float(cap),
                       c_rating=float(spec(part, "c_rating", "c", default=10.0 if chem == "liion" else 25.0)),
                       chemistry=chem, ir_mohm_per_cell=float(spec(part, "ir_mohm_per_cell",
                                                                   default=15.0 if chem == "liion" else 5.0)),
                       mass_g=float(part.mass_g if part.mass_g is not None else spec(part, "mass_g", default=0.0)),
                       cells_p=max(int(cells_p), 1))


def servo_torque_kgcm(part: Part) -> Optional[float]:
    t = spec(part, "torque_kgcm")
    if t is None:
        nm = spec(part, "torque_nm")
        t = float(nm) * 10.197 if nm is not None else None
    return float(t) if t is not None else None


def _material(part: Part) -> Material:
    if part.material is not None:
        return part.material
    try:
        from . import materials
        return materials.resolve_material(part)
    except Exception:
        return Material(name="CF tube", density_kg_m3=1550, kind="composite", tensile_mpa=600, modulus_gpa=70,
                        source="assumed")


def spar_spec(part: Part, warnings: Optional[list] = None) -> Optional[SparSpec]:
    od, idd = spec(part, "outer_d_mm", "od_mm"), spec(part, "inner_d_mm", "id_mm")
    length = spec(part, "length_m")
    ext = np.ptp(np.asarray(part.bbox_m, float), axis=0) if part.bbox_m is not None else None
    if length is None and spec(part, "length_mm") is not None:
        length = float(spec(part, "length_mm")) / 1000
    if ext is not None:
        axis = int(np.argmax(ext))
        if length is None:
            length = float(ext[axis])
        if od is None:
            od = float(np.delete(ext, axis).min()) * 1000
            part.sources.setdefault("outer_d_mm", "inferred")
        if idd is None:
            idd = max(float(od) - 2.0, 0.0)
            part.sources.setdefault("inner_d_mm", "assumed")
    if od is None or length is None:
        if warnings is not None:
            warnings.append(f"{part.id}: spar has no diameter/length (no specs, no CAD); ignored by structure")
        return None
    if idd is None:
        idd = max(float(od) - 2.0, 0.0)
    role = part.role if part.role in ("main_spar", "rear_spar", "tail_boom") else None
    if role is None:
        role = "tail_boom" if ext is not None and int(np.argmax(ext)) == 0 else "main_spar"
    # spans_root: the spar crosses the centreline (bbox y-range reaches within 20 mm of y=0); None if no mesh
    # y_min_m / y_max_m: the spar's spanwise extent (our frame, metres); None if no mesh
    spans_root = y_min = y_max = None
    if part.bbox_m is not None:
        bb = np.asarray(part.bbox_m, float)
        y_min, y_max = float(bb[0, 1]), float(bb[1, 1])
        spans_root = bool(y_min <= 0.02 and y_max >= -0.02)
    extra = {k: v for k, v in (("y_min_m", y_min), ("y_max_m", y_max))
             if k in SparSpec.__dataclass_fields__}   # optional fields (physics builder B)
    return SparSpec(id=part.id, outer_d_mm=float(od), inner_d_mm=float(idd), length_m=float(length),
                    material=_material(part), role=role, spans_root=spans_root, **extra)


def servo_surface(part: Part, links: LinkIndex) -> Optional[str]:
    """Control surface a servo drives: connects/parent, else via linkage/horn in the graph."""
    parts = links.design.parts
    for c in list(part.connects) + [part.parent]:
        if c in parts and parts[c].role in CONTROL_ROLES:
            return c
    seen, q = {part.id}, deque([(part.id, 0)])
    while q:
        cur, d = q.popleft()
        for n in links.nbrs(cur, ("mechanical", "physical")):
            if n in seen:
                continue
            seen.add(n)
            if parts[n].role in CONTROL_ROLES:
                if parts[n].type != "servo":
                    return n
            if parts[n].type in ("linkage", "horn", "hinge") and d < 3:
                q.append((n, d + 1))
    return None


# ---------------------------------------------------------------- trains

def pair_motors(design: Design, links: LinkIndex) -> dict[str, dict]:
    """motor id -> {"prop": id|None, "esc": id|None}. Props by link first, then nearest unused by CG."""
    motors = design.of_type("motor")
    used_props: set[str] = set()
    out = {}
    for m in motors:
        props = [p for p in links.nbrs(m.id, types=("prop",)) if p not in used_props]
        prop = props[0] if props else None
        esc = links.path_to(m.id, ("esc",), kinds=("electrical",))
        out[m.id] = {"prop": prop, "esc": esc}
        if prop:
            used_props.add(prop)
    for m in motors:  # nearest unlinked prop
        if out[m.id]["prop"] is None:
            mc = part_cg(m)
            free = [p for p in design.of_type("prop") if p.id not in used_props]
            if mc is not None and free:
                best = min(free, key=lambda p: np.linalg.norm((part_cg(p) if part_cg(p) is not None else mc + 1e3) - mc))
                if part_cg(best) is not None and np.linalg.norm(part_cg(best) - mc) < 0.15:
                    out[m.id]["prop"] = best.id
                    used_props.add(best.id)
    return out


def combine_batteries(design: Design, warnings: list) -> Optional[BatterySpec]:
    """Combine all battery parts. Same S: parallel packs (physics convention: capacity_mah is PER PACK,
    mean of the packs, x parallel_packs = total; mass_g is the total; c_rating the weakest pack).
    Different S: largest-energy pack only, with a warning."""
    specs = []
    for b in design.of_type("battery"):
        bs = battery_spec(b)
        if bs is None:
            warnings.append(f"{b.id}: battery has no cells_s/capacity_mah; ignored")
            continue
        for _ in range(max(b.qty, 1)):
            specs.append(bs)
    if not specs:
        return None
    if len({s.cells_s for s in specs}) > 1:
        best = max(specs, key=lambda s: s.cells_s * s.capacity_mah)
        warnings.append(f"batteries have different cell counts ({sorted({s.cells_s for s in specs})}S); "
                        f"only {best.id} ({best.cells_s}S) is used for power")
        specs = [s for s in specs if s.cells_s == best.cells_s]
    if len(specs) == 1:
        return specs[0]
    return BatterySpec(id="+".join(dict.fromkeys(s.id for s in specs)), cells_s=specs[0].cells_s,
                       capacity_mah=float(np.mean([s.capacity_mah for s in specs])),
                       c_rating=min(s.c_rating for s in specs), chemistry=specs[0].chemistry,
                       ir_mohm_per_cell=float(np.mean([s.ir_mohm_per_cell for s in specs])),
                       mass_g=sum(s.mass_g for s in specs), parallel_packs=len(specs),
                       cells_p=min(s.cells_p for s in specs))


def build_physics_input(design: Design, geom: AeroGeometry, mass: MassProps, graph=None,
                        links: Optional[LinkIndex] = None) -> PhysicsInput:
    warnings = design.warnings
    links = links or LinkIndex(design, graph)
    trains = []
    for mid, pr in pair_motors(design, links).items():
        m = design.parts[mid]
        ms = motor_spec(m)
        if ms is None:
            warnings.append(f"{mid}: motor has no kv; excluded from propulsion")
            continue
        if pr["prop"] is None or pr["esc"] is None:
            continue  # gate power_chain reports it
        d, p = prop_dims(design.parts[pr["prop"]])
        if d is None:
            warnings.append(f"{pr['prop']}: prop diameter unknown; {mid} excluded from propulsion")
            continue
        blades = int(spec(design.parts[pr["prop"]], "blades", default=2))
        esc_imax = float(spec(design.parts[pr["esc"]], "imax_a", default=30.0))
        trains.append(PowerTrain(motor=ms, prop=PropSpec(id=pr["prop"], diam_in=d, pitch_in=p, blades=blades),
                                 esc_imax_a=esc_imax))
    spars = [s for s in (spar_spec(p, warnings) for p in design.of_type("spar")) if s is not None]
    servos = []
    for s in design.of_type("servo"):
        t = servo_torque_kgcm(s)
        if t is None:
            warnings.append(f"{s.id}: servo torque unknown; assumed 1.5 kg·cm")
            t = 1.5
            s.sources.setdefault("torque_kgcm", "assumed")
        servos.append(ServoSpec(id=s.id, torque_kgcm=t, surface_id=servo_surface(s, links)))
    payload_ids = set(design.mission.payload_ids) | {p.id for p in design.parts.values() if p.locked}
    payload_g = sum(part_mass_g(design.parts[i], mass) for i in payload_ids if i in design.parts)
    cruise = design.mission.cruise_kmh / 3.6 if design.mission.cruise_kmh else None
    return PhysicsInput(geom=geom, mass=mass, trains=trains, battery=combine_batteries(design, warnings),
                        spars=spars, servos=servos, payload_kg=payload_g / 1000, cruise_ms=cruise,
                        rho=design.mission.rho, usable_battery_fraction=design.mission.usable_battery_fraction,
                        load_factor=LOAD_FACTOR)
