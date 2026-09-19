"""Part graph: which parts touch, and which are wired or mechanically linked.

``build_graph(design)`` returns an undirected ``networkx.Graph``:
  * node = part id, attrs ``type``, ``role``, ``group``.
  * edge attr ``links``: list of dicts, one per relation between the two parts (a pair can be both
    touching and wired), each ``{"kind", "joint", "gap_mm", "tolerance", "source", "from", "to", "declared_fit"}``:
      kind       "physical" | "electrical" | "mechanical"
      joint      e.g. "tube in printed bore, slip fit", "power", "PWM signal", "drives", "pushrod linkage"
      gap_mm     measured surface gap (physical only; negative = interpenetration depth), else None
      tolerance  inferred fit, e.g. "+0.1/+0.3 mm" (physical only), else None
      source     "cad" is never used here: "bom" for explicit ``connects``, "inferred" otherwise
      from, to   direction of flow for electrical/mechanical links (power: battery -> ESC -> motor,
                 signal: FC -> servo, drives: motor -> prop, linkage: servo -> surface); for physical
                 links simply the two ids.
      declared_fit True when the pair interpenetrates and either part lists the other in
                 ``specs.fits`` (spar through a wing, servo in its pocket, ...); gap_mm stays as measured.
  * the edge also carries ``kind``, ``joint``, ``gap_mm``, ``tolerance``, ``source`` copied from
    ``links[0]`` (physical links are added first), for simple consumers.
  * links from ``interfaces()`` carry the same keys minus ``from``/``to``.
  * ``graph.graph["warnings"]``: inference problems (e.g. a motor with no free ESC).

Physical links: meshed parts whose bboxes (grown by 2 mm) overlap are compared with surface samples
(up to 2000 points of each part inside the shared box, KD-tree, then exact nearest-surface distance for
the closest candidates). Gap < 1 mm -> link. If a watertight partner contains sample points, the gap is
minus the deepest penetration. Joint type and tolerance come from the two part types (``joint_for``).

Electrical/mechanical links: explicit ``connects`` first (electrical when both parts are electrical
types, else mechanical). Then inference fills what is missing: motor -> nearest free ESC, ESC -> PDB
(or every battery), PDB -> every battery, FC -> servos/rx/gps/telemetry/vtx/camera/remote_id/sensor/ESC
(signal), prop <- the motor it touches or the nearest free motor, control surface <- nearest free servo.
"Free" means not yet linked to another part of the same kind, so one servo never silently drives two
surfaces (declare a shared servo in ``connects``) and removing a part never re-routes to a spare one.
"""
from __future__ import annotations

import re

import networkx as nx
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from dronebench.model import Design, Part

CONTACT_MM = 1.0
BOX_GROW_M = 2e-3
MAX_SAMPLES = 1000
EXACT_CANDIDATES = 16            # closest samples refined with exact point-to-surface distance
DEPTH_REFINE = 16                # deepest apparent penetrations confirmed and measured exactly
PAIR_CACHE_SIZE = 50_000
SAMPLE_SPACING_M = 3e-3
MIN_POOL = 300
MAX_POOL = 12_000
ELECTRICAL_TYPES = {"motor", "esc", "battery", "servo", "fc", "rx", "gps", "telemetry", "vtx", "camera",
                    "remote_id", "pdb", "wire", "connector", "sensor"}
FC_SIGNAL_TYPES = ("servo", "rx", "gps", "telemetry", "vtx", "camera", "remote_id", "sensor", "esc")
CONTROL_ROLES = ("aileron", "flap", "elevon", "elevator", "rudder", "ruddervator")
AIRFRAME_TYPES = {"wing", "fuselage", "tail", "mount", "surface", "hatch"}
# ISO 273 clearance holes (fine, medium) in mm
CLEARANCE_HOLES = {1.6: (1.7, 1.8), 2.0: (2.2, 2.4), 2.5: (2.7, 2.9), 3.0: (3.2, 3.4), 4.0: (4.3, 4.5),
                   5.0: (5.3, 5.5), 6.0: (6.4, 6.6), 8.0: (8.4, 9.0)}
_pair_cache: dict[tuple, float | None] = {}     # (mesh key, mesh key) -> gap_mm, shared across builds


# --------------------------------------------------------------------------- joints

def _is_printed(p: Part) -> bool:
    return p.material is not None and p.material.kind == "printed"


def _bolt_size(p: Part) -> float | None:
    m = re.search(r"M(\d+(?:\.\d+)?)", str(p.specs.get("size", "")) + " " + str(p.part_number or ""))
    return float(m.group(1)) if m else None


def joint_for(a: Part, b: Part) -> tuple[str, str]:
    """(joint type, tolerance) inferred from the two part types for a physical contact."""
    types = {a.type, b.type}

    def other(t: str) -> Part:
        return b if a.type == t else a

    if "fastener" in types:
        size = _bolt_size(a if a.type == "fastener" else b)
        if size is None:
            return "M? clearance hole", "clearance hole per ISO 273 (size unknown)"
        fine, medium = CLEARANCE_HOLES.get(size, (round(size * 1.07, 1), round(size * 1.13, 1)))
        return f"M{size:g} clearance hole", f"hole Ø{fine:g}-{medium:g} mm (ISO 273 fine-medium)"
    if types == {"prop", "motor"}:
        return "shaft/prop adapter", "H7/h7 (0 to +0.015 mm)"
    if "motor" in types and (other("motor").type in {"mount", "fuselage", "wing", "tail"}
                             or other("motor").role == "nacelle"):
        return "bolt pattern", "±0.1 mm"
    if "spar" in types and other("spar").type in {"wing", "fuselage", "tail", "mount"}:
        return "tube in printed bore, slip fit", "+0.1/+0.3 mm"
    for mover in ("surface", "hinge"):
        if mover in types and types != {mover} and other(mover).type in AIRFRAME_TYPES | {"hinge"}:
            return "hinge pin running fit", "H11/c11 (+0.1/+0.2 mm)"
    if "servo" in types and other("servo").type in {"wing", "fuselage", "tail", "mount"}:
        return "servo pocket", "+0.2/+0.5 mm"
    if types == {"battery", "fuselage"}:
        return "battery bay", ">= 2 mm clearance"
    if _is_printed(a) and _is_printed(b):
        return "printed interlock/glue joint", "+0.1/+0.2 mm"
    return "contact", "±0.2 mm"


# --------------------------------------------------------------------------- proximity

class _Cloud:
    """Surface samples of one part (about one per 3 mm, capped) with their face normals and a KD-tree."""

    def __init__(self, p: Part) -> None:
        n = int(np.clip(p.mesh.area / SAMPLE_SPACING_M ** 2, MIN_POOL, MAX_POOL))
        self.points, faces = trimesh.sample.sample_surface(p.mesh, n, seed=0)
        self.normals = p.mesh.face_normals[faces]
        self.tree = cKDTree(self.points)
        self.spacing = float(np.sqrt(p.mesh.area / n))
        self.solid = bool(p.mesh.is_watertight)

    def query(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Approximate (distance to surface, inside) per point, from the nearest sample and its normal."""
        if len(pts) == 0:
            return np.full(0, np.inf), np.zeros(0, dtype=bool)
        dist, idx = self.tree.query(pts)
        inside = np.einsum("ij,ij->i", pts - self.points[idx], self.normals[idx]) < 0
        return dist, inside & self.solid


def _in_box(pts: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return pts[((pts >= lo) & (pts <= hi)).all(axis=1)]


def _thin(pts: np.ndarray, k: int) -> np.ndarray:
    return pts if len(pts) <= k else pts[np.linspace(0, len(pts) - 1, k).astype(int)]


def _depth(pts: np.ndarray, approx: np.ndarray, into: Part) -> float:
    """Penetration depth (m) into ``into``: the deepest approximate-inside points, confirmed by ray test."""
    if len(pts) == 0:
        return 0.0
    top = pts[np.argsort(approx)[-DEPTH_REFINE:]]
    top = top[into.mesh.contains(top)]
    return float(into.mesh.nearest.on_surface(top)[1].max()) if len(top) else 0.0


def _exact_gap(pts: np.ndarray, approx: np.ndarray, onto: Part) -> float:
    if len(pts) == 0:
        return np.inf
    return float(onto.mesh.nearest.on_surface(pts[np.argsort(approx)[:EXACT_CANDIDATES]])[1].min())


def surface_gap_mm(a: Part, b: Part, clouds: dict[str, _Cloud] | None = None) -> float | None:
    """Min surface gap between two meshed parts in mm (negative = penetration depth), None if >= 1 mm apart.

    Both parts' samples inside the shared (grown) bbox are classified against the other's nearest
    sample and normal; the deepest apparent penetrations are confirmed with ``contains`` and measured
    exactly, otherwise the closest samples give the exact gap. Works for a body wholly inside another.
    """
    clouds = {} if clouds is None else clouds
    lo = np.maximum(a.bbox_m[0], b.bbox_m[0]) - BOX_GROW_M
    hi = np.minimum(a.bbox_m[1], b.bbox_m[1]) + BOX_GROW_M
    if (lo > hi).any():
        return None
    for p in (a, b):
        if p.id not in clouds:
            clouds[p.id] = _Cloud(p)
    ca, cb = clouds[a.id], clouds[b.id]
    qa, qb = _thin(_in_box(ca.points, lo, hi), MAX_SAMPLES), _thin(_in_box(cb.points, lo, hi), MAX_SAMPLES)
    da, ia = cb.query(qa)
    db, ib = ca.query(qb)
    depth = max(_depth(qa[ia], da[ia], b), _depth(qb[ib], db[ib], a))
    if depth > 0:
        return float(-depth * 1e3)
    near = CONTACT_MM * 1e-3 + 3 * max(ca.spacing, cb.spacing)
    if min(da.min(initial=np.inf), db.min(initial=np.inf)) > near:
        return None
    gap = min(_exact_gap(qa, da, b), _exact_gap(qb, db, a))
    return float(gap * 1e3) if gap < CONTACT_MM * 1e-3 else None


# --------------------------------------------------------------------------- graph building

def _link(g: nx.Graph, a: str, b: str, kind: str, joint: str, source: str,
          gap_mm: float | None = None, tolerance: str | None = None, declared_fit: bool = False) -> None:
    link = {"kind": kind, "joint": joint, "gap_mm": gap_mm, "tolerance": tolerance, "source": source,
            "from": a, "to": b, "declared_fit": declared_fit}
    if g.has_edge(a, b):
        g.edges[a, b]["links"].append(link)
    else:
        g.add_edge(a, b, links=[link], **{k: link[k] for k in ("kind", "joint", "gap_mm", "tolerance", "source")})


def _has(g: nx.Graph, a: str, b: str, kind: str) -> bool:
    return g.has_edge(a, b) and any(l["kind"] == kind for l in g.edges[a, b]["links"])


def _dist(a: Part, b: Part) -> float:
    if a.cg_m is None or b.cg_m is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(a.cg_m) - np.asarray(b.cg_m)))


def _fits(p: Part) -> set[str]:
    """Ids this part may legitimately sit inside (``specs.fits``, declared on the inner part)."""
    fits = p.specs.get("fits") or []
    return {str(fits)} if isinstance(fits, str) else {str(f) for f in fits}


def _mesh_key(mesh: trimesh.Trimesh) -> tuple:
    v = mesh.vertices
    return len(v), len(mesh.faces), round(float(v.sum()), 9), round(float(np.abs(v).sum()), 9)


def _add_physical(g: nx.Graph, design: Design) -> None:
    """Physical links for all bbox-overlapping mesh pairs. Gaps are cached per mesh pair across calls,
    so rebuilding the graph after removing or editing a few parts only measures the changed pairs."""
    meshed = [p for p in design.parts.values() if p.mesh is not None]
    if len(meshed) < 2:
        return
    boxes = np.array([p.bbox_m for p in meshed])
    lo, hi = boxes[:, 0] - BOX_GROW_M, boxes[:, 1] + BOX_GROW_M
    overlap = ((lo[:, None] <= hi[None]) & (lo[None] <= hi[:, None])).all(axis=2)
    keys = [_mesh_key(p.mesh) for p in meshed]
    clouds: dict[str, _Cloud] = {}
    for i, j in zip(*np.nonzero(np.triu(overlap, k=1))):
        a, b = meshed[i], meshed[j]
        key = tuple(sorted((keys[i], keys[j])))
        if key in _pair_cache:
            gap = _pair_cache[key]
        else:
            gap = _pair_cache[key] = surface_gap_mm(a, b, clouds)
            if len(_pair_cache) > PAIR_CACHE_SIZE:
                _pair_cache.pop(next(iter(_pair_cache)))
        if gap is not None:
            joint, tol = joint_for(a, b)
            declared = gap < 0 and (b.id in _fits(a) or a.id in _fits(b))
            _link(g, a.id, b.id, "physical", joint, "inferred", gap_mm=round(gap, 3), tolerance=tol,
                  declared_fit=declared)


def _signal_joint(dst_type: str) -> str:
    return "PWM signal" if dst_type in ("servo", "esc") else "serial"


def _connect_kind(a: Part, b: Part) -> tuple[str, str, str, str]:
    """(kind, joint, from, to) for an explicit ``connects`` pair, oriented along power/signal flow."""
    order = ("battery", "pdb", "fc", "esc", "motor", "prop", "servo")
    ia = order.index(a.type) if a.type in order else len(order)
    ib = order.index(b.type) if b.type in order else len(order)
    src, dst = (a, b) if ia <= ib else (b, a)
    types = {a.type, b.type}
    if types == {"motor", "prop"}:
        return "mechanical", "drives", src.id, dst.id
    if a.type in ELECTRICAL_TYPES and b.type in ELECTRICAL_TYPES:
        if src.type == "fc":
            joint = _signal_joint(dst.type)
        elif src.type in ("battery", "pdb"):
            joint = "power"
        else:
            joint = "motor phase wires" if types == {"esc", "motor"} else "wired"
        return "electrical", joint, src.id, dst.id
    if "servo" in types:
        return "mechanical", "pushrod linkage", src.id, dst.id
    return "mechanical", "mechanical link", src.id, dst.id


def _add_explicit(g: nx.Graph, design: Design) -> None:
    for p in design.parts.values():
        for other_id in p.connects:
            other = design.parts.get(other_id)
            if other is None:
                continue
            kind, joint, src, dst = _connect_kind(p, other)
            if not _has(g, src, dst, kind):
                _link(g, src, dst, kind, joint, "bom")


def _linked_types(g: nx.Graph, design: Design, pid: str, kind: str, *types: str) -> list[str]:
    return [n for n in linked(g, pid, kind) if design.parts[n].type in types]


def _assign_nearest(g: nx.Graph, users: list[Part], pool: list[Part], kind: str, joint: str) -> list[str]:
    """Link (pool -> user) each user that has no ``kind`` link to the pool yet with its nearest pool part
    not already linked to another user. Returns the ids left unlinked (pool exhausted)."""
    user_ids, pool_ids = {u.id for u in users}, {p.id for p in pool}
    taken = {p.id for p in pool if set(linked(g, p.id, kind)) & user_ids}
    unlinked = []
    for u in users:
        if set(linked(g, u.id, kind)) & pool_ids:
            continue
        free = [p for p in pool if p.id not in taken]
        if not free:
            unlinked.append(u.id)
            continue
        best = min(free, key=lambda p: _dist(u, p))
        taken.add(best.id)
        _link(g, best.id, u.id, kind, joint, "inferred")
    return unlinked


def _add_inferred(g: nx.Graph, design: Design) -> None:
    motors, escs = design.of_type("motor"), design.of_type("esc")
    batteries, pdbs, fcs = design.of_type("battery"), design.of_type("pdb"), design.of_type("fc")
    warnings = g.graph["warnings"]

    for mid in _assign_nearest(g, motors, escs, "electrical", "motor phase wires"):
        warnings.append(f"{mid}: no free ESC to drive it")
    for esc in escs:
        if not _linked_types(g, design, esc.id, "electrical", "battery", "pdb"):
            for src in pdbs or batteries:
                _link(g, src.id, esc.id, "electrical", "power", "inferred")
    for pdb in pdbs:
        if not _linked_types(g, design, pdb.id, "electrical", "battery"):
            for bat in batteries:
                _link(g, bat.id, pdb.id, "electrical", "power", "inferred")
    if fcs:
        for p in design.of_type(*FC_SIGNAL_TYPES):
            if not _linked_types(g, design, p.id, "electrical", "fc"):
                fc = min(fcs, key=lambda f: _dist(f, p))
                _link(g, fc.id, p.id, "electrical", _signal_joint(p.type), "inferred")

    props = design.of_type("prop")
    for prop in props:
        touching = _linked_types(g, design, prop.id, "physical", "motor")
        if touching and not _linked_types(g, design, prop.id, "mechanical", "motor"):
            _link(g, touching[0], prop.id, "mechanical", "drives", "inferred")
    for pid in _assign_nearest(g, props, motors, "mechanical", "drives"):
        warnings.append(f"{pid}: no free motor to drive it")

    surfaces = [p for p in design.parts.values() if p.role in CONTROL_ROLES]
    for sid in _assign_nearest(g, surfaces, design.of_type("servo"), "mechanical", "pushrod linkage"):
        warnings.append(f"{sid}: control surface with no free servo (declare shared servos in connects)")


def build_graph(design: Design) -> nx.Graph:
    """Part graph of ``design`` (see module docstring for the node/edge format)."""
    g = nx.Graph(warnings=[])
    for p in design.parts.values():
        g.add_node(p.id, type=p.type, role=p.role, group=p.group)
    _add_physical(g, design)
    _add_explicit(g, design)
    _add_inferred(g, design)
    for p in design.of_type("motor"):
        if not _linked_types(g, design, p.id, "mechanical", "prop"):
            g.graph["warnings"].append(f"{p.id}: no prop linked")
    return g


# --------------------------------------------------------------------------- queries

def interfaces(graph: nx.Graph, part_id: str) -> list[dict]:
    """Every link of ``part_id``: [{mates, kind, joint, gap_mm, tolerance, source, declared_fit}], physical first."""
    out = []
    for nb in graph.neighbors(part_id):
        for l in graph.edges[part_id, nb]["links"]:
            out.append({"mates": nb, "kind": l["kind"], "joint": l["joint"], "gap_mm": l["gap_mm"],
                        "tolerance": l["tolerance"], "source": l["source"], "declared_fit": l["declared_fit"]})
    rank = {"physical": 0, "mechanical": 1, "electrical": 2}
    return sorted(out, key=lambda d: (rank[d["kind"]], d["mates"]))


def linked(graph: nx.Graph, part_id: str, kind: str | None = None, type: str | None = None) -> list[str]:
    """Neighbour ids of ``part_id`` joined by a link of ``kind`` (any if None), optionally of node ``type``."""
    return sorted(nb for nb in graph.neighbors(part_id)
                  if (kind is None or any(l["kind"] == kind for l in graph.edges[part_id, nb]["links"]))
                  and (type is None or graph.nodes[nb]["type"] == type))


def electrical_neighbors(graph: nx.Graph, part_id: str, type: str | None = None) -> list[str]:
    """Ids wired to ``part_id`` (electrical links), optionally only those of part ``type``."""
    return linked(graph, part_id, "electrical", type)
