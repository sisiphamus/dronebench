"""Load a design folder: BOM + every CAD body, in metres and the DroneBench frame (x aft, y right, z up).

CAD loading
  * STL/OBJ/PLY/3MF are read in ``frame.units``; STEP/STP (via cascadio) and GLB/GLTF are metres by
    their spec, so only the rotation applies to them. A body larger than 20 m after conversion is assumed
    to be millimetres and divided by 1000 (with a warning).
  * ``cad: "file.step#Name"`` selects the scene nodes whose node name, geometry name or ancestor node
    name equals ``Name``; without ``#`` the whole file is concatenated. A file is read once per ingest.
  * Rotation: x_aft = -forward, z = up, y = z x x (right-handed, y = right wing).

Per meshed part: volume_m3 (|volume| if watertight, else 0.7 x convex-hull volume + warning), area_m2,
bbox_m, cg_m (centre of mass if watertight, else bbox centre), mesh. Mass comes from
``materials.part_mass_g`` when that module exists, else ``specs.mass_g`` (bom), else volume x 1240
kg/m^3 (PLA, assumed).

Parts without CAD: cg from ``specs.cg_mm`` (a point in the CAD frame, millimetres, rotated like the
meshes), else the fuselage centroid (bbox centre of fuselage/hatch parts, falling back to the mean cg of
all meshed parts), tagged ``assumed``.

CAD files in ``cad/`` that no BOM entry references become auto-identified parts (confidence 0.4-0.8,
sources "inferred"): type/role from filename keywords, cross-checked with shape features.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import trimesh

from dronebench.bom import UNITS_TO_M, materials_module, load_bom, resolve_material
from dronebench.model import Design, Frame, Part

MESH_EXTS = {".stl", ".obj", ".ply", ".glb", ".gltf", ".3mf", ".step", ".stp"}
METRE_EXTS = {".glb", ".gltf", ".step", ".stp"}
DEFAULT_DENSITY_KG_M3 = 1240.0      # PLA
HULL_FILL = 0.7                     # solid fraction assumed inside the hull of a non-watertight mesh
MAX_PART_EXTENT_M = 20.0
FLAT_SURFACE_TYPES = {"wing", "tail", "surface"}   # one shape class: a thin lifting/control surface

# (filename keyword, type, role). Keywords of >= 4 letters match token prefixes, shorter ones whole tokens.
# Order matters: control surfaces before wing/tail, since "wing_aileron" should be an aileron.
KEYWORDS: tuple[tuple[str, str, str | None], ...] = (
    ("ruddervator", "surface", "ruddervator"), ("aileron", "surface", "aileron"), ("elevon", "surface", "elevon"),
    ("elev", "surface", "elevator"), ("rudder", "surface", "rudder"), ("flap", "surface", "flap"),
    ("prop", "prop", None), ("motor", "motor", None), ("servo", "servo", None), ("batt", "battery", None),
    ("spar", "spar", "main_spar"), ("boom", "spar", "tail_boom"), ("tube", "spar", "main_spar"),
    ("esc", "esc", None), ("gps", "gps", None), ("rx", "rx", None), ("receiver", "rx", None),
    ("fc", "fc", None), ("autopilot", "fc", None), ("camera", "camera", None), ("cam", "camera", None),
    ("hinge", "hinge", None), ("horn", "horn", None), ("screw", "fastener", None), ("bolt", "fastener", None),
    ("nacelle", "mount", "nacelle"), ("mount", "mount", None), ("hatch", "hatch", None),
    ("canopy", "hatch", None), ("fairing", "fairing", None), ("ballast", "ballast", None),
    ("vtail", "tail", "vtail"), ("fin", "tail", "vtail"), ("vstab", "tail", "vtail"),
    ("htail", "tail", "htail"), ("hstab", "tail", "htail"), ("stab", "tail", "htail"), ("tail", "tail", "htail"),
    ("wing", "wing", "wing"), ("fuse", "fuselage", "fuselage"), ("body", "fuselage", "fuselage"),
    ("pod", "fuselage", "fuselage"),
)


def frame_rotation(frame: Frame) -> np.ndarray:
    """3x3 matrix taking CAD-frame vectors to (x aft, y right, z up): p_ours = R @ p_cad."""
    def axis(spec: str) -> np.ndarray:
        v = np.zeros(3)
        v["xyz".index(spec[1])] = 1.0 if spec[0] == "+" else -1.0
        return v

    aft, up = -axis(frame.forward), axis(frame.up)
    return np.vstack([aft, np.cross(up, aft), up])


def frame_transform(frame: Frame, ext: str) -> np.ndarray:
    """4x4 homogeneous transform (rotation + unit scale) for a CAD file with extension ``ext``."""
    scale = 1.0 if ext.lower() in METRE_EXTS else UNITS_TO_M[frame.units]
    T = np.eye(4)
    T[:3, :3] = frame_rotation(frame) * scale
    return T


def _load_scene(path: Path) -> trimesh.Scene:
    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Trimesh):
        return trimesh.Scene(loaded)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    raise ValueError(f"{path.name}: no triangle mesh (got {type(loaded).__name__})")


def _select_mesh(scene: trimesh.Scene, body: str | None) -> trimesh.Trimesh:
    """Concatenate the scene's meshes (world transforms applied), optionally only those under ``body``."""
    parents = scene.graph.transforms.parents
    meshes = []
    for node in scene.graph.nodes_geometry:
        T, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        if body is not None:
            names, n = {geom_name, node}, node
            while n in parents:
                n = parents[n]
                names.add(n)
            if body not in names:
                continue
        meshes.append(geom.copy().apply_transform(T))
    if not meshes:
        what = f"body {body!r}" if body else "any triangle mesh"
        raise ValueError(f"CAD file has no {what}; nodes: {sorted(scene.graph.nodes_geometry)}")
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


class _CadCache:
    """Loads each CAD file once per ingest."""

    def __init__(self, cad_dir: Path):
        self.cad_dir = cad_dir
        self.scenes: dict[str, trimesh.Scene] = {}

    def mesh(self, ref: str, frame: Frame, warnings: list[str], label: str) -> trimesh.Trimesh:
        fname, _, body = ref.partition("#")
        path = self.cad_dir / fname
        if not path.is_file():
            raise FileNotFoundError(f"{label}: CAD file {fname!r} not found in {self.cad_dir}")
        if fname not in self.scenes:
            self.scenes[fname] = _load_scene(path)
        mesh = _select_mesh(self.scenes[fname], body or None)
        mesh.apply_transform(frame_transform(frame, path.suffix))
        if mesh.extents.max() > MAX_PART_EXTENT_M:
            mesh.apply_scale(1e-3)
            warnings.append(f"{label}: {fname} is > {MAX_PART_EXTENT_M:.0f} m after unit conversion; "
                            "assumed millimetres")
        return mesh


def _measure(part: Part, mesh: trimesh.Trimesh, warnings: list[str]) -> None:
    part.mesh = mesh
    part.area_m2 = float(mesh.area)
    part.bbox_m = np.array(mesh.bounds, dtype=float)
    if mesh.is_watertight and mesh.is_winding_consistent:
        part.volume_m3 = float(abs(mesh.volume))
        part.cg_m = np.array(mesh.center_mass, dtype=float)
        part.sources.update(volume_m3="cad", cg_m="cad")
    else:
        part.volume_m3 = float(mesh.convex_hull.volume) * HULL_FILL
        part.cg_m = part.bbox_m.mean(axis=0)
        part.sources.update(volume_m3="assumed", cg_m="computed")
        warnings.append(f"{part.id}: mesh not watertight; volume = {HULL_FILL} x convex hull, cg = bbox centre")
    part.sources.update(area_m2="cad", bbox_m="cad")


def _assign_mass(part: Part, warnings: list[str]) -> None:
    """Set mass_g; a BOM-declared specs.mass_g keeps its BOM tag (e.g. agent_supplied) instead of "bom"."""
    declared = part.sources.get("mass_g")
    materials = materials_module()
    if materials is not None:
        grams, source = materials.part_mass_g(part)
        part.mass_g, part.sources["mass_g"] = float(grams), source
    elif part.specs.get("mass_g") is not None:
        part.mass_g, part.sources["mass_g"] = float(part.specs["mass_g"]), "bom"
    elif part.volume_m3 is not None:
        part.mass_g, part.sources["mass_g"] = part.volume_m3 * DEFAULT_DENSITY_KG_M3 * 1e3, "assumed"
    else:
        part.mass_g, part.sources["mass_g"] = 0.0, "assumed"
        warnings.append(f"{part.id}: no CAD and no specs.mass_g; mass set to 0 g")
    if declared and part.sources["mass_g"] == "bom":
        part.sources["mass_g"] = declared


def _fuselage_centroid(design: Design) -> np.ndarray:
    fus = [p for p in design.parts.values()
           if p.mesh is not None and (p.type in ("fuselage", "hatch") or p.role == "fuselage")]
    if fus:
        bounds = np.vstack([p.bbox_m for p in fus])
        return (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    cgs = [p.cg_m for p in design.parts.values() if p.mesh is not None]
    return np.mean(cgs, axis=0) if cgs else np.zeros(3)


def _tokens(stem: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    return [re.sub(r"\d+", "", t) for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def _keyword_guess(stem: str) -> tuple[str, str | None] | None:
    tokens = _tokens(stem)
    for kw, ptype, role in KEYWORDS:
        if any((t.startswith(kw) if len(kw) >= 4 else t == kw) for t in tokens):
            return ptype, role
    return None


def _shape_guess(mesh: trimesh.Trimesh) -> tuple[str, str | None] | None:
    """Coarse type from bbox proportions in the normalised frame (x aft, y right, z up)."""
    ext = np.maximum(mesh.extents, 1e-9)
    order = np.argsort(ext)[::-1]           # long, mid, short axis indices
    a, b, c = ext[order]
    if a / b > 8 and b / c < 1.4:
        return "spar", "main_spar"
    if order[0] != 0 and a < 0.6 and a / b > 4 and b / c < 4:
        return "prop", None                  # blade: long axis in the y-z disc plane, not flat like a tail
    if a / c > 8 and b / c > 3:
        if order[2] == 1:
            return "tail", "vtail"           # thin along y: a vertical surface
        return ("wing", "wing") if a > 0.6 else ("tail", "htail")
    if order[0] == 0 and 3 < a / b < 15 and b / c < 2:
        return "fuselage", "fuselage"
    return None


def identify(stem: str, mesh: trimesh.Trimesh) -> tuple[str, str | None, float]:
    """Guess (type, role, confidence) for an unreferenced CAD body from its filename and shape."""
    by_name, by_shape = _keyword_guess(stem), _shape_guess(mesh)
    if by_name and by_shape:
        agree = by_name[0] == by_shape[0] or {by_name[0], by_shape[0]} <= FLAT_SURFACE_TYPES
        return (*by_name, 0.8 if agree else 0.6)
    if by_name:
        return (*by_name, 0.7)
    if by_shape:
        return (*by_shape, 0.5)
    return "other", None, 0.4


def _unreferenced_files(design: Design, cad_dir: Path) -> list[Path]:
    if not cad_dir.is_dir():
        return []
    referenced = {p.cad.partition("#")[0] for p in design.parts.values() if p.cad}
    return sorted(f for f in cad_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in MESH_EXTS and f.name not in referenced)


def _unique_id(design: Design, stem: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_") or "part"
    pid, n = base, 2
    while pid in design.parts:
        pid, n = f"{base}_{n}", n + 1
    return pid


def load_design(design_dir: str | Path) -> Design:
    """Load BOM + mission and every CAD body; returns a Design with measured, massed parts."""
    design = load_bom(design_dir)
    cad_dir = Path(design.root) / "cad"
    cache = _CadCache(cad_dir)

    for part in design.parts.values():
        if part.cad:
            _measure(part, cache.mesh(part.cad, design.frame, design.warnings, part.id), design.warnings)

    for path in _unreferenced_files(design, cad_dir):
        mesh = cache.mesh(path.name, design.frame, design.warnings, path.name)
        ptype, role, confidence = identify(path.stem, mesh)
        part = Part(id=_unique_id(design, path.stem), type=ptype, role=role, cad=path.name, confidence=confidence)
        part.sources.update(type="inferred", role="inferred", cad="inferred")
        _measure(part, mesh, design.warnings)
        resolve_material(part)
        design.parts[part.id] = part
        design.warnings.append(f"{part.id}: {path.name} is not in the BOM; identified as type {ptype!r}, "
                               f"role {role!r} (confidence {confidence:.1f})")

    rot = frame_rotation(design.frame)
    centroid = _fuselage_centroid(design)
    for part in design.parts.values():
        if part.mesh is None:
            if part.specs.get("cg_mm") is not None:
                part.cg_m = rot @ (np.asarray(part.specs["cg_mm"], dtype=float) * 1e-3)
                part.sources["cg_m"] = "bom"
            else:
                part.cg_m = centroid.copy()
                part.sources["cg_m"] = "assumed"
        _assign_mass(part, design.warnings)
    return design
