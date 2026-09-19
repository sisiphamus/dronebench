"""Apply JSON edit ops to a COPY of a design folder.

    apply_edits(design_dir, ops, out_dir) -> (out_dir, errors)

The design folder is copied (CAD files hard-linked when possible; an edited mesh is always written to
a fresh file, never through a link). Ops, applied in order:

  {"op": "remove", "id": "ballast_nose"}
  {"op": "swap", "id": "motor_L", "entry": {...bom fields...}, "cad_file": "path/to/new.stl"?}
      merges `entry` into the BOM entry (id kept). A new part_number without new specs clears the old
      specs (so the catalog fills them). Every field supplied is tagged `sources: {field: agent_supplied}`.
  {"op": "set", "id": "wing_L1", "path": "material.infill" | "specs.kv" | "part_number" | ..., "value": v}
  {"op": "scale", "ids": [...], "factors": [sx, sy, sz], "about": "part_cg" | "origin" | [x, y, z]}
      factors and the about point are in the DroneBench frame (x aft, y right, z up; metres). The mesh
      is transformed in its own CAD frame and written as STL in the design's units.
  {"op": "move", "ids": [...], "offset_m": [dx, dy, dz]}   (DroneBench frame; parts without CAD move
      their specs.cg_mm)
  {"op": "add", "entry": {...}, "cad_file": "path"?}
  {"op": "mission", "path": "cruise_kmh", "value": 60}

All ops are validated and staged in memory first; if any op fails, nothing is written and the
errors are returned.
"""
from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from .model import PART_TYPES

AGENT = "agent_supplied"
UNITS_TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}
METRE_EXTS = {".glb", ".gltf", ".step", ".stp"}
SET_ROOTS = {"specs", "material", "part_number", "qty", "role", "type", "connects", "group", "parent", "notes"}
MATERIAL_FIELDS = {"name", "infill", "walls", "orientation"}
MISSION_PATHS = {"preset", "cruise_kmh", "usable_battery_fraction", "rho"}
PRESETS = ("endurance", "range", "speed", "payload", "efficiency")
SCALE_RANGE = (0.2, 5.0)
MAX_MOVE_M = 2.0


def _rotation(frame: dict) -> np.ndarray:
    """p_ours = R @ p_cad (same convention as ingest.frame_rotation)."""
    try:
        from .bom import _parse_frame
        from .ingest import frame_rotation
        return frame_rotation(_parse_frame(frame))
    except Exception:
        def axis(s):
            s = str(s).strip().lower()
            s = s if s[0] in "+-" else "+" + s
            v = np.zeros(3)
            v["xyz".index(s[1])] = 1.0 if s[0] == "+" else -1.0
            return v
        aft, up = -axis(frame.get("forward", "-x")), axis(frame.get("up", "+z"))
        return np.vstack([aft, np.cross(up, aft), up])


def _load_mesh(path: Path):
    try:
        from .ingest import _load_scene, _select_mesh
        return _select_mesh(_load_scene(path), None)
    except Exception:
        import trimesh
        return trimesh.load(str(path), force="mesh")


def _mark(entry: dict, *fields: str) -> None:
    src = entry.setdefault("sources", {})
    for f in fields:
        src[str(f)] = AGENT


def _vec3(v, name) -> Optional[np.ndarray]:
    try:
        a = np.asarray(v, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return a if a.shape == (3,) and np.all(np.isfinite(a)) else None


class _Stage:
    def __init__(self, root: Path):
        self.root = root
        self.bom = yaml.safe_load((root / "bom.yaml").read_text()) or {}
        mp = root / "mission.yaml"
        self.mission = (yaml.safe_load(mp.read_text()) or {}) if mp.exists() else {}
        self.frame = self.mission.get("frame") or {}
        self.units = str(self.frame.get("units", "mm")).lower()
        self.R = _rotation(self.frame)
        self.parts: list[dict] = self.bom.setdefault("parts", [])
        self.meshes: dict[str, Any] = {}          # part id -> mesh in its CAD-file coords (edited)
        self.mesh_unit: dict[str, float] = {}     # part id -> metres per CAD unit of that mesh
        self.new_files: dict[str, Path] = {}      # cad name -> source file to copy in
        self.drop_files: set[str] = set()

    def entry(self, pid) -> Optional[dict]:
        return next((e for e in self.parts if str(e.get("id")) == str(pid)), None)

    def cad_users(self, cad: str) -> list[str]:
        f = cad.partition("#")[0]
        return [str(e["id"]) for e in self.parts if e.get("cad") and str(e["cad"]).partition("#")[0] == f]

    def mesh(self, e: dict):
        pid = str(e["id"])
        if pid not in self.meshes:
            cad = str(e["cad"])
            fname, _, body = cad.partition("#")
            if body:
                raise ValueError(f"{pid}: CAD is one body of a multi-body file ({cad}); cannot edit its mesh")
            others = [u for u in self.cad_users(cad) if u != pid]
            if others:
                raise ValueError(f"{pid}: CAD file {fname} is shared with {others}; cannot edit one part's mesh")
            src = self.new_files.get(fname) or self.root / "cad" / fname
            if not Path(src).is_file():
                raise ValueError(f"{pid}: CAD file {fname} not found")
            self.meshes[pid] = _load_mesh(Path(src)).copy()
            ext = Path(fname).suffix.lower()
            self.mesh_unit[pid] = 1.0 if ext in METRE_EXTS else UNITS_TO_M.get(self.units, 1e-3)
        return self.meshes[pid]


# ---------------------------------------------------------------- ops

def _op_remove(st: _Stage, op: dict) -> None:
    pid = op.get("id")
    e = st.entry(pid)
    if e is None:
        raise ValueError(f"remove: no part {pid!r}")
    st.parts.remove(e)
    st.meshes.pop(str(pid), None)
    if e.get("cad") and not st.cad_users(str(e["cad"])):
        st.drop_files.add(str(e["cad"]).partition("#")[0])
    for o in st.parts:
        if pid in (o.get("connects") or []):
            o["connects"] = [c for c in o["connects"] if c != pid]
        if o.get("parent") == pid:
            del o["parent"]


def _add_cad(st: _Stage, pid: str, cad_file) -> str:
    src = Path(cad_file)
    if not src.is_file():
        raise ValueError(f"{pid}: cad_file {cad_file} not found")
    name = src.name
    if (st.root / "cad" / name).exists() or name in st.new_files:
        name = f"{pid}_{src.name}"
    st.new_files[name] = src
    return name


def _op_swap(st: _Stage, op: dict) -> None:
    pid = op.get("id")
    e = st.entry(pid)
    if e is None:
        raise ValueError(f"swap: no part {pid!r}")
    new = op.get("entry")
    if not isinstance(new, dict):
        raise ValueError(f"swap {pid}: 'entry' must be a dict of BOM fields")
    if "id" in new and str(new["id"]) != str(pid):
        raise ValueError(f"swap {pid}: entry id {new['id']!r} differs (use remove + add to rename)")
    if "type" in new and new["type"] not in PART_TYPES:
        raise ValueError(f"swap {pid}: unknown type {new['type']!r}")
    if "locked" in new:
        raise ValueError(f"swap {pid}: 'locked' cannot be changed")
    if new.get("part_number") not in (None, e.get("part_number")) and "specs" not in new:
        e["specs"] = {}
        for k in list((e.get("sources") or {})):
            if k not in ("part_number", "material"):
                e["sources"].pop(k, None)
    for k, v in new.items():
        if k in ("id", "sources"):
            continue
        e[k] = copy.deepcopy(v)
        if k == "specs":
            _mark(e, *(v or {}).keys())
        else:
            _mark(e, k)
    if op.get("cad_file"):
        old = e.get("cad")
        e["cad"] = _add_cad(st, str(pid), op["cad_file"])
        st.meshes.pop(str(pid), None)
        if old and not st.cad_users(str(old)):
            st.drop_files.add(str(old).partition("#")[0])
        _mark(e, "cad")


def _op_set(st: _Stage, op: dict) -> None:
    pid, path = op.get("id"), op.get("path")
    e = st.entry(pid)
    if e is None:
        raise ValueError(f"set: no part {pid!r}")
    if not isinstance(path, str) or not path:
        raise ValueError(f"set {pid}: 'path' must be a string like 'specs.kv'")
    if "value" not in op:
        raise ValueError(f"set {pid}.{path}: missing 'value'")
    keys = path.split(".")
    root, value = keys[0], copy.deepcopy(op["value"])
    if root not in SET_ROOTS:
        raise ValueError(f"set {pid}: path {path!r} not editable (roots: {', '.join(sorted(SET_ROOTS))})")
    if root == "type" and value not in PART_TYPES:
        raise ValueError(f"set {pid}: unknown type {value!r}")
    if root == "qty" and (not isinstance(value, int) or value < 1):
        raise ValueError(f"set {pid}: qty must be an integer >= 1")
    if root == "material":
        if len(keys) == 1:
            e["material"] = value
        elif len(keys) == 2 and keys[1] in MATERIAL_FIELDS:
            if keys[1] == "infill" and not (isinstance(value, (int, float)) and 0 <= value <= 1):
                raise ValueError(f"set {pid}: material.infill must be 0-1")
            if keys[1] == "walls" and not (isinstance(value, int) and 0 <= value <= 20):
                raise ValueError(f"set {pid}: material.walls must be an integer 0-20")
            m = e.get("material")
            m = {"name": m} if isinstance(m, str) else dict(m or {})
            m[keys[1]] = value
            e["material"] = m
        else:
            raise ValueError(f"set {pid}: material path must be material.{{{','.join(sorted(MATERIAL_FIELDS))}}}")
        _mark(e, "material")
        return
    if root == "specs":
        if len(keys) < 2:
            if not isinstance(value, dict):
                raise ValueError(f"set {pid}: specs must be a dict")
            e["specs"] = value
            _mark(e, *value.keys())
            return
        d = e.setdefault("specs", {}) or {}
        e["specs"] = d
        for k in keys[1:-1]:
            d = d.setdefault(k, {})
            if not isinstance(d, dict):
                raise ValueError(f"set {pid}: {path} goes through a non-dict")
        if value is None:
            d.pop(keys[-1], None)
        else:
            d[keys[-1]] = value
        _mark(e, keys[1])
        return
    if len(keys) != 1:
        raise ValueError(f"set {pid}: {root} has no sub-fields")
    if root == "connects" and not isinstance(value, list):
        raise ValueError(f"set {pid}: connects must be a list of ids")
    e[root] = value
    _mark(e, root)


def _about_file(st: _Stage, e: dict, mesh, about) -> np.ndarray:
    u = st.mesh_unit[str(e["id"])]
    if about in (None, "part_cg"):
        return np.asarray(mesh.center_mass if mesh.is_watertight else mesh.bounds.mean(axis=0), float)
    if about == "origin":
        return np.zeros(3)
    p = _vec3(about, "about")
    if p is None:
        raise ValueError(f"scale: 'about' must be 'part_cg', 'origin' or [x, y, z] metres")
    return st.R.T @ p / u


def _op_scale(st: _Stage, op: dict) -> None:
    ids, f = op.get("ids"), _vec3(op.get("factors"), "factors")
    if not isinstance(ids, list) or not ids:
        raise ValueError("scale: 'ids' must be a non-empty list")
    if f is None or np.any(f < SCALE_RANGE[0]) or np.any(f > SCALE_RANGE[1]):
        raise ValueError(f"scale: 'factors' must be 3 numbers in {SCALE_RANGE[0]}-{SCALE_RANGE[1]}")
    S = st.R.T @ np.diag(f) @ st.R   # our-frame scale expressed in CAD axes (R is a signed permutation)
    for pid in ids:
        e = st.entry(pid)
        if e is None:
            raise ValueError(f"scale: no part {pid!r}")
        if not e.get("cad"):
            raise ValueError(f"scale: {pid} has no CAD mesh")
        mesh = st.mesh(e)
        c = _about_file(st, e, mesh, op.get("about"))
        T = np.eye(4)
        T[:3, :3] = S
        T[:3, 3] = c - S @ c
        mesh.apply_transform(T)
        _mark(e, "cad")
        # printed / tube dimensions in specs no longer match the mesh; spar diameters scale with it
        specs = e.get("specs") or {}
        for key in ("outer_d_mm", "inner_d_mm"):
            if key in specs:
                ax = int(np.argmax(np.abs(st.R @ mesh.extents)))
                cross = [i for i in range(3) if i != ax]
                specs[key] = round(float(specs[key]) * float(np.mean(f[cross])), 3)
                _mark(e, key)


def _op_move(st: _Stage, op: dict) -> None:
    ids, d = op.get("ids"), _vec3(op.get("offset_m"), "offset_m")
    if not isinstance(ids, list) or not ids:
        raise ValueError("move: 'ids' must be a non-empty list")
    if d is None or np.linalg.norm(d) > MAX_MOVE_M:
        raise ValueError(f"move: 'offset_m' must be 3 numbers (metres), |offset| <= {MAX_MOVE_M}")
    for pid in ids:
        e = st.entry(pid)
        if e is None:
            raise ValueError(f"move: no part {pid!r}")
        if e.get("cad"):
            mesh = st.mesh(e)
            mesh.apply_translation(st.R.T @ d / st.mesh_unit[str(pid)])
            _mark(e, "cad")
        else:
            specs = e.setdefault("specs", {}) or {}
            e["specs"] = specs
            if specs.get("cg_mm") is None:
                raise ValueError(f"move: {pid} has no CAD and no specs.cg_mm to move (set specs.cg_mm first)")
            cg = _vec3(specs["cg_mm"], "cg_mm")
            if cg is None:
                raise ValueError(f"move: {pid} specs.cg_mm is not [x, y, z]")
            specs["cg_mm"] = [round(float(v), 3) for v in cg + st.R.T @ d * 1000]
            _mark(e, "cg_mm")


def _op_add(st: _Stage, op: dict) -> None:
    new = op.get("entry")
    if not isinstance(new, dict) or not new.get("id"):
        raise ValueError("add: 'entry' must be a dict with an 'id'")
    pid = str(new["id"])
    if st.entry(pid) is not None:
        raise ValueError(f"add: id {pid!r} already exists")
    if new.get("type") not in PART_TYPES:
        raise ValueError(f"add {pid}: 'type' must be one of PART_TYPES, got {new.get('type')!r}")
    if new.get("locked"):
        raise ValueError(f"add {pid}: agents cannot add locked parts")
    e = copy.deepcopy(new)
    e.pop("sources", None)
    _mark(e, *(k for k in e if k not in ("id", "specs")), *(e.get("specs") or {}).keys())
    if op.get("cad_file"):
        e["cad"] = _add_cad(st, pid, op["cad_file"])
    elif e.get("cad") and not (st.root / "cad" / str(e["cad"]).partition("#")[0]).is_file():
        raise ValueError(f"add {pid}: cad {e['cad']!r} not in cad/ (pass cad_file)")
    st.parts.append(e)


def _op_mission(st: _Stage, op: dict) -> None:
    path, value = op.get("path"), op.get("value")
    if path not in MISSION_PATHS:
        raise ValueError(f"mission: path must be one of {sorted(MISSION_PATHS)} (payload and frame are fixed)")
    if path == "preset" and value not in PRESETS:
        raise ValueError(f"mission: preset must be one of {PRESETS}")
    if path != "preset" and value is not None and not isinstance(value, (int, float)):
        raise ValueError(f"mission: {path} must be a number")
    st.mission[path] = value


OPS = {"remove": _op_remove, "swap": _op_swap, "set": _op_set, "scale": _op_scale, "move": _op_move,
       "add": _op_add, "mission": _op_mission}


# ---------------------------------------------------------------- write

def _copy_tree(src: Path, dst: Path) -> None:
    def link_or_copy(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    shutil.copytree(src, dst, copy_function=link_or_copy,
                    ignore=shutil.ignore_patterns("__pycache__", "report.json", "viewer.html"))


def _write_file_fresh(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()   # never write through a hard link into the source design
    path.write_bytes(data)


def _prepare_out(design_dir: Path, out_dir: Path) -> None:
    if out_dir.resolve() == design_dir.resolve():
        raise ValueError("out_dir must differ from design_dir (edits never modify the source design)")
    if design_dir.resolve() in out_dir.resolve().parents:
        raise ValueError("out_dir must not be inside design_dir")
    if out_dir.exists():
        if (out_dir / "bom.yaml").is_file():
            shutil.rmtree(out_dir)
        elif any(out_dir.iterdir()):
            raise ValueError(f"out_dir {out_dir} exists, is not empty and is not a design folder")
        else:
            out_dir.rmdir()


def apply_edits(design_dir, ops: list, out_dir) -> tuple[Path, list[str]]:
    design_dir, out_dir = Path(design_dir), Path(out_dir)
    errors: list[str] = []
    if not (design_dir / "bom.yaml").is_file():
        return out_dir, [f"no bom.yaml in {design_dir}"]
    if isinstance(ops, dict):
        ops = [ops]
    if not isinstance(ops, list):
        return out_dir, ["ops must be a list of op dicts"]
    try:
        st = _Stage(design_dir)
    except Exception as e:
        return out_dir, [f"cannot read design: {e}"]
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or op.get("op") not in OPS:
            errors.append(f"op {i}: unknown op {op.get('op') if isinstance(op, dict) else op!r}; "
                          f"one of {', '.join(OPS)}")
            continue
        try:
            OPS[op["op"]](st, op)
        except ValueError as e:
            errors.append(f"op {i} ({op['op']}): {e}")
        except Exception as e:
            errors.append(f"op {i} ({op['op']}): {type(e).__name__}: {e}")
    if errors:
        return out_dir, errors
    try:
        _prepare_out(design_dir, out_dir)
        _copy_tree(design_dir, out_dir)
        cad = out_dir / "cad"
        cad.mkdir(exist_ok=True)
        for name in st.drop_files:
            if (cad / name).exists():
                (cad / name).unlink()
        for name, src in st.new_files.items():
            _write_file_fresh(cad / name, Path(src).read_bytes())
        out_unit = UNITS_TO_M.get(st.units, 1e-3)
        for pid, mesh in st.meshes.items():
            e = st.entry(pid)
            if e is None:
                continue
            old = str(e["cad"])
            stem = Path(old).stem
            name = old if Path(old).suffix.lower() == ".stl" else f"{stem}.stl"
            if name != old and (cad / name).exists() and name not in (old,):
                name = f"{stem}_{pid}.stl"
            m = mesh.copy()
            m.apply_scale(st.mesh_unit[pid] / out_unit)   # STL is read in the design's units
            if name != old and (cad / old).exists() and not st.cad_users(old)[1:]:
                (cad / old).unlink()
            _write_file_fresh(cad / name, m.export(file_type="stl"))
            e["cad"] = name
        _write_file_fresh(out_dir / "bom.yaml", yaml.safe_dump(st.bom, sort_keys=False, allow_unicode=True).encode())
        if st.mission or (design_dir / "mission.yaml").exists():
            _write_file_fresh(out_dir / "mission.yaml",
                              yaml.safe_dump(st.mission, sort_keys=False, allow_unicode=True).encode())
    except Exception as e:
        return out_dir, [f"write failed: {type(e).__name__}: {e}"]
    return out_dir, []
