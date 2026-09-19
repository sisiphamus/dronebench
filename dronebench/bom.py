"""Read a design folder's ``bom.yaml`` and ``mission.yaml`` into a :class:`Design` (no meshes yet).

Rules:
  * ids must be unique (a duplicate raises ``ValueError``).
  * unknown ``type`` -> warning, type becomes ``"other"``; unknown ``role`` -> warning, role kept.
  * ``connects`` / ``parent`` / ``mission.payload_ids`` pointing at unknown ids -> warning.
  * parts with ``locked: true`` or listed in ``mission.payload_ids`` get ``locked=True``.
  * ``qty`` stays a mass multiplier; entries are never expanded.
  * every top-level field present in a BOM entry is tagged ``sources[field] = "bom"``.

An optional per-entry ``sources: {field: tag}`` map overrides those tags (e.g. ``agent_supplied`` for
spec fields an agent edited); keys are spec names (kv, mass_g, ...) or top-level fields (material, ...).

Materials: a ``material`` entry (dict or plain name) becomes a :class:`Material` with density 0 and empty
kind; if ``dronebench.materials`` is importable, ``materials.resolve_material(part)`` then fills it
(and infers a material for parts that give none).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dronebench.model import AERO_ROLES, PART_TYPES, Design, Frame, Material, Mission, Part

SPAR_ROLES = ("main_spar", "rear_spar", "tail_boom")
UNITS_TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}
AXES = ("+x", "-x", "+y", "-y", "+z", "-z")

_PART_FIELDS = ("id", "type", "part_number", "cad", "role", "material", "specs", "locked", "connects",
                "group", "parent", "qty", "notes", "sources")


def _read_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def materials_module():
    """Builder C's materials module, or None while it does not exist."""
    try:
        from dronebench import materials
    except ImportError:
        return None
    return materials


def _normalise_axis(value: Any, name: str) -> str:
    axis = str(value).strip().lower()
    if axis in ("x", "y", "z"):
        axis = "+" + axis
    if axis not in AXES:
        raise ValueError(f"mission.yaml frame.{name} must be one of {AXES}, got {value!r}")
    return axis


def _parse_frame(raw: dict | None) -> Frame:
    raw = raw or {}
    frame = Frame()
    units = str(raw.get("units", frame.units)).strip().lower()
    if units not in UNITS_TO_M:
        raise ValueError(f"mission.yaml frame.units must be one of {tuple(UNITS_TO_M)}, got {units!r}")
    frame.units = units
    frame.forward = _normalise_axis(raw.get("forward", frame.forward), "forward")
    frame.up = _normalise_axis(raw.get("up", frame.up), "up")
    if frame.forward[1] == frame.up[1]:
        raise ValueError(f"frame.forward ({frame.forward}) and frame.up ({frame.up}) must be different axes")
    return frame


def _parse_mission(raw: dict | None) -> Mission:
    raw = raw or {}
    mission = Mission()
    mission.preset = str(raw.get("preset", mission.preset))
    mission.payload_ids = [str(i) for i in raw.get("payload_ids") or []]
    if raw.get("cruise_kmh") is not None:
        mission.cruise_kmh = float(raw["cruise_kmh"])
    mission.rho = float(raw.get("rho", mission.rho))
    mission.usable_battery_fraction = float(raw.get("usable_battery_fraction", mission.usable_battery_fraction))
    return mission


def _parse_material(raw: Any) -> Material | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"name": raw}
    return Material(
        name=str(raw.get("name", "unknown")),
        density_kg_m3=float(raw.get("density_kg_m3", 0.0)),
        kind=str(raw.get("kind", "")),     # "" = let the catalog decide
        tensile_mpa=raw.get("tensile_mpa"),
        modulus_gpa=raw.get("modulus_gpa"),
        infill=None if raw.get("infill") is None else float(raw["infill"]),
        walls=None if raw.get("walls") is None else int(raw["walls"]),
        orientation=raw.get("orientation"),
        sub={str(k): str(v) for k, v in (raw.get("sub") or {}).items()},
        source="bom",
    )


def parse_part(entry: dict, warnings: list[str]) -> Part:
    """Build one :class:`Part` from a BOM entry dict (no cross-reference checks)."""
    if "id" not in entry:
        raise ValueError(f"BOM entry without id: {entry!r}")
    pid = str(entry["id"])
    unknown = sorted(set(entry) - set(_PART_FIELDS))
    if unknown:
        warnings.append(f"{pid}: unknown BOM fields ignored: {unknown}")
    ptype = str(entry.get("type", "other"))
    if ptype not in PART_TYPES:
        warnings.append(f"{pid}: unknown type {ptype!r}, treated as 'other'")
        ptype = "other"
    role = entry.get("role")
    if role is not None and role not in AERO_ROLES and role not in SPAR_ROLES:
        warnings.append(f"{pid}: unknown role {role!r}")
    qty = int(entry.get("qty", 1))
    if qty < 1:
        warnings.append(f"{pid}: qty {qty} < 1, set to 1")
        qty = 1
    part = Part(
        id=pid,
        type=ptype,
        part_number=None if entry.get("part_number") is None else str(entry["part_number"]),
        cad=None if entry.get("cad") is None else str(entry["cad"]),
        role=role,
        material=_parse_material(entry.get("material")),
        specs=dict(entry.get("specs") or {}),
        locked=bool(entry.get("locked", False)),
        connects=[str(c) for c in entry.get("connects") or []],
        group=None if entry.get("group") is None else str(entry["group"]),
        parent=None if entry.get("parent") is None else str(entry["parent"]),
        qty=qty,
        notes=entry.get("notes"),
    )
    part.sources = {k: "bom" for k in entry if k in _PART_FIELDS and k not in ("id", "sources")}
    part.sources.update({str(k): str(v) for k, v in (entry.get("sources") or {}).items()})
    return part


def resolve_material(part: Part) -> None:
    """Fill ``part.material`` via builder C's resolver, or leave a placeholder if that module is absent."""
    materials = materials_module()
    if materials is not None:
        part.material = materials.resolve_material(part)
    elif part.material is None:
        part.material = Material(name="unknown", density_kg_m3=0.0, kind="unknown", source="assumed")
    part.sources.setdefault("material", part.material.source)
    if part.sources["material"] == "bom":
        part.sources["material"] = part.material.source


def load_bom(design_dir: str | Path) -> Design:
    """Parse ``bom.yaml`` + ``mission.yaml`` of ``design_dir`` into a Design with mesh-less parts."""
    root = Path(design_dir)
    bom_path = root / "bom.yaml"
    if not bom_path.is_file():
        raise FileNotFoundError(f"no bom.yaml in {root}")
    bom = _read_yaml(bom_path) or {}
    mission_path = root / "mission.yaml"
    mission_raw = _read_yaml(mission_path) if mission_path.is_file() else {}
    mission_raw = mission_raw or {}

    design = Design(name=root.name, root=root, mission=_parse_mission(mission_raw),
                    frame=_parse_frame(mission_raw.get("frame")))
    if not mission_path.is_file():
        design.warnings.append("no mission.yaml: default mission and frame (mm, forward -x, up +z) assumed")

    for entry in bom.get("parts") or []:
        part = parse_part(entry, design.warnings)
        if part.id in design.parts:
            raise ValueError(f"duplicate part id {part.id!r} in bom.yaml")
        design.parts[part.id] = part

    for pid in design.mission.payload_ids:
        if pid in design.parts:
            design.parts[pid].locked = True
        else:
            design.warnings.append(f"mission payload id {pid!r} is not in the BOM")
    for part in design.parts.values():
        for other in part.connects:
            if other not in design.parts:
                design.warnings.append(f"{part.id}: connects to unknown id {other!r}")
        if part.parent is not None and part.parent not in design.parts:
            design.warnings.append(f"{part.id}: parent {part.parent!r} is not in the BOM")
        resolve_material(part)
    return design
