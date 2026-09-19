"""Tiny synthetic design folders, written at test time into a tmp dir.

``write_design`` exports meshes given in the DroneBench frame (x aft, y right, z up, metres) into the
CAD frame/units named by ``frame``, so ingest has to undo exactly that conversion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import yaml

from dronebench.bom import UNITS_TO_M
from dronebench.geometry import make_fuselage, make_naca4_fin, make_naca4_wing, make_tube
from dronebench.ingest import frame_rotation
from dronebench.model import Frame


def to_cad(mesh: trimesh.Trimesh, frame: Frame, metres: bool = False) -> trimesh.Trimesh:
    """Inverse of ingest's normalisation: DroneBench-frame metres -> CAD frame in ``frame.units``."""
    T = np.eye(4)
    T[:3, :3] = frame_rotation(frame).T / (1.0 if metres else UNITS_TO_M[frame.units])
    return mesh.copy().apply_transform(T)


def write_design(root: Path, parts: list[tuple[dict, trimesh.Trimesh | None]], frame: Frame | None = None,
                 mission: dict | None = None, extra_cad: dict[str, trimesh.Trimesh] | None = None) -> Path:
    """Write bom.yaml, mission.yaml and one STL per meshed entry (entry['cad'] defaults to '<id>.stl')."""
    frame = frame or Frame()
    (root / "cad").mkdir(parents=True, exist_ok=True)
    entries = []
    for entry, mesh in parts:
        entry = dict(entry)
        if mesh is not None:
            entry.setdefault("cad", f"{entry['id']}.stl")
            to_cad(mesh, frame).export(root / "cad" / entry["cad"])
        entries.append(entry)
    for name, mesh in (extra_cad or {}).items():
        to_cad(mesh, frame).export(root / "cad" / name)
    mission = dict(mission or {})
    mission["frame"] = {"units": frame.units, "forward": frame.forward, "up": frame.up}
    (root / "bom.yaml").write_text(yaml.safe_dump({"parts": entries}, sort_keys=False))
    (root / "mission.yaml").write_text(yaml.safe_dump(mission, sort_keys=False))
    return root


def mini_trainer(root: Path, frame: Frame | None = None) -> Path:
    """1 m NACA2412 trainer: wing, tail, fuselage, spar, motor + prop, and CAD-less electronics."""
    motor = trimesh.creation.cylinder(radius=0.014, height=0.03)
    motor.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    motor.apply_translation([-0.015, 0, 0])
    prop = trimesh.creation.box((0.004, 0.25, 0.02))
    prop.apply_translation([-0.032, 0, 0])
    parts = [
        ({"id": "fuselage", "type": "fuselage", "role": "fuselage", "material": {"name": "PLA", "infill": 0.1}},
         make_fuselage(0.9, 0.1)),
        ({"id": "wing", "type": "wing", "role": "wing", "material": {"name": "LW-PLA", "walls": 1}},
         make_naca4_wing(1.0, 0.2, 0.2, "2412", x_le=0.25, z=0.06)),
        ({"id": "htail", "type": "tail", "role": "htail"}, make_naca4_wing(0.36, 0.1, 0.1, "0010", x_le=0.78)),
        ({"id": "vtail", "type": "tail", "role": "vtail"}, make_naca4_fin(0.12, 0.1, 0.06, x_le=0.78, z0=0.02)),
        ({"id": "spar", "type": "spar", "role": "main_spar", "specs": {"od_mm": 8, "id_mm": 6, "fits": ["wing"]}},
         make_tube(0.008, 0.006, 0.9, "y", center=(0.3, 0.0, 0.066))),
        ({"id": "motor", "type": "motor", "specs": {"kv": 1000, "mass_g": 60}, "connects": ["esc"]}, motor),
        ({"id": "prop", "type": "prop", "specs": {"diam_in": 9, "pitch_in": 6, "mass_g": 10}}, prop),
        ({"id": "esc", "type": "esc", "specs": {"imax_a": 30, "mass_g": 25, "cg_mm": [-100, 0, 0]}}, None),
        ({"id": "battery", "type": "battery", "specs": {"cells_s": 3, "mass_g": 180}}, None),
        ({"id": "fc", "type": "fc", "specs": {"mass_g": 10}}, None),
        ({"id": "rx", "type": "rx", "specs": {"mass_g": 5}}, None),
        ({"id": "camera", "type": "camera", "specs": {"mass_g": 30}}, None),
    ]
    return write_design(root, parts, frame, mission={"preset": "endurance", "payload_ids": ["camera"]})
