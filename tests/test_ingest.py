"""BOM parsing, CAD ingest, unit/axis normalisation and auto-identification."""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from dronebench.bom import load_bom
from dronebench.geometry import make_naca4_wing, make_tube
from dronebench.ingest import frame_rotation, identify, load_design
from dronebench.model import Frame
from fixtures_min import mini_trainer, write_design

DATA = Path(__file__).parent / "data"
SG90_URL = ("https://raw.githubusercontent.com/FreeCAD/FreeCAD-library/master/"
            "Electrical%20Parts/Servos/SG-90/Servo-sg90.step")


def test_frame_rotation_is_right_handed():
    for fwd, up in [("-x", "+z"), ("+y", "+z"), ("+x", "+y"), ("-z", "+y")]:
        R = frame_rotation(Frame(forward=fwd, up=up))
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)
    R = frame_rotation(Frame(forward="+y", up="+z"))
    assert np.allclose(R @ [0, 1, 0], [-1, 0, 0])   # CAD nose direction -> -x (forward)
    assert np.allclose(R @ [0, 0, 1], [0, 0, 1])


@pytest.mark.parametrize("frame", [Frame("mm", "-x", "+z"), Frame("in", "+y", "+z"), Frame("m", "+x", "+y")])
def test_units_and_axes_round_trip(tmp_path, frame):
    wing = make_naca4_wing(1.0, 0.2, 0.15, "2412", sweep_deg=5, x_le=0.3, z=0.05)
    write_design(tmp_path, [({"id": "wing", "type": "wing", "role": "wing"}, wing)], frame)
    part = load_design(tmp_path).parts["wing"]
    assert np.allclose(part.bbox_m, wing.bounds, atol=1e-6)
    assert np.isclose(part.volume_m3, wing.volume, rtol=1e-6)
    assert np.allclose(part.cg_m, wing.center_mass, atol=1e-6)
    assert part.sources["volume_m3"] == "cad"


def test_bom_parsing_and_validation(tmp_path):
    mini_trainer(tmp_path)
    bom = yaml.safe_load((tmp_path / "bom.yaml").read_text())
    bom["parts"] += [
        {"id": "gizmo", "type": "flux_capacitor", "connects": ["nope"], "qty": 3, "specs": {"mass_g": 2, "kv": 900},
         "sources": {"kv": "agent_supplied", "mass_g": "agent_supplied"}},
        {"id": "screws", "type": "fastener", "group": "motor_mount", "qty": 4, "parent": "ghost",
         "specs": {"size": "M3x8", "mass_g": 0.8}},
    ]
    (tmp_path / "bom.yaml").write_text(yaml.safe_dump(bom))
    assert load_design(tmp_path).parts["gizmo"].sources["mass_g"] == "agent_supplied"
    d = load_bom(tmp_path)
    assert d.parts["gizmo"].type == "other" and d.parts["gizmo"].qty == 3
    assert d.parts["camera"].locked and not d.parts["motor"].locked
    assert d.parts["motor"].sources["specs"] == "bom" and d.parts["motor"].connects == ["esc"]
    assert d.parts["wing"].material.name == "LW-PLA" and d.parts["wing"].material.walls == 1
    assert all(p.material is not None and p.material.name for p in d.parts.values())
    assert d.parts["screws"].group == "motor_mount"
    assert d.parts["gizmo"].sources["kv"] == "agent_supplied" and "sources" not in d.parts["gizmo"].sources
    assert d.parts["wing"].material.kind != "unknown"
    joined = "\n".join(d.warnings)
    assert "flux_capacitor" in joined and "'nope'" in joined and "'ghost'" in joined
    assert d.frame.units == "mm" and d.mission.preset == "endurance"

    bom["parts"].append({"id": "wing", "type": "wing"})
    (tmp_path / "bom.yaml").write_text(yaml.safe_dump(bom))
    with pytest.raises(ValueError, match="duplicate"):
        load_bom(tmp_path)


def test_load_design_mini_trainer(tmp_path):
    d = load_design(mini_trainer(tmp_path))
    for p in d.parts.values():
        assert p.mass_g is not None and p.mass_g >= 0 and p.cg_m is not None and p.cg_m.shape == (3,)
        assert "mass_g" in p.sources and "cg_m" in p.sources
    assert d.parts["motor"].mass_g == pytest.approx(60)
    assert d.parts["wing"].mesh is not None and d.parts["wing"].mesh.is_watertight
    # CAD-less parts: explicit cg_mm (CAD frame, mm) or the fuselage centroid
    assert np.allclose(d.parts["esc"].cg_m, [-0.1, 0, 0])
    fus = d.parts["fuselage"].bbox_m
    assert np.allclose(d.parts["battery"].cg_m, fus.mean(axis=0))
    assert d.parts["battery"].sources["cg_m"] == "assumed"
    spar = d.parts["spar"]
    assert spar.volume_m3 == pytest.approx(np.pi / 4 * (0.008 ** 2 - 0.006 ** 2) * 0.9, rel=0.01)


def test_multibody_glb_selection(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(make_tube(0.01, 0.008, 1.0, "y"), node_name="MainSpar", geom_name="MainSpar")
    box = trimesh.creation.box((0.05, 0.05, 0.05))
    box.apply_translation([0.2, 0, 0])
    scene.add_geometry(box, node_name="Block", geom_name="Block")
    (tmp_path / "cad").mkdir()
    (tmp_path / "cad" / "assembly.glb").write_bytes(scene.export(file_type="glb"))
    (tmp_path / "bom.yaml").write_text(yaml.safe_dump({"parts": [
        {"id": "spar", "type": "spar", "cad": "assembly.glb#MainSpar"},
        {"id": "block", "type": "ballast", "cad": "assembly.glb#Block"}]}))
    (tmp_path / "mission.yaml").write_text(yaml.safe_dump({"frame": {"units": "mm", "forward": "-x", "up": "+z"}}))
    d = load_design(tmp_path)
    # GLB is metres by spec: no mm scaling even though frame.units is mm
    assert np.allclose(d.parts["spar"].bbox_m, [[-0.005, -0.5, -0.005], [0.005, 0.5, 0.005]], atol=1e-6)
    assert np.allclose(d.parts["block"].cg_m, [0.2, 0, 0], atol=1e-6)
    assert not [p for p in d.parts if p not in ("spar", "block")]   # the referenced file is not auto-added


def test_unreferenced_cad_is_identified(tmp_path):
    aileron = make_naca4_wing(0.3, 0.05, 0.05, "0008", x_le=0.4)
    blob = trimesh.creation.icosphere(radius=0.02)
    fin = make_naca4_wing(0.3, 0.12, 0.12, "0008")        # flat plate shape, name says nothing
    write_design(tmp_path, [({"id": "wing", "type": "wing", "role": "wing"}, make_naca4_wing(1.0, 0.2, 0.2))],
                 extra_cad={"Aileron_L.stl": aileron, "thing.stl": blob, "part7.stl": fin})
    d = load_design(tmp_path)
    assert d.parts["Aileron_L"].type == "surface" and d.parts["Aileron_L"].role == "aileron"
    assert d.parts["Aileron_L"].confidence == pytest.approx(0.8)   # name and flat shape agree
    assert d.parts["thing"].type == "other" and d.parts["thing"].confidence == pytest.approx(0.4)
    assert d.parts["part7"].type == "tail" and d.parts["part7"].confidence == pytest.approx(0.5)
    assert d.parts["thing"].sources["type"] == "inferred"
    assert sum("not in the BOM" in w for w in d.warnings) == 3


def test_identify_shapes():
    assert identify("x", make_tube(0.01, 0.008, 1.0, "y"))[:2] == ("spar", "main_spar")
    assert identify("left_wing_panel", make_naca4_wing(1.2, 0.2, 0.2))[:2] == ("wing", "wing")
    blade = trimesh.creation.box((0.01, 0.25, 0.02))
    assert identify("blade", blade)[0] == "prop"
    assert identify("motor_2816", trimesh.creation.cylinder(0.014, 0.03))[0] == "motor"


def test_non_watertight_mesh_uses_hull(tmp_path):
    box = trimesh.creation.box((0.1, 0.1, 0.1))
    box.update_faces(np.arange(len(box.faces)) != 0)     # punch a hole
    write_design(tmp_path, [({"id": "lid", "type": "hatch"}, box)])
    d = load_design(tmp_path)
    assert d.parts["lid"].volume_m3 == pytest.approx(0.7e-3, rel=1e-6)
    assert d.parts["lid"].sources["volume_m3"] == "assumed"
    assert any("not watertight" in w for w in d.warnings)


def _sg90() -> Path:
    path = DATA / "servo_sg90.step"
    if not path.exists():
        try:
            DATA.mkdir(exist_ok=True)
            urllib.request.urlretrieve(SG90_URL, path)
        except OSError as exc:
            pytest.skip(f"SG90 STEP not available offline ({exc}); needs {SG90_URL}")
    return path


def test_step_ingest(tmp_path):
    step = _sg90()
    (tmp_path / "cad").mkdir()
    (tmp_path / "cad" / "servo.step").write_bytes(step.read_bytes())
    (tmp_path / "bom.yaml").write_text(yaml.safe_dump({"parts": [
        {"id": "servo_ail", "type": "servo", "cad": "servo.step", "specs": {"mass_g": 9}}]}))
    (tmp_path / "mission.yaml").write_text(yaml.safe_dump({"frame": {"units": "mm", "forward": "+y", "up": "+z"}}))
    p = load_design(tmp_path).parts["servo_ail"]
    ext = np.sort(p.bbox_m[1] - p.bbox_m[0])
    assert np.allclose(ext, [0.0118, 0.0299, 0.0324], atol=5e-4)   # SG90 incl. tabs, metres
    assert np.isclose(p.bbox_m[1, 0] - p.bbox_m[0, 0], 0.0324, atol=5e-4)   # CAD +y (length) -> our x
    assert p.volume_m3 > 1e-6 and p.mass_g == pytest.approx(9)


def test_ingest_50_parts_under_3s(tmp_path):
    parts = []
    for i in range(50):
        m = trimesh.creation.box((0.05, 0.02, 0.01)) if i % 2 else make_tube(0.01, 0.008, 0.3, "y")
        m.apply_translation([0.06 * i, 0, 0])
        parts.append(({"id": f"p{i}", "type": "other", "specs": {"mass_g": 1}}, m))
    write_design(tmp_path, parts)
    t = time.perf_counter()
    d = load_design(tmp_path)
    assert time.perf_counter() - t < 3.0
    assert len(d.parts) == 50



def test_y_up_frame_and_cg_mm(tmp_path):
    """OpenGeoDrone-style CAD: x aft, y up, z = left span. cg_mm is a CAD-frame point in millimetres."""
    frame = Frame("mm", "-x", "+y")
    R = frame_rotation(frame)
    assert np.allclose(R @ [0, 0, 1], [0, -1, 0]) and np.allclose(R @ [0, 1, 0], [0, 0, 1])
    wing = make_naca4_wing(1.0, 0.2, 0.2, "2412", x_le=0.3, z=0.05)
    write_design(tmp_path, [({"id": "wing", "type": "wing", "role": "wing"}, wing),
                            ({"id": "gps", "type": "gps", "specs": {"mass_g": 20, "cg_mm": [400, 60, -150]}}, None)],
                 frame)
    d = load_design(tmp_path)
    ext = d.parts["wing"].bbox_m[1] - d.parts["wing"].bbox_m[0]
    assert ext[1] == pytest.approx(1.0) and d.parts["wing"].cg_m[2] > 0.04   # span along y, up is +z
    assert np.allclose(d.parts["gps"].cg_m, [0.4, 0.15, 0.06])                # CAD z = -150 (right) -> y = +0.15
    assert d.parts["gps"].sources["cg_m"] == "bom"
