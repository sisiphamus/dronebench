import filecmp
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"
sys.path.insert(0, str(FIX))

from dronebench.materials import part_mass_g  # noqa: E402
from dronebench.model import AERO_ROLES, PART_TYPES, Material, Part  # noqa: E402


def _bom(name):
    return yaml.safe_load(open(FIX / name / "bom.yaml"))["parts"]


def _load(path, scale):
    m = trimesh.load(path, force="mesh")
    m.apply_scale(scale)
    return m


def _mass_g(name, scale=1e-3):
    tot = 0.0
    for e in _bom(name):
        md = e.get("material")
        mat = Material(name=md["name"], density_kg_m3=0, kind="", infill=md.get("infill"), walls=md.get("walls"),
                       orientation=md.get("orientation")) if md else None
        p = Part(id=e["id"], type=e["type"], part_number=e.get("part_number"), material=mat,
                 specs=e.get("specs", {}), qty=e.get("qty", 1))
        if e.get("cad", "").endswith((".stl", ".3mf")):
            p.mesh = _load(FIX / name / "cad" / e["cad"], scale)
        g, _ = part_mass_g(p)
        tot += g * p.qty
    return tot


@pytest.mark.parametrize("name", ["falcon_v2", "trainer", "bad_tail", "step_mix", "third_party"])
def test_design_folder_valid(name):
    d = FIX / name
    parts = _bom(name)
    ids = [p["id"] for p in parts]
    assert len(ids) == len(set(ids))
    for p in parts:
        assert p["type"] in PART_TYPES, p
        role = p.get("role")
        assert role in AERO_ROLES or role in ("main_spar", "rear_spar"), p
        if "cad" in p:
            assert (d / "cad" / p["cad"]).exists(), p["cad"]
        else:
            assert "mass_g" in p.get("specs", {}) and "cg_mm" in p["specs"], p["id"]
        for ref in p.get("connects", []) + p.get("specs", {}).get("fits", []) + ([p["parent"]] if "parent" in p else []):
            assert ref in ids, (p["id"], ref)
    mission = yaml.safe_load(open(d / "mission.yaml"))
    assert mission["frame"]["units"] == "mm"
    for pid in mission.get("payload_ids", []):
        assert pid in ids


def test_falcon_meshes_watertight():
    for p in _bom("falcon_v2"):
        if "cad" in p:
            m = trimesh.load(FIX / "falcon_v2" / "cad" / p["cad"], force="mesh")
            assert m.is_watertight, p["id"]


def test_falcon_mass():
    assert 3300 <= _mass_g("falcon_v2") <= 3700


def test_falcon_contents():
    parts = {p["id"]: p for p in _bom("falcon_v2")}
    for waste in ("ballast_nose", "gps_2", "spar_doubler", "fairing_L", "fairing_R", "fuse_armor"):
        assert waste in parts
    assert "remote_id" not in {p["type"] for p in parts.values()}
    servos = [p for p in parts.values() if p["type"] == "servo"]
    assert len(servos) == 7
    for s in servos:
        assert any(parts[c]["type"] == "surface" for c in s["connects"]), s["id"]
    assert parts["camera"]["locked"] is True
    assert parts["main_spar"]["specs"]["outer_d_mm"] == 10 and parts["rear_spar"]["specs"]["outer_d_mm"] == 6
    assert {p.get("group") for p in parts.values() if p["type"] == "fastener"} >= {"motor_L_mount", "motor_R_mount",
                                                                                     "wing_mount", "servo_mount"}


def test_falcon_planform_area():
    params = yaml.safe_load(open(FIX / "falcon_v2" / "params.yaml"))["wing"]
    b2, cr, ct, yb = params["span_mm"] / 2, params["root_chord_mm"], params["tip_chord_mm"], params["break_y_mm"]
    area_m2 = 2 * (cr * yb + 0.5 * (cr + ct) * (b2 - yb)) * 1e-6
    assert abs(area_m2 - 0.4514) / 0.4514 < 0.03


def test_falcon_wing_meshes_match_planform():
    """Projected (x-y) area of wing + flap + aileron meshes equals the planform."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = []
    for p in _bom("falcon_v2"):
        if p.get("role") in ("wing", "flap", "aileron"):
            m = trimesh.load(FIX / "falcon_v2" / "cad" / p["cad"], force="mesh")
            polys += [Polygon(t[:, :2]).buffer(0.01) for t in m.triangles if Polygon(t[:, :2]).area > 1e-9]
    area = unary_union(polys).area * 1e-6
    assert abs(area - 0.4514) / 0.4514 < 0.03, area


def test_falcon_rebuild_deterministic(tmp_path):
    import build_falcon_v2 as b
    params = yaml.safe_load(open(FIX / "falcon_v2" / "params.yaml"))
    t0 = time.time()
    b.build(params, tmp_path / "a", verbose=False)
    assert time.time() - t0 < 10
    b.build(params, tmp_path / "b", verbose=False)
    assert filecmp.cmp(tmp_path / "a" / "bom.yaml", tmp_path / "b" / "bom.yaml", shallow=False)
    assert filecmp.cmp(tmp_path / "a" / "bom.yaml", FIX / "falcon_v2" / "bom.yaml", shallow=False)
    for f in ("wing_L1.stl", "prop_R.stl", "fuse3.stl"):
        assert filecmp.cmp(tmp_path / "a" / "cad" / f, tmp_path / "b" / "cad" / f, shallow=False)


def test_trainer_mass_and_variants():
    assert 950 <= _mass_g("trainer") <= 1250
    tr = {p["id"]: p for p in _bom("trainer")}
    bad = {p["id"]: p for p in _bom("bad_tail")}
    area = lambda d, i: trimesh.load(FIX / d / "cad" / f"{i}.stl").area  # noqa: E731
    assert area("bad_tail", "hstab") < 0.6 * area("trainer", "hstab")
    bx = lambda d: trimesh.load(FIX / d / "cad" / "battery.stl").bounds.mean(0)[0]  # noqa: E731
    assert bx("bad_tail") > bx("trainer") + 150
    assert set(tr) == set(bad)


def test_step_mix_servos_are_step():
    parts = _bom("step_mix")
    servos = [p for p in parts if p["type"] == "servo"]
    assert len(servos) == 4 and all(p["cad"].endswith(".step") for p in servos)
    head = (FIX / "step_mix" / "cad" / servos[0]["cad"]).read_text()[:200]
    assert head.startswith("ISO-10303-21")


def test_third_party_loads():
    parts = _bom("third_party")
    cad = [p for p in parts if "cad" in p]
    assert any(p["cad"].endswith(".3mf") for p in cad) and any(p["cad"].endswith(".stl") for p in cad)
    mission = yaml.safe_load(open(FIX / "third_party" / "mission.yaml"))
    assert mission["frame"] == {"units": "mm", "forward": "-x", "up": "+y"}
    span = [trimesh.load(FIX / "third_party" / "cad" / p["cad"], force="mesh").bounds[:, 2] for p in cad
            if p.get("role") == "wing"]
    assert np.ptp(np.array(span)) > 1000   # ~1.07 m span along the CAD z axis
    assert (FIX / "third_party" / "SOURCE.md").exists()
    assert 1200 < _mass_g("third_party") < 1700
