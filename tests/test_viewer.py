import base64
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

from dronebench.viewer import OURS_TO_THREE, design_glb, render_viewer

HERE = Path(__file__).parent
SAMPLE = HERE / "sample_report.json"

# box extents (x, y, z) in metres, our frame: x aft, y right, z up
BOXES = {
    "fuselage": ((1.1, 0.10, 0.10), (0.35, 0, 0)),
    "wing_L": ((0.22, 1.0, 0.03), (0.30, -0.55, 0.05)),
    "wing_R": ((0.22, 1.0, 0.03), (0.30, 0.55, 0.05)),
    "htail": ((0.12, 0.45, 0.01), (1.25, 0, 0.05)),
    "vtail": ((0.12, 0.01, 0.2), (1.25, 0, 0.15)),
    "motor": ((0.04, 0.035, 0.035), (-0.22, 0, 0)),
    "ballast_nose": ((0.03, 0.03, 0.02), (-0.15, 0, 0)),
    "gps_2": ((0.02, 0.02, 0.01), (0.27, 0, 0.06)),
}


def load_sample() -> dict:
    return json.loads(SAMPLE.read_text())


def fake_design(name="trainer_demo", skip=()):
    parts = {}
    for pid, (ext, c) in BOXES.items():
        if pid in skip:
            continue
        m = trimesh.creation.box(extents=ext)
        m.apply_translation(c)
        parts[pid] = SimpleNamespace(id=pid, mesh=m)
    parts["wire_main"] = SimpleNamespace(id="wire_main", mesh=None)  # no CAD
    return SimpleNamespace(name=name, parts=parts)


def extract(html: str, script_id: str) -> str:
    m = re.search(r'<script[^>]*id="%s"[^>]*>(.*?)</script>' % re.escape(script_id), html, re.S)
    assert m, f"script {script_id} not found"
    return m.group(1)


def test_renders_self_contained_html(tmp_path):
    rep = load_sample()
    out = render_viewer(None, [rep], tmp_path / "v.html", designs=[fake_design()])
    html = out.read_text()
    # importmap with three r160 from jsDelivr
    imap = json.loads(re.search(
        r'<script type="importmap">(.*?)</script>', html, re.S).group(1))
    assert imap["imports"]["three"].startswith("https://cdn.jsdelivr.net/npm/three@0.160.0/")
    assert "three/addons/" in imap["imports"]
    # embedded data parses and carries the report
    data = json.loads(extract(html, "db-data"))
    assert data["designs"][0]["report"]["score"]["value"] == rep["score"]["value"]
    assert len(data["designs"][0]["report"]["parts"]) == len(rep["parts"])
    assert set(data["designs"][0]["mesh_ids"]) == set(BOXES)
    # embedded GLB decodes back with all part nodes
    glb = base64.b64decode(extract(html, "glb-0"))
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    names = set(scene.graph.nodes_geometry)
    assert set(BOXES) <= names
    assert "__TITLE__" not in html and "/*__DATA__*/" not in html


def test_axis_mapping_nose_left_up_y():
    glb, ids, _ = design_glb(fake_design())
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    geoms = {n: scene.geometry[scene.graph[n][1]] for n in scene.graph.nodes_geometry}
    # motor is at the nose (small x ours) -> most negative X in three
    assert geoms["motor"].bounds[:, 0].mean() < geoms["fuselage"].bounds[:, 0].mean()
    # vtail is up (z ours) -> +Y three
    assert geoms["vtail"].bounds[:, 1].mean() > 0.1
    # right wing (+y ours) -> -Z three
    assert geoms["wing_R"].bounds[:, 2].mean() < -0.3
    assert np.isclose(np.linalg.det(OURS_TO_THREE[:3, :3]), 1.0)


def test_two_designs_and_runs(tmp_path):
    before, after = load_sample(), load_sample()
    after["score"]["value"] = 45.0
    after["parts"] = [p for p in after["parts"] if p["id"] != "ballast_nose"]
    after["aircraft"]["endurance_min"] = float("nan")  # must not break JSON
    runs = tmp_path / "runs.jsonl"
    runs.write_text("\n".join(json.dumps(r) for r in [
        {"step": 0, "score": 41.6, "gate_pass": True, "failures": [], "agent": "claude"},
        {"step": 1, "score": 0, "gate_pass": False, "failures": ["climb"], "agent": "claude"},
        {"step": 2, "score": 45.0, "gate_pass": True, "failures": [], "agent": "claude"},
        {"step": 0, "score": 41.6, "gate_pass": True, "failures": [], "agent": "baseline"},
    ]) + "\n")
    out = render_viewer(None, [before, after], tmp_path / "ba.html", runs_jsonl=runs,
                        designs=[fake_design(), fake_design(skip=("ballast_nose",))])
    html = out.read_text()
    data = json.loads(extract(html, "db-data"))
    assert [d["label"] for d in data["designs"]] == ["before", "after"]
    assert data["designs"][1]["report"]["aircraft"]["endurance_min"] is None
    assert len(data["runs"]) == 4
    for i in (0, 1):
        scene = trimesh.load(io.BytesIO(base64.b64decode(extract(html, f"glb-{i}"))), file_type="glb")
        assert ("ballast_nose" in scene.graph.nodes_geometry) == (i == 0)


def test_decimation_never_fails(tmp_path):
    big = trimesh.creation.icosphere(subdivisions=6)  # 81920 faces
    d = SimpleNamespace(name="big", parts=[SimpleNamespace(id=f"s{i}", mesh=big.copy()) for i in range(3)])
    glb, ids, notes = design_glb(d, max_faces=50_000)
    assert glb and len(ids) == 3


def test_script_injection_escaped(tmp_path):
    rep = load_sample()
    rep["parts"][0]["why"] = "</script><script>alert(1)</script>"
    html = render_viewer(None, [rep], tmp_path / "x.html", designs=[fake_design()]).read_text()
    assert "<script>alert(1)" not in html
    assert json.loads(extract(html, "db-data"))["designs"][0]["report"]["parts"][0]["why"].startswith("</script>")
