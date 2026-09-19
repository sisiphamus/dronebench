from __future__ import annotations

import os

import numpy as np
import pytest
import trimesh
import yaml

from dronebench.edits import apply_edits
from dronebench.env import DroneBench


def make_tiny(root, forward="-x"):
    cad = root / "cad"
    cad.mkdir(parents=True)
    for name, ext, at in (("wing", (100, 400, 10), (0, 0, 0)), ("fuse", (300, 40, 40), (0, 0, -30)),
                          ("ballast", (10, 10, 10), (-100, 0, -30))):
        b = trimesh.creation.box(extents=ext)
        b.apply_translation(at)
        b.export(cad / f"{name}.stl")
    bom = {"parts": [
        {"id": "wing", "type": "wing", "role": "wing", "cad": "wing.stl", "material": {"name": "LW-PLA", "infill": 0.1}},
        {"id": "fuse", "type": "fuselage", "cad": "fuse.stl", "connects": ["ballast"]},
        {"id": "ballast", "type": "ballast", "cad": "ballast.stl", "specs": {"mass_g": 50}},
        {"id": "motor", "type": "motor", "part_number": "M1", "specs": {"kv": 900, "mass_g": 60, "cg_mm": [-150, 0, 0]}},
        {"id": "camera", "type": "camera", "specs": {"mass_g": 30}},
    ]}
    (root / "bom.yaml").write_text(yaml.safe_dump(bom, sort_keys=False))
    (root / "mission.yaml").write_text(yaml.safe_dump({"preset": "endurance", "payload_ids": ["camera"],
                                                       "frame": {"units": "mm", "forward": forward, "up": "+z"}}))
    return root


def bom_of(d):
    return {e["id"]: e for e in yaml.safe_load((d / "bom.yaml").read_text())["parts"]}


def test_remove_set_swap(tmp_path):
    src = make_tiny(tmp_path / "d")
    out, errs = apply_edits(src, [
        {"op": "remove", "id": "ballast"},
        {"op": "set", "id": "wing", "path": "material.infill", "value": 0.0},
        {"op": "set", "id": "motor", "path": "specs.imax_a", "value": 40},
        {"op": "swap", "id": "motor", "entry": {"part_number": "BrotherHobby Avenger 2816 620KV",
                                                "specs": {"kv": 620, "mass_g": 128}}},
    ], tmp_path / "out")
    assert errs == []
    b = bom_of(out)
    assert "ballast" not in b and b["fuse"]["connects"] == []
    assert not (out / "cad" / "ballast.stl").exists() and (src / "cad" / "ballast.stl").exists()
    assert b["wing"]["material"]["infill"] == 0.0 and b["wing"]["sources"]["material"] == "agent_supplied"
    assert b["motor"]["specs"] == {"kv": 620, "mass_g": 128}
    assert b["motor"]["sources"]["kv"] == "agent_supplied" and b["motor"]["sources"]["part_number"] == "agent_supplied"
    assert "ballast" in bom_of(src)  # source untouched


def test_swap_part_number_clears_specs(tmp_path):
    src = make_tiny(tmp_path / "d")
    out, errs = apply_edits(src, [{"op": "swap", "id": "motor", "entry": {"part_number": "Other 1000KV"}}],
                            tmp_path / "o")
    assert errs == [] and bom_of(out)["motor"]["specs"] == {}


@pytest.mark.parametrize("forward", ["-x", "+x"])
def test_scale_and_move(tmp_path, forward):
    src = make_tiny(tmp_path / "d", forward=forward)
    out, errs = apply_edits(src, [
        {"op": "scale", "ids": ["wing"], "factors": [1.0, 1.5, 1.0], "about": "part_cg"},
        {"op": "move", "ids": ["fuse"], "offset_m": [0.01, 0, 0]},
        {"op": "move", "ids": ["motor"], "offset_m": [0.02, 0, 0]},
    ], tmp_path / "o")
    assert errs == []
    wing = trimesh.load(out / "cad" / "wing.stl")
    assert np.allclose(wing.extents, [100, 600, 10], atol=1e-6)
    fuse = trimesh.load(out / "cad" / "fuse.stl")
    sign = 1 if forward == "-x" else -1        # our +x (aft) is CAD +x when the nose points to -x
    assert fuse.bounds.mean(axis=0)[0] == pytest.approx(sign * 10, abs=1e-6)
    assert bom_of(out)["motor"]["specs"]["cg_mm"][0] == pytest.approx(-150 + sign * 20)
    # the edited file is a new inode, the source mesh is unchanged
    assert np.allclose(trimesh.load(src / "cad" / "wing.stl").extents, [100, 400, 10])
    assert os.stat(out / "cad" / "fuse.stl").st_ino != os.stat(src / "cad" / "fuse.stl").st_ino


def test_add_and_mission(tmp_path):
    src = make_tiny(tmp_path / "d")
    extra = tmp_path / "gps.stl"
    trimesh.creation.box(extents=[20, 20, 5]).export(extra)
    out, errs = apply_edits(src, [
        {"op": "add", "entry": {"id": "gps", "type": "gps", "specs": {"mass_g": 20}}, "cad_file": str(extra)},
        {"op": "mission", "path": "cruise_kmh", "value": 50},
    ], tmp_path / "o")
    assert errs == []
    b = bom_of(out)
    assert b["gps"]["cad"] == "gps.stl" and (out / "cad" / "gps.stl").exists()
    assert b["gps"]["sources"]["mass_g"] == "agent_supplied"
    assert yaml.safe_load((out / "mission.yaml").read_text())["cruise_kmh"] == 50


@pytest.mark.parametrize("ops,word", [
    ([{"op": "remove", "id": "nope"}], "no part"),
    ([{"op": "explode", "id": "wing"}], "unknown op"),
    ([{"op": "set", "id": "wing", "path": "locked", "value": False}], "not editable"),
    ([{"op": "set", "id": "wing", "path": "material.infill", "value": 3}], "0-1"),
    ([{"op": "scale", "ids": ["wing"], "factors": [1, 9, 1]}], "factors"),
    ([{"op": "scale", "ids": ["motor"], "factors": [1, 1.1, 1]}], "no CAD"),
    ([{"op": "move", "ids": ["camera"], "offset_m": [0.01, 0, 0]}], "cg_mm"),
    ([{"op": "add", "entry": {"id": "wing", "type": "wing"}}], "already exists"),
    ([{"op": "mission", "path": "payload_ids", "value": []}], "payload"),
])
def test_invalid_ops_write_nothing(tmp_path, ops, word):
    src = make_tiny(tmp_path / "d")
    out, errs = apply_edits(src, [{"op": "remove", "id": "ballast"}] + ops, tmp_path / "o")
    assert errs and word in " ".join(errs)
    assert not (tmp_path / "o").exists()


def test_out_dir_guard(tmp_path):
    src = make_tiny(tmp_path / "d")
    _, errs = apply_edits(src, [], src)
    assert errs and "differ" in errs[0]


# ------------------------------------------------------------------ env

def fake_report(d):
    b = bom_of(d)
    ok = "camera" in b
    score = 10.0 + (5.0 if "ballast" not in b else 0.0)
    return {"design": d.name, "preset": "endurance", "score": {"value": score if ok else 0.0, "unit": "min"},
            "gate": {"pass": ok, "checks": [{"name": "payload", "pass": ok, "detail": "" if ok else "camera missing"}]},
            "aircraft": {}, "parts": []}


def test_env_accept_reject(tmp_path, monkeypatch):
    src = make_tiny(tmp_path / "d")
    monkeypatch.setattr(DroneBench, "_evaluate", lambda self, d, necessity: fake_report(d))
    runs = tmp_path / "runs.jsonl"
    env = DroneBench(src, workdir=tmp_path / "w", max_steps=3, runs_path=runs)
    obs = env.reset()
    assert obs["score"]["value"] == 10.0
    obs, r, done, info = env.step([{"op": "remove", "id": "ballast"}])
    assert r == 5.0 and not info["rejected"] and env.current.name == "step_1" and not done
    obs, r, done, info = env.step([{"op": "remove", "id": "camera"}])
    assert r == 0.0 and info["rejected"] and info["score"] == 0.0 and "payload" in info["failures"][0]
    assert env.current.name == "step_1" and obs["score"]["value"] == 15.0
    obs, r, done, info = env.step([{"op": "remove", "id": "nope"}])
    assert info["rejected"] and info["errors"] and done
    lines = runs.read_text().strip().splitlines()
    assert len(lines) == 4


def test_baseline_greedy(tmp_path, monkeypatch):
    from dronebench import baseline
    src = make_tiny(tmp_path / "d")
    monkeypatch.setattr(DroneBench, "_evaluate", lambda self, d, necessity: {
        **fake_report(d), "parts": [{"id": k, **v} for k, v in bom_of(d).items()]})
    res = baseline.random_search(src, steps=6, seed=1, out_dir=tmp_path / "b", runs_path=tmp_path / "r.jsonl")
    assert res["best_score"] >= res["start_score"] and res["steps"] == 6
    assert len((tmp_path / "r.jsonl").read_text().strip().splitlines()) == 7
