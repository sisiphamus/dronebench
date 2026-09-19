from __future__ import annotations

import json

from dronebench.assemble import build_physics_input
from dronebench.gate import evaluate_gate
from dronebench.model import PhysicsResult
from dronebench.regulations import aircraft_regulations, part_regulations
from dronebench.report import build_report
from dronebench.score import compute_score

from test_gate import MASS, make_design, make_geom, make_pres
from test_necessity import run as run_necessity


def _report(pres=None, necessity=None, design=None):
    d = design or make_design()
    pres = pres or make_pres()
    geom = make_geom()
    gate = evaluate_gate(d, None, geom, pres, mass=MASS)
    return build_report(d, None, geom, MASS, pres, gate, compute_score("endurance", pres, gate), necessity,
                        {"total": 1.0}, preset="endurance")


def test_report_schema_and_json():
    d, nec = run_necessity()
    r = _report(necessity=nec, design=d)
    s = json.dumps(r, allow_nan=False)
    assert json.loads(s)["design"] == "synth"
    for k in ("design", "preset", "dronebench_version", "score", "gate", "aircraft", "regulations", "parts",
              "warnings", "timing_ms"):
        assert k in r
    assert r["score"] == {"preset": "endurance", "value": 30.0, "unit": "min", "error_band": 0.15}
    assert r["gate"]["pass"] is True and {"name", "value", "limit", "pass", "detail"} <= set(r["gate"]["checks"][0])
    parts = {p["id"]: p for p in r["parts"]}
    for p in r["parts"]:
        for k in ("function", "why", "necessity", "material", "specs", "interfaces", "sources", "removal_test"):
            assert k in p
        assert p["function"] and p["why"] and p["material"]["name"]
    assert parts["screws_motor"]["necessity"] == "required"          # group verdict mapped to the fastener
    assert parts["ballast"]["necessity"] == "unnecessary" and parts["ballast"]["score_delta_if_removed"] == 3.0
    assert parts["motor"]["operating_point"]["margin"] == 1.5
    assert parts["battery"]["operating_point"]["c_rating"] == 25
    assert parts["servo_elev"]["operating_point"]["margin"] == 3.0
    assert any(i["mates"] == "esc" for i in parts["motor"]["interfaces"])
    assert r["aircraft"]["span_m"] == 1.2 and r["aircraft"]["sources"]["span_m"] == "cad"


def test_nan_becomes_null_with_warning():
    r = _report(pres=PhysicsResult(ok=False, notes=["no wing"]))
    json.dumps(r, allow_nan=False)
    assert r["aircraft"]["stall_kmh"] is None
    assert any("stall_kmh" in w for w in r["warnings"])
    assert r["score"]["value"] == 0.0 and r["gate"]["pass"] is False
    assert all(p["necessity"] == "not_tested" for p in r["parts"])


def test_regulations():
    d = make_design()
    regs = aircraft_regulations(d, None, mass_kg=1.2)
    assert any("Part 107" in s for s in regs)
    assert "MISSING: Remote ID module required (14 CFR Part 89)" in regs
    d.parts["gps"].specs["has_remote_id"] = True
    assert "MISSING: Remote ID module required (14 CFR Part 89)" not in aircraft_regulations(d, None, mass_kg=1.2)
    assert any("44807" in s for s in aircraft_regulations(d, None, mass_kg=30))
    assert any("registration not required" in s for s in aircraft_regulations(d, None, mass_kg=0.2))
    d.parts["battery"].specs.update(cells_s=6, capacity_mah=8000)  # 177.6 Wh
    regs = aircraft_regulations(d, None, mass_kg=3)
    assert any("160 Wh" in s for s in regs) and any("UN38.3" in s for s in regs)
    assert any("2.1033" in s for s in part_regulations(d.parts["rx"]))
    assert any("UN38.3" in s for s in part_regulations(d.parts["battery"]))
    assert any("RoHS" in s for s in part_regulations(d.parts["fc"]))
    assert part_regulations(d.parts["prop"]) == []
