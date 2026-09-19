"""End-to-end evaluation: design folder -> report dict (PLAN §4).

ingest -> plausibility (agent-supplied specs) -> graph -> geometry -> mass -> assemble -> physics
-> gate -> score -> removal test (optional) -> report.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .model import Design, PhysicsResult

NECESSITY_WORKERS = 4  # threads; ~2x faster on falcon_v2 (3.5 s -> 1.6 s), verdicts identical


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def apply_plausibility(design: Design) -> list[str]:
    """Clamp agent-supplied specs; refresh mass for clamped parts. Returns warning strings."""
    from .plausibility import check_specs, needs_check
    out = []
    for p in design.parts.values():
        if not needs_check(p):
            continue
        changes = check_specs(p)
        for field, old, new, reason in changes:
            out.append(f"{p.id}.{field}: agent value {old} clamped to {new} ({reason})")
        if any(c[0] == "mass_g" for c in changes):
            p.mass_g = float(p.specs["mass_g"])
            p.sources["mass_g"] = "agent_supplied_clamped"
    return out


def core(design: Design, preset: str, graph=None, geom=None, timing: Optional[dict] = None) -> dict:
    """graph (built if None) -> geometry -> mass -> physics -> gate -> score. Returns the pieces."""
    from . import geometry, physics
    from .assemble import LinkIndex, build_physics_input
    from .gate import evaluate_gate
    from .score import compute_score
    t = timing if timing is not None else {}
    t0 = time.perf_counter()
    if graph is None:
        from .graph import build_graph
        graph = build_graph(design)
        t["graph"] = _ms(t0)
    t1 = time.perf_counter()
    if geom is None:
        geom = geometry.extract_geometry(design)
        t["geometry"] = _ms(t1)
    t1 = time.perf_counter()
    mass = geometry.mass_properties(design)
    links = LinkIndex(design, graph)
    pin = build_physics_input(design, geom, mass, graph, links=links)
    try:
        pres = physics.run(pin)
    except Exception as e:
        pres = PhysicsResult(ok=False, mass_kg=mass.total_kg, notes=[f"physics error: {type(e).__name__}: {e}"])
    t["physics"] = _ms(t1)
    t1 = time.perf_counter()
    gate = evaluate_gate(design, graph, geom, pres, pin=pin, mass=mass, links=links)
    score = compute_score(preset, pres, gate)
    t["gate"] = _ms(t1)
    return {"design": design, "graph": graph, "links": links, "geom": geom, "mass": mass, "pin": pin,
            "pres": pres, "gate": gate, "score": score}


def evaluate_design(design: Design, preset: Optional[str] = None, necessity: bool = True,
                    workers: int = NECESSITY_WORKERS) -> dict:
    """Evaluate an in-memory (already ingested) Design and return the report dict."""
    from .report import build_report
    t0 = time.perf_counter()
    timing: dict = {}
    preset = preset or design.mission.preset or "endurance"
    warnings = apply_plausibility(design)
    design.warnings.extend(warnings)
    bits = core(design, preset, timing=timing)
    nec = None
    if necessity:
        from .necessity import removal_test
        nec = removal_test(design, preset, bits, workers=workers, timing=timing)
    timing["total"] = _ms(t0)
    return build_report(design, bits["graph"], bits["geom"], bits["mass"], bits["pres"], bits["gate"],
                        bits["score"], nec, timing, preset=preset, pin=bits["pin"], links=bits["links"])


def evaluate(design_dir, preset: Optional[str] = None, necessity: bool = True) -> dict:
    """Load a design folder and return its report dict (PLAN §4)."""
    from .ingest import load_design
    t0 = time.perf_counter()
    design = load_design(Path(design_dir))
    load_ms = _ms(t0)
    report = evaluate_design(design, preset=preset, necessity=necessity)
    report["timing_ms"]["ingest"] = load_ms
    report["timing_ms"]["total"] = round(report["timing_ms"].get("total", 0) + load_ms, 1)
    return report
