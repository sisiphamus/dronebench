"""Build the report dict (PLAN §4). Everything returned is JSON-serialisable: numpy -> python,
NaN/inf -> null (with a warning naming the field)."""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .assemble import LinkIndex, edge_links, pair_motors, part_mass_g, spec
from .model import AeroGeometry, Design, GateResult, MassProps, Material, Part, PhysicsResult

VERSION = "0.1.0"
SERVO_TYPES = ("servo",)


def _clean(obj: Any, path: str, warnings: list) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v, f"{path}.{k}", warnings) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, f"{path}[{i}]", warnings) for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist(), path, warnings)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            warnings.append(f"{path} is {f}; reported as null")
            return None
        return round(f, 6)
    if obj is None or isinstance(obj, (int, str)):
        return obj
    return str(obj)


def material_dict(part: Part) -> dict:
    m: Optional[Material] = part.material
    if m is None or not m.name or m.name == "unknown":
        try:
            from . import materials
            m = materials.resolve_material(part)
        except Exception:
            m = None
    if m is None:
        m = Material(name=f"unknown {part.type} material", density_kg_m3=0.0, kind="mixed", source="assumed")
    return {"name": m.name, "kind": m.kind, "sub": dict(m.sub), "infill": m.infill, "walls": m.walls,
            "orientation": m.orientation, "density_kg_m3": m.density_kg_m3, "tensile_mpa": m.tensile_mpa,
            "modulus_gpa": m.modulus_gpa, "source": m.source}


def interfaces(part: Part, design: Design, graph, links: LinkIndex) -> list[dict]:
    out = []
    seen = set()
    if graph is not None and part.id in graph:
        for nb in graph.neighbors(part.id):
            if nb not in design.parts:
                continue
            for l in edge_links(graph, part.id, nb):
                out.append({"mates": nb, "kind": l.get("kind"), "joint": l.get("joint"), "gap_mm": l.get("gap_mm"),
                            "tolerance": l.get("tolerance"), "source": l.get("source", "inferred")})
                seen.add((nb, l.get("kind")))
    for nb, kinds in sorted(links.adj.get(part.id, {}).items()):
        for k in kinds:
            if k != "physical" and not any(s[0] == nb for s in seen):
                kind = "electrical" if k == "electrical" else "mechanical"
                out.append({"mates": nb, "kind": kind, "joint": "declared in BOM", "gap_mm": None,
                            "tolerance": None, "source": "bom"})
                seen.add((nb, kind))
    rank = {"physical": 0, "mechanical": 1, "electrical": 2}
    return sorted(out, key=lambda d: (rank.get(d["kind"], 3), d["mates"]))


def _ratio(a, b):
    try:
        return float(a) / float(b) if b and math.isfinite(float(a)) and math.isfinite(float(b)) and float(b) > 0 else None
    except (TypeError, ValueError):
        return None


def operating_point(part: Part, design: Design, pres: PhysicsResult, pin, links: LinkIndex) -> Optional[dict]:
    t = part.type
    if t == "motor":
        imax = float(spec(part, "imax_a", default=float("nan")))
        in_train = pin is not None and any(tr.motor.id == part.id for tr in pin.trains)
        if not in_train:
            return {"cruise_a": None, "max_a": None, "limit_a": imax, "margin": None,
                    "note": "not in a complete power train"}
        return {"cruise_a": pres.motor_current_cruise_a, "max_a": pres.motor_current_max_a, "limit_a": imax,
                "margin": _ratio(imax, pres.motor_current_max_a)}
    if t == "esc":
        imax = float(spec(part, "imax_a", default=30.0))
        return {"cruise_a": pres.motor_current_cruise_a, "max_a": pres.motor_current_max_a, "limit_a": imax,
                "margin": _ratio(imax, pres.motor_current_max_a)}
    if t == "battery":
        b = pin.battery if pin is not None else None
        rating = b.c_rating if b is not None else spec(part, "c_rating")
        return {"c_needed": pres.battery_c_max, "c_rating": rating, "max_a": pres.battery_current_max_a,
                "energy_wh": pres.battery_energy_wh, "margin": _ratio(rating, pres.battery_c_max)}
    if t == "servo":
        sv = next((s for s in (pin.servos if pin is not None else []) if s.id == part.id), None)
        return {"torque_kgcm": sv.torque_kgcm if sv else None, "surface": sv.surface_id if sv else None,
                "margin": pres.servo_margin.get(part.id)}
    if t == "spar":
        return {"load_factor": 3.5, "sf": pres.spar_sf, "margin": pres.spar_sf}
    if t == "prop":
        motor = next((m for m, v in pair_motors(design, links).items() if v.get("prop") == part.id), None)
        return {"motor": motor, "max_a": pres.motor_current_max_a if motor else None}
    return None


def aircraft_block(design: Design, geom: AeroGeometry, mass: MassProps, pres: PhysicsResult) -> dict:
    w = geom.wing if geom is not None else None
    kmh = lambda v: v * 3.6 if v is not None else None  # noqa: E731
    ac = {
        "mass_g": mass.total_kg * 1000, "cg_m": mass.cg_m,
        "span_m": w.span_m if w else None, "wing_area_m2": w.area_m2 if w else None,
        "aspect_ratio": w.aspect_ratio if w else None, "mac_m": w.mac_m if w else None,
        "airfoil": w.airfoil if w else None,
        "stall_kmh": kmh(pres.stall_ms), "cruise_kmh": kmh(pres.cruise_ms), "vmax_kmh": kmh(pres.vmax_ms),
        "best_endurance_kmh": kmh(pres.best_endurance_ms), "best_range_kmh": kmh(pres.best_range_ms),
        "climb_ms": pres.climb_ms, "endurance_min": pres.endurance_min, "range_km": pres.range_km,
        "wh_per_km": pres.wh_per_km, "km_per_wh": pres.km_per_wh, "static_margin": pres.static_margin,
        "neutral_point_x_m": pres.neutral_point_x_m, "tail_vh": pres.tail_vh, "tail_vv": pres.tail_vv,
        "spar_sf": pres.spar_sf, "cd0": pres.cd0, "oswald_e": pres.oswald_e, "ld_max": pres.ld_max,
        "cl_max": pres.cl_max, "battery_energy_wh": pres.battery_energy_wh, "payload_fraction": pres.payload_fraction,
        "fuselage_length_m": geom.fuselage_length_m if geom is not None else None,
    }
    src = {k: "computed" for k in ac}
    for k in ("span_m", "wing_area_m2", "aspect_ratio", "mac_m", "airfoil", "fuselage_length_m"):
        src[k] = "cad"
    src["mass_g"] = src["cg_m"] = "computed"
    if design.mission.cruise_kmh:
        src["cruise_kmh"] = "bom"
    ac["sources"] = src
    return ac


def part_entry(part: Part, design: Design, graph, links: LinkIndex, mass: MassProps, pres: PhysicsResult, pin,
               necessity: Optional[dict]) -> dict:
    from .functions import describe
    from .necessity import key_for
    from .regulations import part_regulations
    fn, why = describe(part, design, graph, links=links)
    if necessity is None:
        verdict, reason, delta = "not_tested", "removal test not run (necessity off)", None
    else:
        r = necessity.get(key_for(part))
        if r is None:
            verdict, reason, delta = "not_tested", "no removal-test result for this part", None
        else:
            verdict, reason = r.verdict, r.reason
            delta = None if r.verdict == "required" else r.score_delta
    sources = dict(part.sources)
    mat = material_dict(part)
    sources.setdefault("material", mat["source"])
    sources.setdefault("function", "inferred")
    sources.setdefault("necessity", "computed")
    sources.setdefault("specs", "bom" if part.specs else "catalog")
    specs = dict(part.specs)
    if not specs:
        try:
            from .components import lookup
            hit = lookup(part.part_number) if part.part_number else None
            if hit:
                specs = dict(hit.get("specs") or {})
                sources["specs"] = "catalog"
        except Exception:
            pass
    return {
        "id": part.id, "part_number": part.part_number, "type": part.type, "role": part.role, "qty": part.qty,
        "group": part.group, "locked": part.locked,
        "function": fn, "why": why, "necessity": verdict, "removal_test": reason, "score_delta_if_removed": delta,
        "mass_g": part_mass_g(part, mass), "cg_m": part.cg_m, "specs": specs, "material": mat,
        "interfaces": interfaces(part, design, graph, links),
        "operating_point": operating_point(part, design, pres, pin, links),
        "regulations": part_regulations(part, design, pres),
        "confidence": part.confidence, "sources": sources, "cad": part.cad,
    }


def build_report(design: Design, graph, geom: AeroGeometry, mass: MassProps, pres: PhysicsResult, gate: GateResult,
                 score, necessity: Optional[dict], timing: Optional[dict], preset: Optional[str] = None, pin=None,
                 links: Optional[LinkIndex] = None) -> dict:
    from .regulations import aircraft_regulations
    preset = preset or design.mission.preset
    links = links or LinkIndex(design, graph)
    if pin is None:
        from .assemble import build_physics_input
        pin = build_physics_input(design, geom, mass, graph, links=links)
    value, unit = score if isinstance(score, tuple) else (score, None)
    warnings = list(dict.fromkeys(list(design.warnings)
                                  + list((graph.graph.get("warnings") or []) if graph is not None else [])
                                  + list(geom.warnings if geom is not None else [])
                                  + [f"physics: {n}" for n in pres.notes]))
    report = {
        "design": design.name, "preset": preset, "dronebench_version": VERSION,
        "score": {"preset": preset, "value": value, "unit": unit, "error_band": pres.error_band},
        "gate": {"pass": gate.passed, "checks": [{"name": c.name, "value": c.value, "limit": c.limit,
                                                   "pass": c.passed, "detail": c.detail} for c in gate.checks]},
        "aircraft": aircraft_block(design, geom, mass, pres),
        "regulations": aircraft_regulations(design, pres, mass_kg=mass.total_kg),
        "parts": [part_entry(p, design, graph, links, mass, pres, pin, necessity) for p in design.parts.values()],
        "necessity_summary": _summary(necessity),
        "warnings": warnings,
        "timing_ms": dict(timing or {}),
    }
    nan_warnings: list = []
    out = _clean(report, "report", nan_warnings)
    # NaNs in physics outputs of a failed run are expected; list each field once
    out["warnings"].extend(w.replace("report.", "") for w in nan_warnings
                           if not w.startswith("report.parts") or "operating_point" not in w)
    return out


def _summary(necessity: Optional[dict]) -> dict:
    if not necessity:
        return {}
    out = {"required": 0, "helpful": 0, "unnecessary": 0}
    for r in necessity.values():
        out[r.verdict] = out.get(r.verdict, 0) + 1
    return out
