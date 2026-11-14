# DroneBench — Build Plan

DroneBench is a benchmark environment for small electric fixed-wing UAVs. You give it a design folder (CAD + parts list), and it returns a score, a pass/fail flight check, and a report on every part: its function, why it is there, whether it is needed, specs, material, tolerances, headroom and regulations. It never suggests changes. An outside agent (Claude Code in the demo, a recommender later) edits the design and re-scores it.

## 1. How it works (the pipeline)

```
design/ ──► ingest ──► part graph ──► geometry ──► mass ──► physics ──► gate ──► score
  cad/        meshes     contacts +     wing/tail    CG +     aero,        pass/     preset
  bom.yaml    per part   electrical     planform,    total    propulsion,  fail      value
  mission.yaml           links          airfoil               structure
                                                                  │
                                           removal test ◄─────────┘ (rerun per part / fastener group)
                                                  │
                                              report.json ──► viewer.html (3D, colored by necessity)
```

1. **Ingest** (`bom.py`, `ingest.py`): read `bom.yaml` + `mission.yaml`, load every CAD file (STL/OBJ/GLB/3MF via trimesh, STEP via cascadio), convert units, rotate to our frame (x aft, y right, z up, metres). One BOM entry maps to one CAD body. BOM entries with no CAD (wires, small electronics) are allowed; their mass comes from specs.
2. **Materials + mass** (`materials.py`, `components.py`): every part gets a Material. Printed parts: effective density from filament density, walls and infill, using the mesh's volume and surface area. Bought parts: specs mass, or the catalog. A missing material is inferred from the part type and tagged `assumed`.
3. **Part graph** (`graph.py`): a networkx graph. Physical edges come from mesh proximity (contact or a gap under 1 mm), with the measured gap. Electrical edges come from `connects`, or are inferred by type rules (battery→ESC→motor, FC→servo, FC→RX/GPS). Each edge gets a joint type and an inferred tolerance.
4. **Geometry** (`geometry.py`): group parts by aero role (wing, htail, vtail, plus control surfaces added into their parent). Slice spanwise to get chord, leading-edge x and z per station. From those: span, area, MAC, AR, taper, sweep, dihedral, aerodynamic centre x, and airfoil thickness/camber (matched to a NACA 4-digit designation). Also fuselage length/diameter and wetted areas.
5. **Physics** (`physics/`), pure functions with no I/O:
   - Airfoil: NACA 4-digit lift slope, zero-lift angle and CLmax at the flight Reynolds number (empirical).
   - Drag: CD0 from component build-up (flat-plate skin friction × form factor × wetted area / S_ref, plus 10% misc). Induced drag uses the Oswald e estimate.
   - Stability: neutral point from wing + tail lift slopes and downwash; static margin = (x_np − x_cg)/MAC; tail volumes Vh, Vv.
   - Propulsion: motor model (KV, Rm, I0) + generic prop model (Ct, Cp versus advance ratio from pitch/diameter) + battery with internal resistance. Solve for the throttle/rpm that makes thrust = drag.
   - Performance: stall, best-endurance speed, best-range speed, Vmax, max climb rate, endurance (min), range (km), Wh/km.
   - Structure: wing spar as a cantilever with elliptic lift at load factor 3.5; root bending stress in the tube(s) versus strength gives the safety factor. Printed skins are non-structural.
   - Servos: hinge moment = Cq × q_max × S_surface × c_surface (Cq ≈ 0.3 at full deflection) versus rated torque.
   - Calibration target: the Falcon V2 at 3.5 kg AUW gives 1.6 Wh/km ± 15% at cruise (45–65 km/h).
6. **Gate** (`gate.py`): these checks apply to every design.

| Check | Pass if |
|---|---|
| lifting_surfaces | wing present; horizontal and vertical stability present (tail, or V-tail) |
| static_margin | 0.05 ≤ SM ≤ 0.20 |
| stall_margin | cruise ≥ 1.3 × stall |
| climb | ≥ 2 m/s |
| spar | safety factor ≥ 1.5 at 3.5 g |
| tail_volume | Vh 0.35–0.7 and Vv 0.02–0.06 (V-tail: projected equivalents) |
| servo_torque | every surface servo margin ≥ 1.5 |
| motor_current | full-throttle current ≤ motor imax |
| esc_current | ≤ ESC rating |
| battery_current | C needed ≤ C rating |
| power_chain | battery→ESC→motor→prop complete for every motor |
| control_chain | FC present; RX present; every control surface has a servo linked to FC |
| payload | every locked part present |
| clearance | no part interpenetrates another by > 1 mm (except declared fits); prop disc clears airframe |

7. **Score** (`score.py`): 0 if the gate fails, otherwise:

| Preset | Score |
|---|---|
| endurance | minutes at best-endurance speed |
| range | km at best-range speed |
| speed | Vmax km/h |
| payload | payload mass / all-up mass |
| efficiency | km per Wh at cruise |

8. **Removal test** (`necessity.py`): remove each part (fasteners by `group`) in a copy, rerun steps 3–7. The result is `required` (the gate fails; the reason is the failing check), `helpful` (the gate passes but a margin gets worse by > 5%), or `unnecessary` (the gate passes and the score does not drop). Aero geometry is cached by the set of aero parts, so removing a non-aero part costs only a mass + physics rerun (milliseconds).
9. **Report** (`report.py`): the JSON schema is in §4. Per part: function, why, necessity, material, specs, interfaces + tolerances, operating point + margin, regulations, confidence, and a source tag on every value.
10. **Edits + env** (`edits.py`, `env.py`): JSON ops `remove`, `swap` (replace the BOM entry, optionally with a new CAD file), `set` (a field path), `scale` (mesh scale on part ids). `DroneBench(design).reset()/step(ops)` works like Gymnasium: reward = change in score, and invalid edits are rejected with reasons. Agent-supplied specs go through the plausibility checks (`plausibility.py`) and are clamped and flagged if they fail.
11. **CLI + viewer** (`cli.py`, `viewer/`): `dronebench score|parts|view|edit|baseline`. `view` writes one self-contained HTML file: a three.js 3D model with parts colored by necessity, click-for-part-report, explode slider, a before/after mode for two designs, and the score curve from runs.jsonl.
12. **Demo** (`demo/TASK.md`): the brief Claude Code gets, plus `baseline` (random search) for a comparison line.

## 2. Modules, owners, signatures

All types live in `dronebench/model.py` (frozen contract; add optional fields only, never rename).

| Module | Owner | Public functions |
|---|---|---|
| bom.py | A | `load_bom(design_dir) -> Design` (parts without meshes, mission, frame) |
| ingest.py | A | `load_design(design_dir) -> Design` (bom + meshes + volume/area/bbox/cg, frame normalised) |
| graph.py | A | `build_graph(design) -> networkx.Graph` (node = part id; edge attrs: `kind` physical/electrical, `joint`, `gap_mm`, `tolerance`, `source`) |
| geometry.py | A | `extract_geometry(design) -> AeroGeometry`; `mass_properties(design) -> MassProps` (uses Part.mass_g) |
| materials.py | C | `MATERIALS: dict[str, Material]`; `resolve_material(part) -> Material`; `part_mass_g(part) -> (grams, source)` |
| components.py | C | `lookup(part_number) -> dict or None` (specs + material sub-parts for known COTS parts) |
| physics/*.py | B | `physics.run(PhysicsInput) -> PhysicsResult` (plus airfoil, aero, propulsion, structure, performance submodules) |
| assemble.py | E | `build_physics_input(design, geom, mass, graph) -> PhysicsInput` |
| gate.py | E | `evaluate_gate(design, graph, geom, pres) -> GateResult` |
| score.py | E | `compute_score(preset, pres, gate) -> float` |
| necessity.py | E | `removal_test(design, preset, base) -> dict[id, NecessityResult]` |
| regulations.py | E | `part_regulations(part, design, pres) -> list[str]`, `aircraft_regulations(design, pres) -> list[str]` |
| plausibility.py | E | `check_specs(part) -> list[(field, old, new, reason)]` (clamps in place) |
| functions.py | E | `describe(part, design, graph) -> (function, why)` |
| report.py | E | `build_report(...) -> dict` (schema §4) |
| pipeline.py | E | `evaluate(design_dir, preset=None, necessity=True) -> dict` (the report) |
| edits.py | E | `apply_edits(design_dir, ops, out_dir) -> (Path, list[str] errors)` |
| env.py | E | `class DroneBench: reset() -> obs; step(ops) -> (obs, reward, done, info)` |
| cli.py | D | argparse entry `dronebench` |
| viewer/ | D | `render_viewer(design_dirs, reports, out_html, runs_jsonl=None)` |
| fixtures/ | C | design folders + generators |

## 3. Design folder format

```
design/
  cad/            # STL/OBJ/GLB/3MF/STEP, one body per part
  bom.yaml        # list of parts (below)
  mission.yaml    # preset, payload ids, optional cruise_kmh, frame
```
`bom.yaml`:
```yaml
parts:
  - id: wing_L1
    type: wing            # PART_TYPES in model.py
    role: wing            # AERO_ROLES
    cad: wing_L1.stl
    part_number: Falcon V2 Wing1 L
    material: {name: LW-PLA, infill: 0.0, walls: 1, orientation: vase}
    specs: {}
  - id: motor_L
    type: motor
    cad: motor_L.stl
    part_number: BrotherHobby Avenger 2816 620KV
    specs: {kv: 620, rm_ohm: 0.061, i0_a: 0.9, imax_a: 35, mass_g: 128}
    connects: [esc_L, prop_L]
  - id: screws_motor_L
    type: fastener
    group: motor_L_mount
    qty: 4
    specs: {size: M3x8, mass_g: 0.8}
```
`mission.yaml`:
```yaml
preset: endurance
payload_ids: [camera]
cruise_kmh: 55        # optional
frame: {units: mm, forward: "-x", up: "+z"}
```

## 4. report.json schema

```json
{
  "design": "falcon_v2", "preset": "endurance", "dronebench_version": "0.1.0",
  "score": {"preset": "endurance", "value": 94.2, "unit": "min", "error_band": 0.15},
  "gate": {"pass": true, "checks": [{"name": "static_margin", "value": 0.11, "limit": [0.05, 0.2], "pass": true, "detail": ""}]},
  "aircraft": {"mass_g": 3480, "cg_m": [0.31, 0, 0.02], "span_m": 2.093, "wing_area_m2": 0.4514, "aspect_ratio": 9.7,
               "mac_m": 0.23, "stall_kmh": 35, "cruise_kmh": 55, "vmax_kmh": 110, "climb_ms": 4.1,
               "endurance_min": 94, "range_km": 80, "wh_per_km": 1.43, "static_margin": 0.11,
               "tail_vh": 0.5, "tail_vv": 0.035, "spar_sf": 2.3, "cd0": 0.03, "ld_max": 13.5,
               "sources": {"span_m": "cad", "stall_kmh": "computed"}},
  "regulations": ["FAA Part 107: ..."],
  "parts": [{
    "id": "motor_L", "part_number": "...", "type": "motor", "role": null, "qty": 1,
    "function": "propulsion", "why": "drives prop_L; 1 of 2 thrust sources",
    "necessity": "required", "removal_test": "gate fail: climb 0.4 m/s < 2.0", "score_delta_if_removed": null,
    "mass_g": 128, "cg_m": [0.3, -0.25, 0.0],
    "specs": {"kv": 620}, "material": {"name": "...", "kind": "...", "sub": {"bell": "Al 6061"}, "infill": null, "walls": null, "orientation": null, "density_kg_m3": 0, "source": "catalog"},
    "interfaces": [{"mates": "nacelle_L", "kind": "physical", "joint": "bolt pattern 4xM3", "gap_mm": 0.0, "tolerance": "±0.1 mm", "source": "inferred"}],
    "operating_point": {"cruise_a": 4.2, "limit_a": 35, "margin": 8.3},
    "regulations": [],
    "confidence": 1.0,
    "sources": {"mass_g": "bom", "material": "catalog"}
  }],
  "warnings": [], "timing_ms": {"total": 350}
}
```

## 5. Fixtures (test designs)

| Design | What it tests |
|---|---|
| `fixtures/falcon_v2` | Parametric Falcon V2 replica (2093 mm, 0.4514 m², NACA 4411→3411, twin 2816 620 kV, 6S Li-ion, 7 servos, 10/8 mm CF spars). Seeded with waste: nose ballast, duplicate GPS, spar doubler, cosmetic fairings. Calibration + necessity demo. |
| `fixtures/trainer` | 1.2 m single-motor trainer, 3S LiPo, 4 servos. A normal, passing design. |
| `fixtures/bad_tail` | Trainer with a tail that's too small and CG too far aft. Must fail the gate with clear reasons. |
| `fixtures/step_mix` | Trainer with some parts as STEP (servo from FreeCAD-library). Tests the STEP ingest path. |
| `fixtures/third_party` | A real downloaded open design (OpenGeoDrone STLs, or any public fixed-wing STL set) with a hand-written BOM. Tests messy real-world CAD. |

## 6. Acceptance criteria (the "works exactly as needed" bar)

1. `dronebench score fixtures/falcon_v2` runs in < 20 s with necessity, < 1 s without.
2. Falcon V2 at ~3.5 kg: Wh/km at 55 km/h within 1.36–1.84 (1.6 ± 15%), stall 30–40 km/h, gate passes.
3. Every part in every report has: function, why, necessity, material (never empty), specs, interfaces, sources. Nothing is left as null without a reason.
4. Seeded waste in falcon_v2 (ballast, duplicate GPS, fairing) comes back `unnecessary`. Motors, battery, main spar, wing panels, FC and elevator servo come back `required`.
5. `bad_tail` fails the gate on static_margin and/or tail_volume, with values and limits.
6. STEP and STL both ingest; the third-party design produces a full report.
7. `env.step([{"op":"remove","id":"ballast_nose"}])` raises the score; `step([{"op":"remove","id":"motor_L"}])` gives score 0 and a reason.
8. `dronebench view` produces an HTML file that loads, shows all parts in 3D colored by necessity, and shows the part panel on click (checked in a browser).
9. Implausible agent specs (a 5 g motor rated at 2 kW, 400 Wh/kg LiPo) get clamped and flagged.
10. A short Claude Code loop on `demo/TASK.md` improves the falcon_v2 score over several steps without breaking the gate.
