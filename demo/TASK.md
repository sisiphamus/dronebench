# DroneBench task: improve the design

You are improving a small electric fixed-wing UAV. The design lives in `design/` (a copy of
`fixtures/falcon_v2`). DroneBench scores it; you change it.

## Goal

Raise the preset score of `design/` (the preset is in `design/mission.yaml`, e.g. `endurance` =
minutes of flight) **without failing the gate**. A design that fails any gate check scores 0.
The mission payload (`payload_ids` in `mission.yaml`, marked `locked`) must stay.

## What you may edit

- `design/bom.yaml`: parts, part numbers, specs, materials, `connects`, `qty`.
- `design/cad/*`: meshes, but only through `dronebench edit` (scale / move) or by regenerating them.
- `design/params.yaml` + its rebuild script, if the design has one (the Falcon V2 is parametric:
  change a parameter, rerun the script, then re-score).

## What you may not edit

- Anything under `dronebench/` (the benchmark source).
- `design/mission.yaml` (preset, payload, cruise speed and frame are fixed).
- `locked` flags, or the `sources` tags DroneBench writes.

Specs you type in for a new or swapped part are tagged `agent_supplied`. They go through plausibility
checks (energy density, power per gram, torque per gram, ...); implausible values are clamped to the
limit and flagged `agent_supplied_clamped` in the report. Inventing a 400 Wh/kg battery does not help.

## Commands

```bash
dronebench score design --summary --runs runs.jsonl --label claude   # score, log the step
dronebench score design --out report.json                            # full report (JSON)
dronebench parts design --table                                      # per-part table: function, necessity, margin
dronebench edit design --ops ops.json --out design_next              # apply JSON ops to a copy
dronebench view design --runs runs.jsonl --out viewer.html           # 3D view, score curve
```

Edit ops (`ops.json` is a JSON list):

```json
[{"op": "remove", "id": "ballast_nose"},
 {"op": "swap", "id": "prop_L", "entry": {"part_number": "APC 12x6E"}},
 {"op": "set", "id": "wing_L1", "path": "material.infill", "value": 0.0},
 {"op": "scale", "ids": ["wing_L1", "wing_R1"], "factors": [1.0, 1.05, 1.0], "about": "origin"},
 {"op": "move", "ids": ["battery"], "offset_m": [-0.01, 0, 0]},
 {"op": "add", "entry": {"id": "gps2", "type": "gps", "specs": {"mass_g": 20}}}]
```

Frame: x aft, y right wing, z up, metres. `edit` writes a new folder and never touches the input.

## Loop protocol (one change per step)

1. Read the report: `dronebench score design --summary`, then `dronebench parts design --table`.
   Parts marked `unnecessary` are candidates for removal; low `margin` shows what limits you.
2. Pick ONE change and say why in one line (what you expect to move, and which gate margin could suffer).
3. Apply it: `dronebench edit design --ops ops.json --out design_next`.
4. Score it: `dronebench score design_next --summary --runs runs.jsonl --label claude`.
5. Keep or revert: if the gate passes and the score went up, replace `design` with `design_next`
   (`rm -rf design && mv design_next design`); otherwise delete `design_next`.
6. Log it: the `--runs` flag already appended the step; add a one-line note of what you tried.

Stop after **10 steps** (or earlier if three changes in a row fail to improve the score).
Finish with: the starting score, the final score, the list of kept changes, and the final gate summary.
