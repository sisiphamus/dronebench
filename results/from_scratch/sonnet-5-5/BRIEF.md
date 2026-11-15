# Brief: design an endurance drone from nothing, then improve it with the test bench

You are an aircraft engineer. There is no starting aircraft. You write the first design yourself, test it in
DroneBench, and keep improving it from what the tests tell you. You are being observed to see how a design
progresses when an agent has only the tests to learn from.

## The mission

A small electric fixed-wing drone that carries one camera and stays in the air as long as possible.

- Score: minutes of flight (preset `endurance`), measured by flying the design in a simulator.
- The camera is the payload. It must be a part with `id: camera` and `type: camera`, and `mission.yaml` must list
  `payload_ids: [camera]`.
- Takeoff mass limit 7.0 kg (`max_mass_kg: 7.0`). Cruise speed in the mission: 55 km/h (`cruise_kmh: 55`), you may
  change it.
- Conventional layout only: one main wing, a horizontal tail, a vertical tail, a fuselage. No V-tail or flying wing.

## What DroneBench does with a design

1. Reads the folder (`DESIGN_FORMAT.md` in this folder describes it: meshes in `cad/`, `bom.yaml`, `mission.yaml`).
2. Builds an aerodynamic model from the geometry (AeroSandbox vortex lattice and airfoil data).
3. Runs the **gate**: structure and systems checks (spar strength, motor and battery current, wiring, every part
   attached to the airframe, continuous fuselage, battery fit, takeoff mass, nothing passing through a propeller) and
   flight checks (static margin, stall margin, climb, tail volume, servo torque).
4. **Flies** the aircraft in a 6-degree-of-freedom simulator with a stock autopilot: a full-throttle climb, orbits and
   upwind and downwind legs, in calm air, in light wind and in strong wind with turbulence. The score is measured from
   those flights. A design that fails the gate gets no score, only the reasons.

## Commands

Run everything from this folder (`work/`). `dronebench` and `python` (with numpy and trimesh) are on your PATH.

```
dronebench score designs/v01 --flown --summary --runs runs.jsonl --label sonnet   # test one design folder
dronebench score designs/v01 --flown --pretty --out report.json                    # the full report, as JSON
dronebench parts designs/v01 --table                                               # every part, what it is for, whether the removal test says it is needed
dronebench init ws --fixture designs/v07 --steps 25                                # start a workspace from a design of yours (see phase 2)
dronebench status ws        dronebench try ws --ops ops.json --note "why"          # inside a workspace
```

The catalog of real parts (motors, props, servos, ESCs, batteries, cameras, flight controllers) is in
`../code/dronebench/components.py`. A part with a catalog `part_number` takes its real specs and mass. You may also
write your own specs; implausible ones are clamped.

## How to work

**Phase 1: get a design that passes.** Write a Python generator (`make_design.py`, with the dimensions at the top) that
produces a complete design folder with trimesh/numpy meshes plus the parts list. Each version goes in its own folder
`designs/v01`, `designs/v02`, ... Never overwrite or delete an earlier version: the whole history is the record of
how you progressed. Score each one with the command above. When it fails, read which check failed and by how much,
change one or two things for a stated reason, and write the next version. You may write at most 25 versions in this
phase. Score every version you write.

**Phase 2: improve it.** As soon as a version passes the gate, you may continue with it as a workspace
(`dronebench init ws --fixture designs/vNN --steps 25`) and use `try` with edit operations (the ops are described in
`ws/TASK.md`, which `init` writes). The workspace keeps a design only if it passes and scores at least as well. Use all
25 steps unless five attempts in a row fail to help.

**Keep `LOG.md`.** After every scored version or step, add one line: what you changed, why, what you expected, and
what the test said. Be honest: if a change made things worse, say so.

## Rules

- You have no example aircraft. Do not look for one. Use only this folder and `../code/` (the program and its parts
  catalog). Do not open anything else on this computer.
- Do not edit anything under `../code/`.
- Specs you invent are checked for plausibility. Do not try to get around the checks; fix the aircraft.
- Finish with a short report: first passing version and its score, final score, the five changes that helped most,
  the changes that did not work, and what you would try next.
