# DroneBench

An environment for small electric fixed-wing drones where designs are **flown**, not estimated. It has two halves that
share one simulator:

- **Design environment.** An agent (an LLM, a black-box optimiser, an RL policy) edits an aircraft. DroneBench builds the
  aerodynamic model from the geometry, flies the design in a 6-degree-of-freedom simulator with wind and turbulence, and keeps
  or rejects the edit with the reason. Reward is the change in a mission score that was measured in flight.
- **Flight environment.** A Gymnasium environment for learning to fly a given design: continuous control of aileron, elevator,
  rudder and throttle at 50 Hz, about 4000 steps per second.

```
edit  ->  geometry  ->  AeroSandbox aero model  ->  6-DOF simulator  ->  autopilot flies climb, orbits, legs
 ^                      (vortex lattice, NeuralFoil)   (numba RK4)          in calm, light and strong wind + turbulence
 |                                                                                  |
 +---- kept or rejected, with the check, value and limit  <-- gate  <-- measured endurance, range, climb, loads
```

What it will and will not tell you is in [VALIDATION.md](VALIDATION.md). Short version: the physics is checked against theory,
against two independent aerodynamic methods, and against one manufacturer figure (without tuning). It has not been compared with
measured flight logs, and it only covers conventional wing + tail layouts.

## Quick start

```bash
uv venv && uv pip install -e .                                   # adds AeroSandbox, numba, gymnasium
dronebench init runs/demo --fixture falcon_v2 --steps 12         # a workspace: design, brief, step log
dronebench serve runs/demo --open                                # live dashboard
dronebench run runs/demo --agent claude                          # in another terminal: Claude Code iterates
```

Other agents on the same workspace: `--agent cma` (CMA-ES over the design parameters), `--agent random-params`, `--agent random` (random part edits).
Or drive it by hand, which is what an agent does:

```bash
dronebench status runs/demo
dronebench try   runs/demo --ops ops.json --note "ballast is dead weight"     # edit parts, fly, accept or reject
dronebench build runs/demo --set wing.span_mm=2300 --set htail.area_m2=0.09  # change parameters, rebuild, fly
dronebench score fixtures/falcon_v2 --flown --summary                          # fly one design folder
```

## As a Gymnasium environment

```python
import gymnasium as gym, dronebench.envs

# learn to design: each step edits parameters, rebuilds, flies (about 20 s), reward = change in score
env = gym.make("DroneBench/Design-v0", fixture="falcon_v2", max_steps=20)

# learn to fly: continuous control in wind and turbulence
env = gym.make("DroneBench/Flight-v0", design="fixtures/falcon_v2", task="orbit")   # or "line", "hold"
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.unwrapped.expert_action()            # the design-aware cascaded autopilot, as a reference policy
```

`python -m dronebench.train --task orbit` trains PPO (stable-baselines3, optional) and writes a learning curve with the autopilot,
zero-action and random policies as reference lines.

## What gets flown, and what is measured

| Segment | Weather | Measured |
|---|---|---|
| Full-throttle climb at best-climb speed | calm | rate of climb |
| Orbit, 150 m radius, 130 s | calm / 3 m/s + light turbulence / 6 m/s + moderate turbulence, at 100 / 70 / 40% charge | battery power, altitude and speed holding, bank, load factor |
| Upwind and downwind legs | the same three | energy per ground kilometre |

Endurance is usable battery energy divided by mean flown orbit power. Range is usable energy divided by flown energy per
kilometre. Weather and seeds are the same for every design, so scores compare. The gate checks static margin, stall margin,
climb, that every segment is completed, that peak load factor in turbulence stays under the spar's limit, and that the aircraft does
not stall, plus the structure and systems checks: spar, current limits, wiring, every part attached, continuous fuselage, printed shells with ribs.

An aircraft that cannot be trimmed, cannot be flown by the autopilot, or has no tail the model can represent fails with the
reason, instead of receiving a made-up score.

## What an agent can change

- **`try`**: JSON ops on the parts list and meshes: `remove`, `swap` (new part number and specs), `set`, `scale`, `move`, `add`. Specs an
  agent types for a new part are tagged `agent_supplied` and checked for plausibility (battery Wh/kg, power per gram); implausible values are clamped.
- **`build`**: for parametric designs (`falcon_v2`) the builder's knobs: span, chords, taper break, dihedral, tail area and arm, fin area, shell
  thickness, motor, prop and battery choice. The whole aircraft, CAD included, is regenerated.

Locked parts (the mission payload) and `mission.yaml` cannot be changed.

## Calibrating to a real aircraft

DroneBench has one explicit tuning knob, a drag factor for what CAD cannot see (seams, hinge gaps, print roughness). With a few
steady-flight measurements of battery power:

```bash
dronebench calibrate fixtures/falcon_v2 --point 15.3:92 --point 18:124      # airspeed m/s : battery watts (use your own log)
```

## Fixtures

| Name | What | Parts |
|---|---|---|
| `falcon_v2` | Titan Falcon V2 replica, parametric, with seeded dead weight | 80 |
| `trainer` | small trainer | 42 |
| `bad_tail` | trainer with an undersized tail (statically unstable; a repair task) | 42 |
| `step_mix` | trainer with STEP parts mixed in | 42 |
| `third_party` | real downloaded CAD, no tail (flying wing); fails with a message | 48 |

## Dashboard

`dronebench serve` is read-only and polls the workspace, so it works while an agent runs in another process: the 3D aircraft (needed, helps,
removable; what changed against the baseline), the score curve with rejected steps marked, every attempt with the reason, and every gate check
with its headroom, including the flown ones. `dronebench export WS --out site/demo` writes a static copy for hosting.

## Layout

```
dronebench/aero.py          AeroSandbox model from geometry, coefficient tables (cached by geometry hash)
dronebench/flight.py        6-DOF simulator, powertrain, trim; _dyn.py is the compiled inner loop
dronebench/autopilot.py     cascaded autopilot with gains from the design's own derivatives
dronebench/mission.py       flown climb / orbit / legs, performance sweep, flight checks
dronebench/flown.py         merges the flown evaluation into the report
dronebench/envs.py          Gymnasium environments
dronebench/workspace.py     the agent loop: init, try, build, status
dronebench/server.py viewer/   live dashboard
dronebench/ (the rest)      ingest, part graph, geometry, closed-form physics, structure and systems gate, removal test
fixtures/  validation/  tests/
VALIDATION.md               what was checked and what is open
```
